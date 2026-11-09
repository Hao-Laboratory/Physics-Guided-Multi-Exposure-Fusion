"""ICVL data loading for the released three-exposure training protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def scene_name_hash(scene_names: list[str]) -> str:
    return hashlib.sha256("\n".join(scene_names).encode("utf-8")).hexdigest()


def _to_chw(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(
        np.ascontiguousarray(array.transpose(2, 0, 1), dtype=np.float32)
    )


def _apply_augmentation(
    arrays: list[np.ndarray],
    horizontal_flip: bool,
    vertical_flip: bool,
    quarter_turns: int,
) -> list[np.ndarray]:
    result = []
    for array in arrays:
        if horizontal_flip:
            array = np.flip(array, axis=1)
        if vertical_flip:
            array = np.flip(array, axis=0)
        if quarter_turns:
            array = np.rot90(array, k=int(quarter_turns), axes=(0, 1))
        result.append(np.ascontiguousarray(array))
    return result


class ExposureProtocol:
    """Load the public configuration and deterministic training schedule."""

    def __init__(
        self,
        training_config_path: str | Path,
        data_root: str | Path | None = None,
    ) -> None:
        self.training_config_path = Path(training_config_path).resolve()
        self.project_root = self.training_config_path.parent.parent
        self.config = load_json(self.training_config_path)

        protocol_path = self._resolve(self.config["exposure_protocol"])
        self.exposure_config = load_json(protocol_path)
        self.schedule_path = self._resolve(self.config["training_schedule"])
        self.schedule = np.load(self.schedule_path, allow_pickle=False)

        configured_root = self.config["data"]["root"]
        if data_root is None:
            root_path = Path(configured_root)
            if not root_path.is_absolute():
                root_path = self.project_root / root_path
        else:
            root_path = Path(data_root)
        self.data_root = root_path.resolve()
        self.gt_filename = self.config["data"]["gt_filename"]
        self.max_dn = float(self.config["data"]["max_dn"])
        self.hdr_scale = float(self.config["data"]["hdr_scale"])
        self.exposure_by_ev = {
            float(item["ev"]): dict(item)
            for item in self.exposure_config["exposure_pool"]
        }
        self._validate()

    def _resolve(self, path_value: str) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _validate(self) -> None:
        required_arrays = {
            "epochs",
            "patch_size",
            "scene_names",
            "training_combinations",
            "heldout_test_combinations",
            "combo_ids",
            "scene_order",
            "crop_top",
            "crop_left",
            "horizontal_flip",
            "vertical_flip",
            "rotation_quarter_turns",
        }
        missing = sorted(required_arrays - set(self.schedule.files))
        if missing:
            raise KeyError(f"training schedule is missing arrays: {missing}")
        epochs = int(self.schedule["epochs"][0])
        scene_count = len(self.schedule["scene_names"])
        expected_shape = (epochs, scene_count)
        for name in (
            "combo_ids",
            "scene_order",
            "crop_top",
            "crop_left",
            "horizontal_flip",
            "vertical_flip",
            "rotation_quarter_turns",
        ):
            if self.schedule[name].shape != expected_shape:
                raise ValueError(
                    f"schedule array {name} has shape {self.schedule[name].shape}; "
                    f"expected {expected_shape}"
                )
        configured_combinations = np.asarray(
            self.exposure_config["training_combinations"], dtype=np.float32
        )
        if not np.array_equal(
            configured_combinations, self.schedule["training_combinations"]
        ):
            raise ValueError("training combinations differ from the shipped schedule")
        for ev, item in self.exposure_by_ev.items():
            if abs(float(item["ratio"]) - 2.0**ev) > 1e-12:
                raise ValueError(f"invalid exposure ratio for EV {ev:g}")

    @property
    def train_scene_names(self) -> list[str]:
        return [str(name) for name in self.schedule["scene_names"].tolist()]

    @property
    def validation_combinations(self) -> list[tuple[float, ...]]:
        return [
            tuple(float(value) for value in combination)
            for combination in self.exposure_config["validation_combinations"]
        ]

    def training_combination(self, combo_id: int) -> tuple[float, ...]:
        values = self.schedule["training_combinations"][int(combo_id)]
        return tuple(float(value) for value in values)

    def split_scene_names(self, split: str) -> list[str]:
        split_config = self.config["data"]["splits"][split]
        split_root = self.data_root / split_config["directory"]
        if not split_root.is_dir():
            raise FileNotFoundError(f"dataset split not found: {split_root}")
        # Ignore result/visualization directories that users may keep beside
        # the scene folders. A valid scene is identified by its HDR target.
        names = sorted(
            entry.name
            for entry in split_root.iterdir()
            if entry.is_dir() and (entry / self.gt_filename).is_file()
        )
        expected_count = split_config.get("expected_scene_count")
        if expected_count is not None and len(names) != int(expected_count):
            raise RuntimeError(
                f"{split} contains {len(names)} scenes; expected {expected_count}"
            )
        expected_hash = split_config.get("scene_name_sha256")
        if expected_hash and scene_name_hash(names) != expected_hash:
            raise RuntimeError(f"{split} scene names do not match the released split")
        return names


class _BaseDataset(Dataset):
    def __init__(self, protocol: ExposureProtocol) -> None:
        self.protocol = protocol

    def _load_crop(
        self,
        scene_dir: Path,
        evs: tuple[float, ...],
        top: int,
        left: int,
        patch_size: int,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        quarter_turns: int = 0,
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        gt_path = scene_dir / self.protocol.gt_filename
        gt = np.load(gt_path, mmap_mode="r")
        if gt.ndim != 3 or gt.shape[-1] != 9:
            raise ValueError(f"expected HxWx9 ground truth at {gt_path}")
        rows = slice(int(top), int(top) + int(patch_size))
        columns = slice(int(left), int(left) + int(patch_size))
        label = np.asarray(gt[rows, columns], dtype=np.float32) / self.protocol.hdr_scale

        raw_arrays = []
        normalized_arrays = []
        for ev in evs:
            exposure = self.protocol.exposure_by_ev[float(ev)]
            raw_path = scene_dir / exposure["filename"]
            raw = np.load(raw_path, mmap_mode="r")
            if raw.shape != gt.shape:
                raise ValueError(f"shape mismatch: {raw_path} and {gt_path}")
            raw01 = np.asarray(raw[rows, columns], dtype=np.float32) / self.protocol.max_dn
            normalized = (
                raw01 / float(exposure["ratio"]) / self.protocol.hdr_scale
            ).astype(np.float32, copy=False)
            raw_arrays.append(raw01)
            normalized_arrays.append(normalized)

        arrays = _apply_augmentation(
            raw_arrays + normalized_arrays + [label],
            horizontal_flip,
            vertical_flip,
            quarter_turns,
        )
        count = len(evs)
        return arrays[:count], arrays[count : 2 * count], arrays[-1]

    @staticmethod
    def _format_sample(
        raw_arrays: list[np.ndarray],
        normalized_arrays: list[np.ndarray],
        label: np.ndarray,
        evs: tuple[float, ...],
        metadata: dict,
    ) -> dict:
        return {
            "raw": torch.stack([_to_chw(array) for array in raw_arrays]),
            "normalized": torch.stack(
                [_to_chw(array) for array in normalized_arrays]
            ),
            "ev": torch.tensor(evs, dtype=torch.float32),
            "mask": torch.ones(len(evs), dtype=torch.bool),
            "label": _to_chw(label),
            **metadata,
        }


class ScheduledTrainingDataset(_BaseDataset):
    """One deterministic crop and exposure combination per scene and epoch."""

    def __init__(self, protocol: ExposureProtocol) -> None:
        super().__init__(protocol)
        self.scene_names = protocol.train_scene_names
        split_dir = protocol.config["data"]["splits"]["train"]["directory"]
        self.root = protocol.data_root / split_dir
        self.patch_size = int(protocol.schedule["patch_size"][0])
        self.epoch_index = 0

    def __len__(self) -> int:
        return len(self.scene_names)

    def set_epoch(self, epoch: int) -> None:
        index = int(epoch) - 1
        if not 0 <= index < int(self.protocol.schedule["epochs"][0]):
            raise ValueError("epoch is outside the shipped schedule")
        self.epoch_index = index

    def __getitem__(self, scene_index: int) -> dict:
        scene_index = int(scene_index)
        schedule = self.protocol.schedule
        combo_id = int(schedule["combo_ids"][self.epoch_index, scene_index])
        evs = self.protocol.training_combination(combo_id)
        raw, normalized, label = self._load_crop(
            self.root / self.scene_names[scene_index],
            evs,
            int(schedule["crop_top"][self.epoch_index, scene_index]),
            int(schedule["crop_left"][self.epoch_index, scene_index]),
            self.patch_size,
            bool(schedule["horizontal_flip"][self.epoch_index, scene_index]),
            bool(schedule["vertical_flip"][self.epoch_index, scene_index]),
            int(schedule["rotation_quarter_turns"][self.epoch_index, scene_index]),
        )
        return self._format_sample(
            raw,
            normalized,
            label,
            evs,
            {
                "scene": self.scene_names[scene_index],
                "scene_index": scene_index,
                "combo_id": combo_id,
            },
        )


class ScheduledEpochSampler(Sampler[int]):
    def __init__(self, dataset: ScheduledTrainingDataset, limit: int | None = None) -> None:
        self.dataset = dataset
        self.limit = None if limit is None else int(limit)
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")

    def __iter__(self):
        order = self.dataset.protocol.schedule["scene_order"][
            self.dataset.epoch_index
        ]
        if self.limit is not None:
            order = order[: self.limit]
        return iter(int(index) for index in order)

    def __len__(self) -> int:
        return len(self.dataset) if self.limit is None else min(self.limit, len(self.dataset))


class ValidationDataset(_BaseDataset):
    """Evaluate every validation scene on each fixed validation combination."""

    def __init__(self, protocol: ExposureProtocol, limit_scenes: int | None = None) -> None:
        super().__init__(protocol)
        self.scene_names = protocol.split_scene_names("val")
        if limit_scenes is not None:
            self.scene_names = self.scene_names[: int(limit_scenes)]
        split_dir = protocol.config["data"]["splits"]["val"]["directory"]
        self.root = protocol.data_root / split_dir
        self.combinations = protocol.validation_combinations
        self.patch_size = int(protocol.config["training"]["patch_size"])

    def __len__(self) -> int:
        return len(self.scene_names) * len(self.combinations)

    def __getitem__(self, index: int) -> dict:
        combo_count = len(self.combinations)
        scene_index = int(index) // combo_count
        combo_id = int(index) % combo_count
        scene_name = self.scene_names[scene_index]
        scene_dir = self.root / scene_name
        gt = np.load(scene_dir / self.protocol.gt_filename, mmap_mode="r")
        height, width = gt.shape[:2]
        top = (height - self.patch_size) // 2
        left = (width - self.patch_size) // 2
        evs = self.combinations[combo_id]
        raw, normalized, label = self._load_crop(
            scene_dir, evs, top, left, self.patch_size
        )
        return self._format_sample(
            raw,
            normalized,
            label,
            evs,
            {"scene": scene_name, "scene_index": scene_index, "combo_id": combo_id},
        )
