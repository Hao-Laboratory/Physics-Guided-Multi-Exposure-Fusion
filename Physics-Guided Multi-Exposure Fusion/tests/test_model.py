import unittest

import torch

from models import ReconstructionLoss, build_model


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = build_model().eval()
        self.raw = torch.rand(1, 3, 9, 32, 40)
        ev = torch.tensor([[-2.0, 0.0, 2.0]])
        self.normalized = self.raw / (2.0**ev[:, :, None, None, None]) / 2.5712735652923584
        self.ev = ev
        self.mask = torch.ones((1, 3), dtype=torch.bool)

    def test_output_shape_and_range(self):
        with torch.no_grad():
            prediction = self.model(self.raw, self.normalized, self.ev, self.mask)
        self.assertEqual(prediction.shape, (1, 9, 32, 40))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertGreaterEqual(float(prediction.min()), 0.0)
        self.assertLessEqual(float(prediction.max()), 1.0)

    def test_exposure_permutation_invariance(self):
        permutation = torch.tensor([2, 0, 1])
        with torch.no_grad():
            reference = self.model(self.raw, self.normalized, self.ev, self.mask)
            permuted = self.model(
                self.raw[:, permutation],
                self.normalized[:, permutation],
                self.ev[:, permutation],
                self.mask[:, permutation],
            )
        self.assertTrue(torch.allclose(reference, permuted, atol=2e-6, rtol=1e-6))

    def test_loss_backward(self):
        self.model.train()
        prediction = self.model(self.raw, self.normalized, self.ev, self.mask)
        loss = ReconstructionLoss()(prediction, torch.rand_like(prediction))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
