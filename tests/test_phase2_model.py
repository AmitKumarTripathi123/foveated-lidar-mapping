"""Unit tests for FoveatedPointSegNet."""
import unittest
import torch
import numpy as np

from phase2.models.point_seg_net import FoveatedPointSegNet


class TestPhase2Model(unittest.TestCase):
    def setUp(self):
        self.model = FoveatedPointSegNet(in_channels=4, num_classes=4)

    def test_forward_pass_shape(self):
        pts = torch.randn(500, 4)
        logits = self.model(pts)
        self.assertEqual(logits.shape, (500, 4))

    def test_empty_input(self):
        pts = torch.empty(0, 4)
        logits = self.model(pts)
        self.assertEqual(logits.shape, (0, 4))

    def test_single_point(self):
        pts = torch.randn(1, 4)
        logits = self.model(pts)
        self.assertEqual(logits.shape, (1, 4))

    def test_predict_probabilities_and_confidence(self):
        pts = torch.randn(200, 4)
        out = self.model.predict(pts)
        probs = out["probabilities"]
        preds = out["predicted_class"]
        conf = out["confidence"]

        self.assertEqual(probs.shape, (200, 4))
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0001).all())
        sum_p = torch.sum(probs, dim=-1)
        np.testing.assert_allclose(sum_p.numpy(), np.ones(200), atol=1e-5)
        self.assertEqual(len(preds), 200)
        self.assertEqual(len(conf), 200)
        self.assertTrue((conf >= 0.0).all() and (conf <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
