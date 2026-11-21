"""Behavioral checks for the CNN's leakage and checkpoint boundaries."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from cnn import build_model, normalization_stats, prepare_batch, save_checkpoint, load_model
from train_cnn import recover_best, merge_resource_peaks
from verify_cnn import audit_near_ties


class CNNChecks(unittest.TestCase):
    def test_reload_audit_accepts_only_a_documented_top_two_near_tie(self):
        rows = audit_near_ties(np.array([0, 1]), np.array([1, 1]), np.array([[2., 2.0002, -4.]]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['position'], 0)
        self.assertLess(rows[0]['logit_gap'], 0.001)

    def test_reload_audit_rejects_high_confidence_or_unrelated_label_changes(self):
        with self.assertRaises(AssertionError):
            audit_near_ties(np.array([0]), np.array([1]), np.array([[2., 2.1, -4.]]))
        with self.assertRaises(AssertionError):
            audit_near_ties(np.array([2]), np.array([1]), np.array([[2., 2.0002, -4.]]))

    def test_resume_restores_committed_best_instead_of_newer_uncommitted_export(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'best.pt'
            save_checkpoint(path, {'epoch': 5, 'validation_f1_macro': 0.9, 'model_state': {'w': torch.tensor([9.])}})
            committed = {'epoch': 3, 'validation_f1_macro': 0.8, 'model_state': {'w': torch.tensor([3.])}}
            result = recover_best({'best_checkpoint': committed}, path)
            self.assertEqual(result['epoch'], 3)
            exported = torch.load(path, weights_only=True)
            self.assertEqual(exported['epoch'], 3)
            torch.testing.assert_close(exported['model_state']['w'], torch.tensor([3.]))

    def test_resume_keeps_previous_resource_peaks(self):
        previous = {'rss_mib': 800., 'allocated_mib': 900., 'reserved_mib': 1000., 'complete': True}
        current = {'rss_mib': 100., 'allocated_mib': 200., 'reserved_mib': 1100., 'complete': True}
        merged = merge_resource_peaks(previous, current)
        self.assertEqual(merged, {'rss_mib': 800., 'allocated_mib': 900., 'reserved_mib': 1100., 'complete': True})

    def test_normalization_uses_only_training_indices(self):
        images = np.zeros((3, 32, 32, 3), dtype=np.uint8)
        images[1] = 255
        images[2] = 100  # held-out image must not influence statistics
        mean, std = normalization_stats(images, np.array([0, 1]))
        np.testing.assert_allclose(mean, [0.5, 0.5, 0.5], atol=1e-7)
        np.testing.assert_allclose(std, [0.5, 0.5, 0.5], atol=1e-7)

    def test_preprocessing_preserves_input_and_normalizes_per_image(self):
        raw = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)
        raw[1] = 255
        original = raw.clone()
        expected = torch.stack([torch.full((3, 32, 32), -1.), torch.ones(3, 32, 32)])
        for augment in [False, True]:
            result = prepare_batch(raw, [0.5]*3, [0.5]*3, augment=augment)
            torch.testing.assert_close(result, expected)
            self.assertTrue(torch.isfinite(result).all())
        torch.testing.assert_close(raw, original)

    def test_real_training_step_and_checkpoint_reload_preserve_eval_predictions(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = build_model()
        x = torch.randn(4, 3, 32, 32)
        y = torch.tensor([0, 1, 2, 3])
        before = model.fc.weight.detach().clone()
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.fc.weight))
        model.eval()
        bn_before = model.bn1.running_mean.clone()
        with torch.inference_mode():
            expected = model(x)
        self.assertEqual(tuple(expected.shape), (4, 10))
        torch.testing.assert_close(model.bn1.running_mean, bn_before)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'best.pt'
            save_checkpoint(path, {'model_state': model.state_dict(), 'mean': [0.5]*3,
                'std': [0.5]*3, 'classes': [str(i) for i in range(10)], 'architecture': 'resnet18_cifar_v1'})
            loaded, metadata = load_model(path, 'cpu')
            with torch.inference_mode():
                actual = loaded(x)
            torch.testing.assert_close(actual, expected)
            self.assertEqual(metadata['architecture'], 'resnet18_cifar_v1')


if __name__ == '__main__':
    unittest.main()
