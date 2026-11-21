"""Small behavioral checks: leakage, feature contract, and usable serialization."""
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from baseline import extract_features, make_pipeline, predict_images
from prepare_data import clean_and_split


class BaselineChecks(unittest.TestCase):
    def test_duplicate_images_never_cross_splits(self):
        rng = np.random.default_rng(10)
        train = rng.integers(0, 256, (80, 32, 32, 3), dtype=np.uint8)
        labels = np.repeat([0, 1], 40)
        train[1] = train[0]  # same-label repeated image
        train[41] = train[40]
        train[43] = train[3]  # conflicting labels: drop both copies
        test = rng.integers(0, 256, (20, 32, 32, 3), dtype=np.uint8)
        test[0] = train[5]  # official test takes priority, independent of labels
        test[1] = test[2]
        test_labels = np.repeat([0, 1], 10)
        a = clean_and_split(train, labels, test, test_labels, seed=42)
        b = clean_and_split(train, labels, test, test_labels, seed=42)
        self.assertEqual(a['audit']['train_conflicting_hash_groups'], 1)
        self.assertEqual(a['audit']['train_test_overlap_removed_train'], 1)
        self.assertEqual(len(a['test']), 19)
        self.assertNotIn(3, np.r_[a['train'], a['validation']])
        self.assertNotIn(43, np.r_[a['train'], a['validation']])
        self.assertNotIn(5, np.r_[a['train'], a['validation']])
        for name in ['train', 'validation', 'test']:
            np.testing.assert_array_equal(a[name], b[name])
        train_hashes = {train[i].tobytes() for i in a['train']}
        val_hashes = {train[i].tobytes() for i in a['validation']}
        test_hashes = {test[i].tobytes() for i in a['test']}
        self.assertFalse(train_hashes & val_hashes)
        self.assertFalse(train_hashes & test_hashes)
        self.assertFalse(val_hashes & test_hashes)

    def test_features_are_finite_and_input_is_validated(self):
        images = np.zeros((2, 32, 32, 3), dtype=np.uint8)
        images[1, :, 16:, 0] = 255
        features = extract_features(images)
        self.assertEqual(features.shape, (2, 1860))  # 7*7*2*2*9 + 3*32
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(float(np.abs(features[1] - features[0]).sum()), 0)
        with self.assertRaises(ValueError):
            extract_features(np.zeros((2, 16, 16, 3), dtype=np.uint8))

    def test_saved_model_reproduces_image_predictions_without_refitting(self):
        rng = np.random.default_rng(22)
        images = rng.integers(0, 256, (20, 32, 32, 3), dtype=np.uint8)
        y = np.tile([0, 1], 10)
        features = extract_features(images)
        model = make_pipeline(use_color=True, c=0.001)
        model.fit(features[:16], y[:16])
        scaler_mean = model.named_steps['scale'].mean_.copy()
        bundle = {'pipeline': model, 'classes': ['a', 'b'], 'feature_schema': 'hog4_rgbhist32_v1'}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.joblib'
            joblib.dump(bundle, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(
                predict_images(loaded, images[16:]), model.predict(features[16:]))
            np.testing.assert_array_equal(loaded['pipeline'].named_steps['scale'].mean_, scaler_mean)


if __name__ == '__main__':
    unittest.main()
