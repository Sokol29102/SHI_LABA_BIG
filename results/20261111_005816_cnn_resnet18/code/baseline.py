"""Deterministic, stateless image features plus a train-only fitted SVM pipeline."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog

HOG_SIZE = 1764
FEATURE_SCHEMA = 'hog4_rgbhist32_v1'
FEATURE_CONFIG = {'schema': FEATURE_SCHEMA, 'image_shape': [32, 32, 3],
                  'hog': {'orientations': 9, 'pixels_per_cell': [4, 4],
                          'cells_per_block': [2, 2], 'block_norm': 'L2-Hys', 'channel_axis': -1},
                  'color': {'bins_per_rgb_channel': 32, 'normalization': 'each channel sums to 1'},
                  'hog_features': HOG_SIZE, 'total_features': 1860}


def extract_features(images):
    images = np.asarray(images)
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3) or images.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB images of shape (N, 32, 32, 3).')
    result = np.empty((len(images), 1860), dtype=np.float32)
    for i, image in enumerate(images):
        result[i, :HOG_SIZE] = hog(image, orientations=9, pixels_per_cell=(4, 4),
                                  cells_per_block=(2, 2), block_norm='L2-Hys', channel_axis=-1)
        for channel in range(3):
            start = HOG_SIZE + channel * 32
            result[i, start:start + 32] = np.bincount(
                image[:, :, channel].ravel() // 8, minlength=32) / 1024
    return result


def make_pipeline(use_color=False, c=0.001):
    selector = 'passthrough' if use_color else ColumnTransformer(
        [('hog', 'passthrough', slice(0, HOG_SIZE))], remainder='drop')
    return Pipeline([('select', selector), ('scale', StandardScaler()),
                     ('svm', LinearSVC(C=c, dual=False, max_iter=10000, tol=1e-4, random_state=42))])


def predict_images(bundle, images):
    if bundle['feature_schema'] != FEATURE_SCHEMA:
        raise ValueError('Model feature schema does not match this feature extractor.')
    return bundle['pipeline'].predict(extract_features(images))
