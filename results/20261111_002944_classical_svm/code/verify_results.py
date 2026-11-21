"""Independently verify a completed run, including fresh-process model loading."""
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from baseline import FEATURE_SCHEMA, predict_images
from prepare_data import ROOT, file_hash


def verify(root=ROOT):
    directory = Path((root / 'results' / 'latest_classical.txt').read_text(encoding='utf-8').strip())
    results = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    status = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    assert status['status'] == 'complete'
    assert file_hash(directory / 'model.joblib') == status['model_sha256']
    assert file_hash(root / 'data' / 'split_indices.npz') == config['split_sha256']
    assert file_hash(root / 'data' / 'dataset.npz') == config['dataset_sha256']
    with np.load(root / 'data' / 'split_indices.npz') as data:
        split = {name: data[name] for name in ['train', 'validation', 'test']}
    with np.load(root / 'data' / 'image_hashes.npz') as hashes:
        train_hashes = set(hashes['train'][split['train']])
        validation_hashes = set(hashes['train'][split['validation']])
        test_hashes = set(hashes['test'][split['test']])
    assert not (train_hashes & validation_hashes or train_hashes & test_hashes or validation_hashes & test_hashes)
    with np.load(directory / 'cv_fold_indices.npz') as folds:
        validation_members = []
        for fold in range(3):
            train = set(folds[f'fold_{fold}_train'])
            validation = set(folds[f'fold_{fold}_validation'])
            assert not train & validation
            assert train | validation == set(split['train'])
            validation_members.extend(validation)
        assert sorted(validation_members) == sorted(split['train'])
    bundle = joblib.load(directory / 'model.joblib')
    features = np.load(root / 'data' / 'features' / f'official_test_{FEATURE_SCHEMA}.npy', mmap_mode='r')
    reloaded = bundle['pipeline'].predict(features[split['test']])
    with (directory / 'test_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    indices = np.array([int(r['original_index']) for r in rows])
    truth = np.array([int(r['true_label']) for r in rows])
    saved_pred = np.array([int(r['predicted_label']) for r in rows])
    np.testing.assert_array_equal(indices, split['test'])
    np.testing.assert_array_equal(saved_pred, reloaded)
    with np.load(root / 'data' / 'dataset.npz') as dataset:
        np.testing.assert_array_equal(truth, dataset['test_labels'][indices])
        raw = dataset['test_images'][indices[:64]]
        np.testing.assert_array_equal(predict_images(bundle, raw), saved_pred[:64])
        # Save an ordinary image for checking the user-facing prediction CLI.
        example = directory / 'example_test_image.png'
        Image.fromarray(raw[0]).save(example)
    assert abs(accuracy_score(truth, saved_pred) - results['test']['accuracy']) < 1e-12
    assert abs(f1_score(truth, saved_pred, average='macro') - results['test']['f1_macro']) < 1e-12
    saved_cm = np.loadtxt(directory / 'confusion_matrix.csv', delimiter=',', skiprows=1, dtype=int)
    np.testing.assert_array_equal(saved_cm, confusion_matrix(truth, saved_pred, labels=np.arange(10)))
    for name in ['confusion_matrix.png', 'confusion_matrix_normalized.png', 'cross_validation.png', 'error_examples.png']:
        with Image.open(directory / name) as image:
            image.verify()
    evidence = {'status': 'passed', 'verified_test_predictions': len(rows),
                'fresh_raw_image_predictions_checked': len(raw), 'split_leakage': False,
                'cv_folds_validated': 3, 'test_accuracy': accuracy_score(truth, saved_pred),
                'test_macro_f1': f1_score(truth, saved_pred, average='macro'),
                'example_cli_expected_prediction': bundle['classes'][saved_pred[0]],
                'model_sha256': file_hash(directory / 'model.joblib')}
    (directory / 'verification.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))
    print('Verified run:', directory)


if __name__ == '__main__':
    verify()
