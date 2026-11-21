"""Reload all frozen CNN checkpoints and recompute their held-out predictions."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from cnn import load_model, predict_raw, prepare_batch
from prepare_data import ROOT, file_hash, save_json
from train_cnn import configure_gpu


def audit_near_ties(saved, fresh, changed_logits, tolerance=1e-3):
    """Audit declared non-bitwise GPU inference; reject substantive changes."""
    positions = np.flatnonzero(saved != fresh)
    assert len(positions) == len(changed_logits)
    records = []
    for i, logits in zip(positions, changed_logits):
        top = np.argsort(logits)[-2:][::-1]
        gap = float(logits[top[0]]-logits[top[1]])
        assert gap <= tolerance, f'Changed prediction {i} is not a numerical near-tie: {gap}'
        assert {int(saved[i]), int(fresh[i])} == set(top.tolist()), f'Changed labels at {i} are not the top two'
        records.append({'position': int(i), 'saved_label': int(saved[i]), 'fresh_label': int(fresh[i]),
                        'top_labels': top.tolist(), 'logit_gap': gap})
    return records


def verify():
    configure_gpu()
    directory = Path((ROOT/'results/latest_cnn.txt').read_text(encoding='utf-8').strip())
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    results = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    frozen = json.loads((directory/'all_selections_frozen_before_test.json').read_text(encoding='utf-8'))
    assert json.loads((directory/'status.json').read_text(encoding='utf-8'))['status'] == 'complete'
    assert file_hash(ROOT/'data/dataset.npz') == config['dataset_sha256']
    assert file_hash(ROOT/'data/split_indices.npz') == config['split_sha256']
    with np.load(ROOT/'data/split_indices.npz') as split:
        train, validation, indices = split['train'], split['validation'], split['test']
    assert not set(train) & set(validation)
    with np.load(ROOT/'data/image_hashes.npz') as hashes:
        assert not set(hashes['train'][train]) & set(hashes['test'][indices])
        assert not set(hashes['train'][validation]) & set(hashes['test'][indices])
    with np.load(ROOT/'data/dataset.npz') as data:
        images, truth = data['test_images'][indices], data['test_labels'][indices]
    raw = torch.from_numpy(images).permute(0, 3, 1, 2).to('cuda')
    evidence, accuracies, scores = [], [], []
    for seed in config['seeds']:
        target = directory/f'seed_{seed}'
        assert file_hash(target/'best.pt') == frozen['seeds'][str(seed)]['checkpoint_sha256']
        model, checkpoint = load_model(target/'best.pt')
        pred = predict_raw(model, raw, checkpoint['mean'], checkpoint['std'], config['batch_size'])
        with (target/'test_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        np.testing.assert_array_equal([int(row['original_index']) for row in rows], indices)
        np.testing.assert_array_equal([int(row['true_label']) for row in rows], truth)
        saved_pred = np.array([int(row['predicted_label']) for row in rows])
        changed = np.flatnonzero(saved_pred != pred)
        changed_logits = []
        for i in changed:
            start = (int(i)//config['batch_size'])*config['batch_size']
            with torch.inference_mode():
                output = model(prepare_batch(raw[start:start+config['batch_size']], checkpoint['mean'], checkpoint['std']))
            changed_logits.append(output[int(i)-start].cpu().numpy())
        near_ties = audit_near_ties(saved_pred, pred, changed_logits)
        accuracy, f1 = accuracy_score(truth, saved_pred), f1_score(truth, saved_pred, average='macro')
        assert abs(accuracy - results['per_seed'][str(seed)]['test']['accuracy']) < 1e-12
        assert abs(f1 - results['per_seed'][str(seed)]['test']['f1_macro']) < 1e-12
        history = json.loads((target/'history.json').read_text(encoding='utf-8'))
        assert len(history) == config['epochs']
        expected = max(history, key=lambda row: row['validation_f1_macro'])
        assert checkpoint['epoch'] == expected['epoch']
        assert checkpoint['validation_f1_macro'] == expected['validation_f1_macro']
        accuracies.append(accuracy)
        scores.append(f1)
        evidence.append({'seed': seed, 'test_predictions_verified': len(pred), 'best_epoch': checkpoint['epoch'],
                         'test_accuracy': accuracy, 'test_macro_f1': f1,
                         'exact_reload_matches': int(np.count_nonzero(saved_pred == pred)),
                         'near_tie_changes': near_ties, 'fresh_reload_accuracy': accuracy_score(truth, pred),
                         'fresh_reload_macro_f1': f1_score(truth, pred, average='macro')})
        if seed == 42:
            Image.fromarray(images[0]).save(directory/'example_test_image.png')
            example_prediction = checkpoint['classes'][int(pred[0])]
        del model
        torch.cuda.empty_cache()
    for field, values in [('test_accuracy', accuracies), ('test_macro_f1', scores)]:
        assert abs(float(np.mean(values)) - results['aggregate'][field+'_mean']) < 1e-12
        assert abs(float(np.std(values)) - results['aggregate'][field+'_std']) < 1e-12
    for name in ['learning_curves.png', 'confusion_matrix_seed42.png', 'error_examples_seed42.png', 'comparison_svm_cnn.png']:
        with Image.open(directory/name) as image:
            image.verify()
    result = {'status': 'passed', 'per_seed': evidence, 'total_test_predictions_verified': sum(r['test_predictions_verified'] for r in evidence),
              'checkpoint_selection_verified': 'each earliest maximum validation-F1 epoch',
              'seed42_example_prediction': example_prediction,
              'aggregate_mean_std_verified': True, 'immutable_split_hash_verified': True,
              'near_tie_logit_gap_tolerance': 1e-3,
              'reload_guarantee': 'Exact checkpoint/data hashes and recorded-metric recomputation. Fresh GPU labels must match or differ only between top two logits with gap <=0.001; every exception is listed. Not a bitwise-identical GPU guarantee.'}
    save_json(directory/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
