"""Fresh-process, exact reload check under the frozen strict-FP32 policy."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from prepare_data import ROOT, file_hash, save_json
from vit import load_model, predict_raw


def verify():
    torch.set_num_threads(4)
    directory = Path((ROOT/'results/latest_transformer.txt').read_text(encoding='utf-8').strip())
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
    raw = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).to('cuda')
    evidence, accuracies, scores = [], [], []
    for seed in config['seeds']:
        target = directory/f'seed_{seed}'
        assert file_hash(target/'best.pt') == frozen['seeds'][str(seed)]['checkpoint_sha256']
        model, checkpoint = load_model(target/'best.pt')
        pred = predict_raw(model, raw, checkpoint['mean'], checkpoint['std'], config['batch_size'])
        with (target/'test_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
            saved = list(csv.DictReader(stream))
        np.testing.assert_array_equal([int(row['original_index']) for row in saved], indices)
        np.testing.assert_array_equal([int(row['true_label']) for row in saved], truth)
        np.testing.assert_array_equal([int(row['predicted_label']) for row in saved], pred)
        accuracy, f1 = accuracy_score(truth, pred), f1_score(truth, pred, average='macro')
        assert abs(accuracy-results['per_seed'][str(seed)]['test']['accuracy']) < 1e-12
        assert abs(f1-results['per_seed'][str(seed)]['test']['f1_macro']) < 1e-12
        history = json.loads((target/'history.json').read_text(encoding='utf-8'))
        assert len(history) == config['epochs']
        best = max(history, key=lambda row: row['validation_f1_macro'])
        assert checkpoint['epoch'] == best['epoch']
        assert checkpoint['validation_f1_macro'] == best['validation_f1_macro']
        accuracies.append(accuracy)
        scores.append(f1)
        evidence.append({'seed': seed, 'best_epoch': checkpoint['epoch'], 'exact_predictions_verified': len(pred),
                         'test_accuracy': accuracy, 'test_macro_f1': f1})
        if seed == 42:
            Image.fromarray(images[0]).save(directory/'example_test_image.png')
            example = checkpoint['classes'][int(pred[0])]
        del model
        torch.cuda.empty_cache()
    for field, values in [('test_accuracy', accuracies), ('test_macro_f1', scores)]:
        assert abs(float(np.mean(values))-results['aggregate'][field+'_mean']) < 1e-12
        assert abs(float(np.std(values))-results['aggregate'][field+'_std']) < 1e-12
    for name in ['learning_curves.png', 'confusion_matrix_seed42.png', 'error_examples_seed42.png', 'comparison_all_models.png']:
        with Image.open(directory/name) as image:
            image.verify()
    audit = {'status': 'passed', 'exact_predictions_verified': sum(row['exact_predictions_verified'] for row in evidence),
             'seeds': evidence, 'checkpoint_selection_and_hashes_verified': True,
             'recorded_metrics_and_aggregates_verified': True, 'evaluation': 'strict FP32, same GPU/software/batch policy',
             'example_prediction_seed42': example}
    save_json(directory/'verification.json', audit)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    verify()
