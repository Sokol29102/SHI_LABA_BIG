"""Inspect numerical prediction changes without altering models or recorded results."""
import csv
import json

import numpy as np
import torch

from cnn import load_model, predict_raw, prepare_batch
from prepare_data import ROOT, save_json
from train_cnn import configure_gpu

configure_gpu()
directory = ROOT/'results/20261111_005816_cnn_resnet18'
with np.load(ROOT/'data/split_indices.npz') as data:
    indices = data['test']
with np.load(ROOT/'data/dataset.npz') as data:
    raw_cpu = torch.from_numpy(data['test_images'][indices].copy()).permute(0, 3, 1, 2)
    truth = data['test_labels'][indices]
raw = raw_cpu.to('cuda')
report = []
for seed in [42, 123, 2026]:
    configure_gpu()
    model, checkpoint = load_model(directory/f'seed_{seed}/best.pt')
    with (directory/f'seed_{seed}/test_predictions.csv').open(encoding='utf-8-sig', newline='') as stream:
        saved = np.array([int(row['predicted_label']) for row in csv.DictReader(stream)])
    repeated = predict_raw(model, raw, checkpoint['mean'], checkpoint['std'])
    wrong = np.flatnonzero(saved != repeated)
    details = []
    for i in wrong:
        # Keep the original full batch shape and position, avoiding a batch-size change.
        start = (int(i)//256)*256
        with torch.inference_mode():
            logits = model(prepare_batch(raw[start:start+256], checkpoint['mean'], checkpoint['std']))[int(i)-start].float().cpu()
        top = logits.topk(3)
        details.append({'position': int(i), 'original_index': int(indices[i]), 'true': int(truth[i]),
            'saved_label': int(saved[i]), 'fresh_label': int(repeated[i]),
            'top_labels': top.indices.tolist(), 'top_logits': top.values.tolist(),
            'top2_logit_margin': float(top.values[0]-top.values[1])})
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    strict = predict_raw(model, raw, checkpoint['mean'], checkpoint['std'])
    item = {'seed': seed, 'fast_reload_changed': len(wrong), 'details': details,
            'strict_vs_saved_changed': int(np.count_nonzero(strict != saved)),
            'saved_accuracy': float(np.mean(saved == truth)), 'fast_reload_accuracy': float(np.mean(repeated == truth)),
            'strict_accuracy': float(np.mean(strict == truth))}
    report.append(item)
    print(json.dumps(item), flush=True)
    del model
    torch.cuda.empty_cache()
save_json(directory/'reload_numerics_diagnostic.json', report)
