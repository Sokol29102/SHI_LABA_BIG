"""Train model 2 on the fixed CIFAR-10 split and save reproducible results."""
import argparse
import gc
import json
import math
import shutil
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from torch import nn

from cnn import ARCHITECTURE, build_model, load_model, normalization_stats, prepare_batch, predict_raw, save_checkpoint, set_seed
from prepare_data import ROOT, file_hash, save_json
from train_baseline import MemorySampler, metrics, write_csv


def now():
    return datetime.now(ZoneInfo('Europe/Kiev')).isoformat(timespec='seconds')


def configure_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for this experiment; no silent CPU fallback.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')


def profile_gpu(images, indices, mean, std, batch_size=256):
    configure_gpu()
    set_seed(42)
    model = build_model().to(device='cuda', memory_format=torch.channels_last).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    raw = torch.from_numpy(images[indices[:batch_size]].copy()).permute(0, 3, 1, 2).to('cuda')
    labels = torch.arange(len(raw), device='cuda') % 10  # synthetic targets, runtime only
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = nn.functional.cross_entropy(model(prepare_batch(raw, mean, std, True)), labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = nn.functional.cross_entropy(model(prepare_batch(raw, mean, std, True)), labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    result = {'gpu': torch.cuda.get_device_name(0), 'batch_size': batch_size,
              'measured_steps': 12, 'warmup_steps': 4, 'training_images_per_second': 12*len(raw)/seconds,
              'seconds_per_step': seconds/12,
              'estimated_train_epoch_seconds': seconds/12*math.ceil(len(indices)/batch_size),
              'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
              'peak_cuda_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
              'precision': 'BF16 autocast, FP32 weights',
              'note': 'Throwaway training-only timing profile; repeated batch and synthetic targets; excludes validation, checkpoint I/O, startup and full resident dataset. Not a quality experiment.'}
    del model, optimizer, raw, labels, loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def evaluate(model, raw, labels, indices, mean, std, batch_size):
    model.eval()
    total_loss = torch.zeros((), device='cuda')
    predictions = []
    for batch in indices.split(batch_size):
        output = model(prepare_batch(raw[batch], mean, std))
        total_loss += nn.functional.cross_entropy(output, labels[batch], reduction='sum')
        predictions.append(output.argmax(1))
    pred = torch.cat(predictions).cpu().numpy()
    truth = labels[indices].cpu().numpy()
    return float(total_loss.item()/len(indices)), pred, truth


def train_epoch(model, optimizer, raw, labels, indices, mean, std, batch_size):
    model.train()
    order = indices[torch.randperm(len(indices), device='cuda')]
    loss_sum, correct = torch.zeros((), device='cuda'), torch.zeros((), device='cuda')
    for batch in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(prepare_batch(raw[batch], mean, std, augment=True))
            loss = nn.functional.cross_entropy(output, labels[batch])
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Non-finite training loss.')
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach()*len(batch)
        correct += (output.argmax(1) == labels[batch]).sum()
    return float(loss_sum.item()/len(indices)), float(correct.item()/len(indices))


def recover_best(last, path):
    """The atomically committed last checkpoint owns the authoritative best state."""
    if 'best_checkpoint' in last:
        best = last['best_checkpoint']
        save_checkpoint(path, best)
        return best
    # Підтримуємо старий формат контрольних точок, якщо збережений найкращий стан узгоджується з результатами.
    best = torch.load(path, map_location='cpu', weights_only=True)
    expected = max(last['history'], key=lambda row: row['validation_f1_macro'])
    if best['epoch'] != expected['epoch'] or best['validation_f1_macro'] != expected['validation_f1_macro']:
        raise RuntimeError('Legacy best/last checkpoints disagree; refusing an unsafe resume.')
    return best


def merge_resource_peaks(previous, current):
    return {**{key: max(previous.get(key, 0.), current.get(key, 0.))
               for key in ['rss_mib', 'allocated_mib', 'reserved_mib']},
            'complete': previous.get('complete', False) and current.get('complete', False)}


def train_seed(directory, config, raw, labels, train_idx, val_idx, log):
    seed = config['seed']
    target = directory / f'seed_{seed}'
    target.mkdir(exist_ok=True)
    set_seed(seed)
    model = build_model().to(device='cuda', memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    best, history, start_epoch, previous_elapsed = -1., [], 1, 0.
    best_checkpoint = None
    previous_peaks = {'rss_mib': 0., 'allocated_mib': 0., 'reserved_mib': 0., 'complete': True}
    last_path = target / 'last.pt'
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location='cpu', weights_only=True)
        if checkpoint['config'] != config:
            raise ValueError('Resume configuration differs from the saved configuration.')
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        best, history = checkpoint['best_validation_f1'], checkpoint['history']
        start_epoch, previous_elapsed = checkpoint['epoch']+1, checkpoint['training_seconds']
        best_checkpoint = recover_best(checkpoint, target/'best.pt')
        previous_peaks = checkpoint.get('resource_peaks', {'complete': False})
        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        log(f'Resuming seed {seed}, epoch {start_epoch}')
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with MemorySampler() as memory:
        for epoch in range(start_epoch, config['epochs']+1):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_accuracy = train_epoch(model, optimizer, raw, labels, train_idx,
                config['mean'], config['std'], config['batch_size'])
            torch.cuda.synchronize()
            train_seconds = time.perf_counter()-tick
            val_loss, pred, truth = evaluate(model, raw, labels, val_idx,
                config['mean'], config['std'], config['batch_size'])
            val_f1 = float(f1_score(truth, pred, average='macro'))
            record = {'epoch': epoch, 'learning_rate': lr, 'train_loss': train_loss,
                      'train_accuracy_augmented': train_accuracy, 'validation_loss': val_loss,
                      'validation_accuracy': float(np.mean(pred == truth)), 'validation_f1_macro': val_f1,
                      'train_seconds': train_seconds}
            history.append(record)
            scheduler.step()
            metadata = {'architecture': ARCHITECTURE, 'seed': seed, 'epoch': epoch,
                        'mean': config['mean'], 'std': config['std'], 'classes': config['classes'],
                        'config': config, 'validation_f1_macro': val_f1}
            if val_f1 > best:
                best = val_f1
                best_checkpoint = {**metadata, 'model_state': {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
                save_checkpoint(target / 'best.pt', best_checkpoint)
            peaks = merge_resource_peaks(previous_peaks, {
                'rss_mib': memory.peak/2**20,
                'allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                'reserved_mib': torch.cuda.max_memory_reserved()/2**20, 'complete': True})
            save_checkpoint(last_path, {**metadata, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
                'best_validation_f1': best, 'history': history,
                'best_checkpoint': best_checkpoint, 'resource_peaks': peaks,
                'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state(),
                'training_seconds': previous_elapsed+time.perf_counter()-started})
            write_csv(target / 'history.csv', history)
            log(f'seed={seed} epoch={epoch}/{config["epochs"]} train_loss={train_loss:.4f} '
                f'val_acc={record["validation_accuracy"]:.4f} val_F1={val_f1:.4f} train_s={train_seconds:.2f}')
    torch.cuda.synchronize()
    peaks = merge_resource_peaks(previous_peaks, {
        'rss_mib': memory.peak/2**20, 'allocated_mib': torch.cuda.max_memory_allocated()/2**20,
        'reserved_mib': torch.cuda.max_memory_reserved()/2**20, 'complete': True})
    resources = {'training_wall_seconds': previous_elapsed+time.perf_counter()-started,
                 'train_epoch_seconds_sum': sum(r['train_seconds'] for r in history),
                 'sampled_peak_process_tree_rss_mib': peaks['rss_mib'],
                 'peak_cuda_allocated_mib': peaks['allocated_mib'],
                 'peak_cuda_reserved_mib': peaks['reserved_mib'],
                 'peak_measurement_complete': peaks['complete'],
                 'parameters_total': sum(p.numel() for p in model.parameters()),
                 'parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
                 'best_validation_f1': best, 'checkpoint_bytes': (target/'best.pt').stat().st_size,
                 'note': ('' if peaks['complete'] else 'Incomplete legacy-resume peak history: memory values are lower bounds. ')
                         + 'Training wall includes validation and checkpoint I/O. VRAM includes resident official-train images; RAM is sampled tree RSS and may overcount shared pages.'}
    save_json(target / 'training_resources.json', resources)
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return resources


def benchmark_inference(model, raw_cpu, mean, std, batch_size=512):
    raw = raw_cpu[:batch_size].to('cuda')
    prepared = prepare_batch(raw, mean, std)
    model.eval()
    only, full = [], []
    with torch.inference_mode():
        for _ in range(4):
            model(prepared)
        for _ in range(5):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            model(prepared).argmax(1)
            torch.cuda.synchronize()
            only.append(time.perf_counter()-tick)
        for _ in range(2):
            predict_raw(model, raw_cpu[:batch_size].to('cuda'), mean, std, batch_size)
        for _ in range(5):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            batch = raw_cpu[:batch_size].to('cuda')
            predict_raw(model, batch, mean, std, batch_size)
            torch.cuda.synchronize()
            full.append(time.perf_counter()-tick)
    return {'device': 'RTX 5060 Ti GPU', 'batch_size': batch_size, 'repetitions': 5,
            'model_only_seconds': only, 'model_only_median_ms_per_image': float(np.median(only)/batch_size*1000),
            'end_to_end_seconds': full, 'end_to_end_median_ms_per_image': float(np.median(full)/batch_size*1000),
            'scope': 'End-to-end: CPU uint8 NCHW batch, host-to-GPU transfer, normalization, FP32 model, argmax and CPU label return. Excludes file decode and resize. Model-only input is pre-normalized on GPU.'}


def plots(directory, rows, classes, test_y, example_pred, example_images):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for seed in [42, 123, 2026]:
        history = json.loads((directory/f'seed_{seed}'/'history.json').read_text(encoding='utf-8'))
        axes[0].plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label=f'{seed} train')
        axes[0].plot([r['epoch'] for r in history], [r['validation_loss'] for r in history], '--', label=f'{seed} val')
        axes[1].plot([r['epoch'] for r in history], [r['validation_f1_macro'] for r in history], label=f'seed {seed}')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Training and validation loss')
    axes[1].set(xlabel='Epoch', ylabel='Validation macro-F1', title='Checkpoint selection uses validation only')
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory/'learning_curves.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(test_y, example_pred, labels=np.arange(10), display_labels=classes,
        normalize='true', values_format='.2f', ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
    ax.set_title('ResNet-18 from scratch: test, predeclared seed 42')
    fig.tight_layout()
    fig.savefig(directory/'confusion_matrix_seed42.png', dpi=170)
    plt.close(fig)
    wrong = np.flatnonzero(test_y != example_pred)
    chosen = np.random.default_rng(42).choice(wrong, min(16, len(wrong)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax in axes.flat:
        ax.axis('off')
    for ax, i in zip(axes.flat, chosen):
        ax.imshow(example_images[i], interpolation='nearest')
        ax.set_title(f'True: {classes[test_y[i]]}\nPred: {classes[example_pred[i]]}', fontsize=10)
    fig.suptitle('ResNet-18: random test errors, seed 42', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(directory/'error_examples_seed42.png', dpi=170)
    plt.close(fig)
    baseline_dir = Path((ROOT/'results/latest_classical.txt').read_text(encoding='utf-8').strip())
    baseline = json.loads((baseline_dir/'metrics.json').read_text(encoding='utf-8'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    axes[0].bar(['HOG + SVM\nCPU', 'ResNet-18\nGPU, 3 seeds'],
        [baseline['test']['f1_macro'], np.mean([r['test_macro_f1'] for r in rows])],
        yerr=[0., np.std([r['test_macro_f1'] for r in rows])], capsize=5, color=['#64748b', '#0284c7'])
    axes[0].set(ylabel='Held-out test macro-F1', ylim=(0, 1), title='Quality; CNN error bar = seed std')
    axes[1].bar(['SVM final fit\nCPU', 'CNN train + val\nGPU, mean/seed'],
        [baseline['resources']['final_refit_seconds']/60, np.mean([r['training_wall_seconds'] for r in rows])/60],
        color=['#64748b', '#0284c7'])
    axes[1].set(ylabel='Minutes', title='Different devices and training procedures')
    fig.tight_layout()
    fig.savefig(directory/'comparison_svm_cnn.png', dpi=170)
    plt.close(fig)


def run(epochs=60, resume=None):
    configure_gpu()
    if resume:
        directory = Path(resume)
        config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
        if json.loads((directory/'status.json').read_text(encoding='utf-8'))['status'] == 'complete':
            raise ValueError('This run is already complete; use verify_cnn.py instead.')
    else:
        directory = ROOT/'results'/datetime.now(ZoneInfo('Europe/Kiev')).strftime('%Y%m%d_%H%M%S_cnn_resnet18')
        directory.mkdir(exist_ok=False)
        config = None
    def log(message):
        line = now()+' '+message
        print(line, flush=True)
        with (directory/'run.log').open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
    status = {'status': 'training', 'started': now()}
    save_json(directory/'status.json', status)
    try:
        dataset_hash = file_hash(ROOT/'data/dataset.npz')
        split_hash = file_hash(ROOT/'data/split_indices.npz')
        provenance = json.loads((ROOT/'data/provenance.json').read_text(encoding='utf-8'))
        if dataset_hash != provenance['dataset_sha256'] or split_hash != provenance['split_sha256']:
            raise ValueError('Dataset or immutable shared split changed.')
        with np.load(ROOT/'data/dataset.npz') as data:
            images, labels, classes = data['train_images'], data['train_labels'], data['classes'].tolist()
        with np.load(ROOT/'data/split_indices.npz') as split:
            train_indices, val_indices, test_indices = split['train'], split['validation'], split['test']
        mean, std = normalization_stats(images, train_indices)
        if config is None:
            config = {'architecture': ARCHITECTURE, 'weights': None, 'seeds': [42, 123, 2026],
                'epochs': epochs, 'batch_size': 256, 'learning_rate': 0.1, 'optimizer': 'SGD Nesterov',
                'momentum': 0.9, 'weight_decay': 5e-4, 'scheduler': 'CosineAnnealingLR',
                'training_precision': 'BF16 autocast, FP32 weights', 'evaluation_precision': 'FP32',
                'checkpoint_selection': 'highest validation macro-F1, earliest epoch on tie',
                'mean': mean, 'std': std, 'classes': classes, 'dataset_sha256': dataset_hash,
                'split_sha256': split_hash, 'sizes': {'train': len(train_indices), 'validation': len(val_indices), 'test': len(test_indices)},
                'gpu': torch.cuda.get_device_name(0), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                'cudnn_benchmark': True, 'tf32': True, 'strict_bitwise_determinism': False,
                'preprocessing': 'Train-only RGB population statistics; per-image random reflect-padded 4-pixel crop and 0.5 horizontal flip during training. Eval normalization only.',
                'data_pipeline': 'Resident uint8 official-train image tensor; GPU augmentation; no DataLoader workers',
                'python': sys.version, 'packages': {p: version(p) for p in ['torch', 'torchvision', 'numpy', 'scikit-learn']}}
            save_json(directory/'config.json', config)
        elif config['dataset_sha256'] != dataset_hash or config['split_sha256'] != split_hash:
            raise ValueError('Cannot resume with changed data.')
        source_dir = directory/'code'
        source_dir.mkdir(exist_ok=True)
        for name in ['cnn.py', 'train_cnn.py', 'predict_cnn.py', 'verify_cnn.py', 'test_cnn.py', 'requirements-cnn-lock.txt']:
            if (ROOT/name).exists() and not (source_dir/name).exists():
                shutil.copy2(ROOT/name, source_dir/name)
        raw = torch.from_numpy(images).permute(0, 3, 1, 2).to('cuda')
        y = torch.from_numpy(labels).to('cuda')
        train_idx = torch.from_numpy(train_indices).to('cuda')
        val_idx = torch.from_numpy(val_indices).to('cuda')
        resources = {}
        for seed in config['seeds']:
            resource_path = directory/f'seed_{seed}'/'training_resources.json'
            if resource_path.exists():
                resources[seed] = json.loads(resource_path.read_text(encoding='utf-8'))
                log(f'Seed {seed} training already complete; preserving it.')
            else:
                log(f'Starting seed {seed}: {config["epochs"]} epochs, {len(train_indices)} train images.')
                resources[seed] = train_seed(directory, {**config, 'seed': seed}, raw, y, train_idx, val_idx, log)
            last = torch.load(directory/f'seed_{seed}'/'last.pt', map_location='cpu', weights_only=True)
            recover_best(last, directory/f'seed_{seed}'/'best.pt')
            save_json(directory/f'seed_{seed}'/'history.json', last['history'])
            del last
        selections = {str(seed): {'checkpoint_sha256': file_hash(directory/f'seed_{seed}'/'best.pt'),
            'best_validation_f1': resources[seed]['best_validation_f1']} for seed in config['seeds']}
        save_json(directory/'all_selections_frozen_before_test.json', {'time': now(), 'seeds': selections})
        status.update(status='evaluating_frozen_checkpoints')
        save_json(directory/'status.json', status)
        # Завантажуємо тестові зображення й мітки після фіксації вибору для всіх трьох запусків.
        with np.load(ROOT/'data/dataset.npz') as data:
            test_images, test_labels = data['test_images'], data['test_labels']
        test_cpu = torch.from_numpy(test_images[test_indices].copy()).permute(0, 3, 1, 2)
        test_raw = test_cpu.to('cuda')
        rows, all_metrics = [], {}
        for seed in config['seeds']:
            target = directory/f'seed_{seed}'
            model, checkpoint = load_model(target/'best.pt')
            pred = predict_raw(model, test_raw, mean, std, config['batch_size'])
            _, val_pred, val_truth = evaluate(model, raw, y, val_idx, mean, std, config['batch_size'])
            result = {'test': metrics(test_labels[test_indices], pred, classes),
                'validation': metrics(val_truth, val_pred, classes), 'best_epoch': checkpoint['epoch'],
                'training_resources': resources[seed], 'inference': benchmark_inference(model, test_cpu, mean, std)}
            save_json(target/'metrics.json', result)
            write_csv(target/'test_predictions.csv', [{'original_index': int(i), 'true_label': int(test_labels[i]),
                'predicted_label': int(p), 'true_class': classes[int(test_labels[i])], 'predicted_class': classes[int(p)]}
                for i, p in zip(test_indices, pred)])
            write_csv(target/'per_class_metrics.csv', [{'class': name, **result['test']['classification_report'][name]} for name in classes])
            if seed == 42:
                example_pred = pred.copy()
            row = {'seed': seed, 'best_epoch': checkpoint['epoch'], 'test_accuracy': result['test']['accuracy'],
                'test_macro_f1': result['test']['f1_macro'], 'validation_macro_f1': result['validation']['f1_macro'],
                **resources[seed], 'inference_end_to_end_ms_per_image': result['inference']['end_to_end_median_ms_per_image']}
            rows.append({k: v for k, v in row.items() if k != 'note'})
            all_metrics[str(seed)] = result
            log(f'TEST seed={seed} best_epoch={checkpoint["epoch"]} accuracy={result["test"]["accuracy"]:.5f} macro_F1={result["test"]["f1_macro"]:.5f}')
            del model
            torch.cuda.empty_cache()
        write_csv(directory/'summary_per_seed.csv', rows)
        aggregate = {f'{field}_{stat}': float(func([row[field] for row in rows]))
            for field in ['test_accuracy', 'test_macro_f1', 'training_wall_seconds', 'peak_cuda_allocated_mib', 'inference_end_to_end_ms_per_image']
            for stat, func in [('mean', np.mean), ('std', np.std)]}
        save_json(directory/'metrics.json', {'aggregate': aggregate, 'per_seed': all_metrics,
            'std_definition': 'Population std across three training seeds; not CV fold std or confidence interval.'})
        plots(directory, rows, classes, test_labels[test_indices], example_pred, test_images[test_indices])
        baseline_dir = Path((ROOT/'results/latest_classical.txt').read_text(encoding='utf-8').strip())
        baseline = json.loads((baseline_dir/'metrics.json').read_text(encoding='utf-8'))
        write_csv(directory/'comparison.csv', [
            {'model': 'HOG+RGB+LinearSVC', 'device': 'CPU', 'test_macro_f1': baseline['test']['f1_macro'],
             'test_accuracy': baseline['test']['accuracy'], 'seed_test_std': '', 'training_seconds': baseline['resources']['final_refit_seconds'],
             'training_scope': 'single final refit; CV search reported separately', 'model_bytes': baseline['resources']['model_bytes'],
             'end_to_end_ms_per_image': baseline['resources']['end_to_end_inference']['median_ms_per_image']},
            {'model': 'ResNet18_CIFAR_from_scratch', 'device': 'RTX 5060 Ti GPU', 'test_macro_f1': aggregate['test_macro_f1_mean'],
             'test_accuracy': aggregate['test_accuracy_mean'], 'seed_test_std': aggregate['test_macro_f1_std'],
             'training_seconds': aggregate['training_wall_seconds_mean'], 'training_scope': 'mean per seed, 60 epochs including validation and checkpoint I/O',
             'model_bytes': resources[42]['checkpoint_bytes'], 'end_to_end_ms_per_image': aggregate['inference_end_to_end_ms_per_image_mean']}])
        (directory/'RESULTS_UA.md').write_text(
            '# Друга модель: ResNet-18 з нуля\n\n'
            f'Спільний split: {config["sizes"]}. Навчання: {config["epochs"]} епох, seeds 42, 123, 2026; без pretrained weights.\n\n'
            f'- Test accuracy: {aggregate["test_accuracy_mean"]*100:.2f}% ± {aggregate["test_accuracy_std"]*100:.2f} п.п.\n'
            f'- Test macro-F1: {aggregate["test_macro_f1_mean"]:.4f} ± {aggregate["test_macro_f1_std"]:.4f}.\n'
            f'- SVM test macro-F1: {baseline["test"]["f1_macro"]:.4f}; різниця: {aggregate["test_macro_f1_mean"]-baseline["test"]["f1_macro"]:.4f}.\n'
            f'- Середній training wall / seed: {aggregate["training_wall_seconds_mean"]:.1f} с.\n'
            f'- Середній peak CUDA allocated: {aggregate["peak_cuda_allocated_mib_mean"]:.1f} MiB.\n\n'
            'Std обчислено по трьох seeds, а не folds. Кожен checkpoint обрано лише за validation macro-F1; усі три вибори зафіксовано до test. '
            'Seed 42 наперед обрано для ілюстрацій, а не за результатами test. Нормалізація використовує лише train; augmentation вимкнено на evaluation. '
            'CUDA kernels не гарантують bitwise reproducibility.\n\n'
            'Порівняння часу не є GPU-vs-CPU speedup одного алгоритму: різні моделі, пристрої та процедури. CNN training wall включає validation і checkpoint I/O; '
            'SVM final refit відокремлено від CV search. End-to-end inference обох моделей використовує batch 512 і виключає читання файлів; CNN також включає host-to-GPU transfer. '
            'Peak VRAM містить resident training image buffer; це allocated memory процесу, а не загальна VRAM desktop.\n', encoding='utf-8')
        status.update(status='complete', finished=now())
        save_json(directory/'status.json', status)
        (ROOT/'results/latest_cnn.txt').write_text(str(directory), encoding='utf-8')
        log('COMPLETE '+json.dumps(aggregate))
        return directory
    except BaseException as error:
        status.update(status='failed', error=repr(error), updated=now())
        save_json(directory/'status.json', status)
        log('FAILED '+repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    run(args.epochs, args.resume)
