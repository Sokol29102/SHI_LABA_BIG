"""Frozen checkpoint evaluation and comparison of all three completed procedures."""
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from threadpoolctl import threadpool_limits

from baseline import predict_images as svm_predict
from cnn import load_model as load_cnn, prepare_batch as cnn_prepare
from prepare_data import ROOT, save_json
from train_baseline import metrics, write_csv
from vit import load_model, predict_raw, prepare_batch, strict_evaluation


def benchmark_neural(model, raw_cpu, mean, std, prepare_fn=prepare_batch):
    batch_size = 512
    raw_cpu = raw_cpu[:batch_size]
    raw_gpu = raw_cpu.to('cuda')
    model.eval()
    only, full = [], []
    with strict_evaluation(), torch.inference_mode():
        normalized = prepare_fn(raw_gpu, mean, std)
        for _ in range(3):
            model(normalized)
        for _ in range(5):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            model(normalized).argmax(1)
            torch.cuda.synchronize()
            only.append(time.perf_counter()-tick)
        for _ in range(2):
            model(prepare_fn(raw_cpu.to('cuda'), mean, std)).argmax(1).cpu()
        for _ in range(5):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            model(prepare_fn(raw_cpu.to('cuda'), mean, std)).argmax(1).cpu()
            torch.cuda.synchronize()
            full.append(time.perf_counter()-tick)
    return {'device': 'RTX 5060 Ti GPU', 'precision': 'strict FP32, TF32 off', 'batch_size': batch_size,
        'repetitions': 5, 'model_only_seconds': only, 'end_to_end_seconds': full,
        'model_only_median_ms_per_image': float(np.median(only)/batch_size*1000),
        'end_to_end_median_ms_per_image': float(np.median(full)/batch_size*1000),
        'scope': 'CPU raw uint8 NCHW input; transfer, model-specific resize/normalization, model, argmax, CPU labels; excludes file decode. Model-only starts normalized on GPU.'}


def make_figures(directory, config, rows, truth, pred42, images, comparison):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for seed, color in zip(config['seeds'], ['#0284c7', '#d97706', '#16a34a']):
        history = json.loads((directory/f'seed_{seed}/history.json').read_text(encoding='utf-8'))
        epochs = [r['epoch'] for r in history]
        axes[0].plot(epochs, [r['train_loss'] for r in history], color=color, label=f'{seed} train')
        axes[0].plot(epochs, [r['validation_loss'] for r in history], '--', color=color, label=f'{seed} val')
        axes[1].plot(epochs, [r['validation_f1_macro'] for r in history], color=color, marker='o', label=f'seed {seed}')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='DeiT-Tiny: local fine-tuning')
    axes[1].set(xlabel='Epoch', ylabel='Validation macro-F1', title='Checkpoint selection: validation only')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory/'learning_curves.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(truth, pred42, labels=np.arange(10), display_labels=config['classes'],
        normalize='true', values_format='.2f', cmap='Blues', colorbar=False, xticks_rotation=45, ax=ax)
    ax.set_title('Fine-tuned DeiT-Tiny: test, predeclared seed 42')
    fig.tight_layout()
    fig.savefig(directory/'confusion_matrix_seed42.png', dpi=170)
    plt.close(fig)
    wrong = np.flatnonzero(truth != pred42)
    chosen = np.random.default_rng(42).choice(wrong, min(16, len(wrong)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax in axes.flat:
        ax.axis('off')
    for ax, index in zip(axes.flat, chosen):
        ax.imshow(images[index], interpolation='nearest')
        ax.set_title(f'True: {config["classes"][truth[index]]}\nPred: {config["classes"][pred42[index]]}', fontsize=10)
    fig.suptitle('DeiT-Tiny: random test errors, seed 42', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(directory/'error_examples_seed42.png', dpi=170)
    plt.close(fig)
    labels = ['HOG + SVM\nCPU', 'ResNet-18\nGPU, scratch', 'DeiT-Tiny\nGPU, pretrained']
    colors = ['#64748b', '#0284c7', '#7c3aed']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    axes[0].bar(labels, [r['test_macro_f1'] for r in comparison], color=colors,
                yerr=[r['test_macro_f1_seed_std'] or 0. for r in comparison], capsize=4)
    axes[0].set(ylabel='Held-out test macro-F1', ylim=(0, 1), title='Quality (NN error bars = seed std)')
    axes[1].bar(labels, [r['local_train_seconds_per_fit']/60 for r in comparison], color=colors)
    axes[1].set(ylabel='Minutes', title='Local fit / fine-tune per seed')
    axes[2].bar(labels, [r['matched_inference_ms_per_image'] for r in comparison], color=colors)
    axes[2].set(ylabel='ms/image, batch 512', title='End-to-end inference; NN strict FP32')
    fig.text(.5, .015, 'Different devices, epoch budgets and prior training. SVM CV search and original ImageNet pretraining are separate costs.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(directory/'comparison_all_models.png', dpi=170)
    plt.close(fig)


def finalize(directory, config, resources, log):
    with np.load(ROOT/'data/split_indices.npz') as split:
        indices = split['test']
    with np.load(ROOT/'data/dataset.npz') as data:
        images, labels = data['test_images'][indices], data['test_labels'][indices]
    raw_cpu = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
    raw = raw_cpu.to('cuda')
    seed_rows, per_seed = [], {}
    for seed in config['seeds']:
        target = directory/f'seed_{seed}'
        model, checkpoint = load_model(target/'best.pt')
        pred = predict_raw(model, raw, batch_size=config['batch_size'])
        result = {'test': metrics(labels, pred, config['classes']), 'best_epoch': checkpoint['epoch'],
                  'training_resources': resources[seed]}
        save_json(target/'metrics.json', result)
        write_csv(target/'test_predictions.csv', [{'original_index': int(i), 'true_label': int(y),
            'predicted_label': int(p), 'true_class': config['classes'][int(y)], 'predicted_class': config['classes'][int(p)]}
            for i, y, p in zip(indices, labels, pred)])
        write_csv(target/'per_class_metrics.csv', [{'class': name, **result['test']['classification_report'][name]} for name in config['classes']])
        np.savez(target/'test_arrays.npz', indices=indices, truth=labels, predictions=pred)
        if seed == 42:
            pred42 = pred.copy()
            vit_benchmark = benchmark_neural(model, raw_cpu, checkpoint['mean'], checkpoint['std'])
        row = {'seed': seed, 'best_epoch': checkpoint['epoch'], 'test_accuracy': result['test']['accuracy'],
               'test_macro_f1': result['test']['f1_macro'], **{k: v for k, v in resources[seed].items() if k != 'scope'}}
        seed_rows.append(row)
        per_seed[str(seed)] = result
        log(f'TEST seed={seed} epoch={checkpoint["epoch"]} accuracy={result["test"]["accuracy"]:.5f} macro_F1={result["test"]["f1_macro"]:.5f}')
        del model
        torch.cuda.empty_cache()
    aggregate = {f'{field}_{name}': float(operation([r[field] for r in seed_rows]))
        for field in ['test_accuracy', 'test_macro_f1', 'training_wall_seconds', 'peak_cuda_allocated_mib']
        for name, operation in [('mean', np.mean), ('std', np.std)]}
    save_json(directory/'metrics.json', {'aggregate': aggregate, 'per_seed': per_seed,
        'std_definition': 'Population std across three training seeds; SVM CV fold std is distinct.'})
    write_csv(directory/'summary_per_seed.csv', seed_rows)
    baseline_dir = Path((ROOT/'results/latest_classical.txt').read_text(encoding='utf-8').strip())
    cnn_dir = Path((ROOT/'results/latest_cnn.txt').read_text(encoding='utf-8').strip())
    svm_metrics = json.loads((baseline_dir/'metrics.json').read_text(encoding='utf-8'))
    cnn_metrics = json.loads((cnn_dir/'metrics.json').read_text(encoding='utf-8'))
    cnn_model, cnn_checkpoint = load_cnn(cnn_dir/'seed_42/best.pt')
    cnn_benchmark = benchmark_neural(cnn_model, raw_cpu, cnn_checkpoint['mean'], cnn_checkpoint['std'], cnn_prepare)
    del cnn_model
    torch.cuda.empty_cache()
    svm = joblib.load(baseline_dir/'model.joblib')
    with threadpool_limits(limits=1):
        for _ in range(2):
            svm_predict(svm, images[:512])
        timings = []
        for _ in range(5):
            tick = time.perf_counter()
            svm_predict(svm, images[:512])
            timings.append(time.perf_counter()-tick)
    svm_benchmark = {'device': 'CPU, 1 native thread', 'batch_size': 512, 'repetitions': 5,
        'end_to_end_seconds': timings, 'end_to_end_median_ms_per_image': float(np.median(timings)/512*1000),
        'scope': 'CPU raw uint8 NHWC input; HOG/RGB, selection/scaling/SVM and labels; excludes file decode.'}
    save_json(directory/'matched_inference_benchmark.json', {'svm': svm_benchmark, 'cnn': cnn_benchmark, 'transformer': vit_benchmark,
        'scope': 'Same first 512 test images, warmup, 5 timed repeats. Fixed seed42 for both neural architectures. Strict FP32 and TF32 disabled for both neural timing; recorded CNN quality used its original TF32-allowed evaluation.'})
    cnn_resources = [value['training_resources'] for value in cnn_metrics['per_seed'].values()]
    comparison = [
        {'model': 'HOG+RGB+LinearSVC', 'device': 'CPU', 'prior_training': 'none', 'input_resolution': 32,
         'test_accuracy': svm_metrics['test']['accuracy'], 'test_accuracy_seed_std': None,
         'test_macro_f1': svm_metrics['test']['f1_macro'], 'test_macro_f1_seed_std': None,
         'local_train_seconds_per_fit': svm_metrics['resources']['final_refit_seconds'],
         'total_experiment_training_seconds': svm_metrics['resources']['cv_search_and_refit_seconds'],
         'parameters_or_svm_coefficients': svm_metrics['resources']['svm_coefficients_and_intercepts'],
         'checkpoint_bytes': svm_metrics['resources']['model_bytes'],
         'peak_training_ram_mib': svm_metrics['resources']['sampled_peak_fit_tree_rss_mb'], 'peak_cuda_allocated_mib': 0.,
         'matched_inference_ms_per_image': svm_benchmark['end_to_end_median_ms_per_image']},
        {'model': 'ResNet18_CIFAR', 'device': 'RTX 5060 Ti GPU', 'prior_training': 'none', 'input_resolution': 32,
         'test_accuracy': cnn_metrics['aggregate']['test_accuracy_mean'], 'test_accuracy_seed_std': cnn_metrics['aggregate']['test_accuracy_std'],
         'test_macro_f1': cnn_metrics['aggregate']['test_macro_f1_mean'], 'test_macro_f1_seed_std': cnn_metrics['aggregate']['test_macro_f1_std'],
         'local_train_seconds_per_fit': cnn_metrics['aggregate']['training_wall_seconds_mean'],
         'total_experiment_training_seconds': sum(r['training_wall_seconds'] for r in cnn_resources),
         'parameters_or_svm_coefficients': cnn_resources[0]['parameters_total'], 'checkpoint_bytes': cnn_resources[0]['checkpoint_bytes'],
         'peak_training_ram_mib': max(r['sampled_peak_process_tree_rss_mib'] for r in cnn_resources),
         'peak_cuda_allocated_mib': max(r['peak_cuda_allocated_mib'] for r in cnn_resources),
         'matched_inference_ms_per_image': cnn_benchmark['end_to_end_median_ms_per_image']},
        {'model': 'DeiT_Tiny_finetuned', 'device': 'RTX 5060 Ti GPU', 'prior_training': 'ImageNet-1K, cost excluded', 'input_resolution': 224,
         'test_accuracy': aggregate['test_accuracy_mean'], 'test_accuracy_seed_std': aggregate['test_accuracy_std'],
         'test_macro_f1': aggregate['test_macro_f1_mean'], 'test_macro_f1_seed_std': aggregate['test_macro_f1_std'],
         'local_train_seconds_per_fit': aggregate['training_wall_seconds_mean'],
         'total_experiment_training_seconds': sum(r['training_wall_seconds'] for r in resources.values()),
         'parameters_or_svm_coefficients': resources[42]['parameters_total'], 'checkpoint_bytes': resources[42]['checkpoint_bytes'],
         'peak_training_ram_mib': max(r['sampled_peak_process_tree_rss_mib'] for r in resources.values()),
         'peak_cuda_allocated_mib': max(r['peak_cuda_allocated_mib'] for r in resources.values()),
         'matched_inference_ms_per_image': vit_benchmark['end_to_end_median_ms_per_image']}]
    write_csv(directory/'comparison_all_models.csv', comparison)
    save_json(directory/'comparison_all_models.json', comparison)
    make_figures(directory, config, seed_rows, labels, pred42, images, comparison)
    (directory/'RESULTS_UA.md').write_text('# Третя модель: fine-tuning DeiT-Tiny\n\n'
        f'Test accuracy: {aggregate["test_accuracy_mean"]*100:.2f}% ± {aggregate["test_accuracy_std"]*100:.2f} п.п.; '
        f'macro-F1: {aggregate["test_macro_f1_mean"]:.4f} ± {aggregate["test_macro_f1_std"]:.4f}.\n\n'
        f'Локальне навчання: {config["epochs"]} епох x 3 seeds; середній час: {aggregate["training_wall_seconds_mean"]:.1f} с/seed. '
        'Backbone попередньо навчено на ImageNet-1K; його початкова вартість не вимірювалася і не включена. Це порівняння практичних процедур, а не ізольований доказ переваги attention.\n\n'
        'Train/validation/test незмінні. Усі checkpoint обрано за validation macro-F1 та зафіксовано до test. '
        '32x32 збільшується до 224x224 bicubic без додавання деталей; використано published ImageNet normalization. '
        'Transformer evaluation - strict FP32, TF32 off. Спільний inference benchmark для обох нейромереж також strict FP32; CPU SVM має інший hardware. '
        'Якість CNN збережено з його початкового FP32/TF32-allowed запуску з документованим near-tie caveat. '
        'Seed42 наперед обрано для ілюстрацій і latency, без вибору seed за test.\n\n'
        'Детальні quality/time/RAM/VRAM/parameters/storage/latency дані: comparison_all_models.csv, metrics.json, matched_inference_benchmark.json. '
        'SVM RAM peak охоплює CV workers; neural RAM/VRAM включають resident dataset і можуть залежати від startup/autotuning. '
        'Std SVM по folds не прирівнюється до neural seed std; test seed std SVM не оцінено і залишено порожнім у CSV.\n', encoding='utf-8')
    return aggregate
