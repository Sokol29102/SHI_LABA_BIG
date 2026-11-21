"""Full classical baseline: 3-fold CV, final train-only fit, held-out evaluation."""
import argparse
import csv
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import psutil
from joblib import parallel_config
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from threadpoolctl import threadpool_limits

from baseline import FEATURE_CONFIG, FEATURE_SCHEMA, extract_features, make_pipeline, predict_images
from prepare_data import ROOT, file_hash, save_json


class MemorySampler:
    """Sample aggregate RSS of parent and worker processes (shared pages may count twice)."""
    def __init__(self):
        self.peak = 0
        self.stop = threading.Event()

    def sample(self):
        while not self.stop.is_set():
            process = psutil.Process()
            total = 0
            for p in [process] + process.children(recursive=True):
                try:
                    total += p.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak = max(self.peak, total)
            self.stop.wait(0.1)

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(y, pred, classes):
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
    return {'accuracy': float(accuracy_score(y, pred)), 'precision_macro': float(precision),
            'recall_macro': float(recall), 'f1_macro': float(f1), 'n_images': len(y),
            'classification_report': classification_report(y, pred, labels=np.arange(len(classes)),
                target_names=classes, output_dict=True, zero_division=0)}


def features_with_cache(images, name, root, log):
    directory = root / 'data' / 'features'
    directory.mkdir(exist_ok=True)
    path, meta_path = directory / f'{name}_{FEATURE_SCHEMA}.npy', directory / f'{name}_{FEATURE_SCHEMA}.json'
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['dataset_sha256'] != file_hash(root / 'data' / 'dataset.npz'):
            raise ValueError('Feature cache belongs to different dataset.')
        result = np.load(path, mmap_mode='r')
        if result.shape != (len(images), FEATURE_CONFIG['total_features']):
            raise ValueError('Invalid feature cache shape.')
        log(f'Using cached {name} features: {result.shape}')
        return result, {**meta, 'cache_reused_this_run': True}
    partial = path.with_suffix('.npy.part')
    start = time.perf_counter()
    with MemorySampler() as monitor:
        result = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32,
                                          shape=(len(images), FEATURE_CONFIG['total_features']))
        for i in range(0, len(images), 2000):
            end = min(i + 2000, len(images))
            result[i:end] = extract_features(images[i:end])
            log(f'{name} features {end}/{len(images)}; {time.perf_counter()-start:.1f}s')
        result.flush()
        del result
    partial.replace(path)
    meta = {'feature_config': FEATURE_CONFIG, 'dataset_sha256': file_hash(root / 'data' / 'dataset.npz'),
            'images': len(images), 'extraction_seconds': time.perf_counter()-start,
            'sampled_peak_tree_rss_mb': monitor.peak / 2**20, 'cache_reused_this_run': False,
            'features_sha256': file_hash(path)}
    save_json(meta_path, meta)
    return np.load(path, mmap_mode='r'), meta


def plot_results(directory, classes, y, pred, images, cv_rows):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    cm = confusion_matrix(y, pred, labels=np.arange(len(classes)))
    for normalized in [False, True]:
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay.from_predictions(y, pred, labels=np.arange(len(classes)),
            display_labels=classes, normalize='true' if normalized else None,
            values_format='.2f' if normalized else 'd', ax=ax, cmap='Blues', xticks_rotation=45, colorbar=False)
        ax.set_title('HOG + Linear SVM: held-out test' + (' (row-normalized)' if normalized else ''))
        fig.tight_layout()
        fig.savefig(directory / ('confusion_matrix_normalized.png' if normalized else 'confusion_matrix.png'), dpi=170)
        plt.close(fig)
    np.savetxt(directory / 'confusion_matrix.csv', cm, delimiter=',', fmt='%d', header=','.join(classes), comments='')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for color, name in [(False, 'HOG'), (True, 'HOG + RGB histogram')]:
        rows = [r for r in cv_rows if r['use_color'] == color]
        ax.errorbar([r['C'] for r in rows], [r['cv_f1_macro_mean'] for r in rows],
                    yerr=[r['cv_f1_macro_std'] for r in rows], marker='o', capsize=4, label=name)
    ax.set(xscale='log', xlabel='SVM regularization parameter C', ylabel='3-fold CV macro-F1',
           title='Hyperparameter selection on training data only')
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / 'cross_validation.png', dpi=170)
    plt.close(fig)
    # Фіксуємо seed для вибору прикладів і не підбираємо пороги впевненості після перегляду тесту.
    rng = np.random.default_rng(42)
    wrong = np.flatnonzero(pred != y)
    chosen = rng.choice(wrong, size=min(16, len(wrong)), replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax in axes.flat:
        ax.axis('off')
    for ax, index in zip(axes.flat, chosen):
        ax.imshow(images[index], interpolation='nearest')
        ax.set_title(f'True: {classes[y[index]]}\nPred: {classes[pred[index]]}', fontsize=10)
    fig.suptitle('Random test errors (seed 42)', fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(directory / 'error_examples.png', dpi=170)
    plt.close(fig)


def run(root=ROOT):
    root = Path(root)
    stamp = datetime.now(ZoneInfo('Europe/Kiev')).strftime('%Y%m%d_%H%M%S')
    directory = root / 'results' / f'{stamp}_classical_svm'
    directory.mkdir(parents=True, exist_ok=False)
    def log(message):
        line = datetime.now(ZoneInfo('Europe/Kiev')).isoformat(timespec='seconds') + ' ' + message
        print(line, flush=True)
        with (directory / 'run.log').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
    status = {'status': 'running', 'started': datetime.now(ZoneInfo('Europe/Kiev')).isoformat()}
    save_json(directory / 'status.json', status)
    started = time.perf_counter()
    try:
        with np.load(root / 'data' / 'dataset.npz') as data:
            train_images, train_labels = data['train_images'], data['train_labels']
            test_images, test_labels, classes = data['test_images'], data['test_labels'], data['classes'].tolist()
        with np.load(root / 'data' / 'split_indices.npz') as saved:
            split = {name: saved[name] for name in ['train', 'validation', 'test']}
        config = {'model': 'HOG + optional RGB histogram + StandardScaler + LinearSVC',
                  'seed': 42, 'cv_folds': 3, 'C_grid': [0.0001, 0.001, 0.01],
                  'feature_variants': ['HOG', 'HOG+RGB'], 'selection_metric': 'f1_macro',
                  'refit_data': 'shared train only; validation preserved', 'cv_jobs': 3,
                  'native_threads_per_worker': 1, 'svm_dual': False, 'svm_tol': 1e-4,
                  'svm_max_iter': 10000, 'feature_config': FEATURE_CONFIG,
                  'split_sha256': file_hash(root / 'data' / 'split_indices.npz'),
                  'dataset_sha256': file_hash(root / 'data' / 'dataset.npz'),
                  'sizes': {k: len(v) for k, v in split.items()},
                  'python': sys.version, 'platform': platform.platform(),
                  'cpu': platform.processor(), 'logical_cpus': os.cpu_count(),
                  'total_ram_gb': psutil.virtual_memory().total / 2**30,
                  'device': 'CPU', 'gpu_used': False,
                  'packages': {p: version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'scikit-image', 'joblib', 'matplotlib', 'psutil', 'threadpoolctl']}}
        save_json(directory / 'config.json', config)
        for name in ['prepare_data.py', 'baseline.py', 'train_baseline.py', 'predict.py', 'test_baseline.py']:
            source = root / name
            if source.exists():
                target = directory / 'code'
                target.mkdir(exist_ok=True)
                (target / name).write_bytes(source.read_bytes())
        train_features, train_feature_cost = features_with_cache(train_images, 'official_train', root, log)
        X, y = np.asarray(train_features[split['train']]), train_labels[split['train']]
        cv = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(X, y))
        np.savez(directory / 'cv_fold_indices.npz', **{
            f'fold_{i}_{part}': split['train'][indices]
            for i, pair in enumerate(cv) for part, indices in zip(['train', 'validation'], pair)})
        params = [{'select': [make_pipeline(use_color=color).named_steps['select']],
                   'svm__C': config['C_grid']} for color in [False, True]]
        search = GridSearchCV(make_pipeline(), params, scoring={'f1_macro': 'f1_macro', 'accuracy': 'accuracy'},
            refit='f1_macro', cv=cv, n_jobs=3, pre_dispatch=3, return_train_score=True, error_score='raise', verbose=2)
        log(f'Starting 18 CV fits and one refit on {len(y)} training images.')
        tick = time.perf_counter()
        with MemorySampler() as fit_memory, threadpool_limits(limits=1), parallel_config(backend='loky', inner_max_num_threads=1):
            search.fit(X, y)
        search_seconds = time.perf_counter() - tick
        rows = []
        for i, p in enumerate(search.cv_results_['params']):
            row = {'C': p['svm__C'], 'use_color': isinstance(p['select'], str),
                   'cv_f1_macro_mean': float(search.cv_results_['mean_test_f1_macro'][i]),
                   'cv_f1_macro_std': float(search.cv_results_['std_test_f1_macro'][i]),
                   'cv_train_f1_macro_mean': float(search.cv_results_['mean_train_f1_macro'][i]),
                   'cv_accuracy_mean': float(search.cv_results_['mean_test_accuracy'][i]),
                   'mean_fit_seconds': float(search.cv_results_['mean_fit_time'][i]),
                   'std_fit_seconds': float(search.cv_results_['std_fit_time'][i])}
            row.update({f'fold_{fold}_f1_macro': float(search.cv_results_[f'split{fold}_test_f1_macro'][i]) for fold in range(3)})
            rows.append(row)
        write_csv(directory / 'cv_results.csv', rows)
        selected = rows[search.best_index_]
        save_json(directory / 'selected_config_before_test.json', selected)
        log(f'Selected by CV: {selected}')
        model = search.best_estimator_
        if model.named_steps['svm'].n_iter_ >= config['svm_max_iter']:
            raise RuntimeError('Final SVM did not converge; results are not finalized.')
        bundle = {'pipeline': model, 'classes': classes, 'feature_schema': FEATURE_SCHEMA,
                  'feature_config': FEATURE_CONFIG, 'selected_config': selected,
                  'split_sha256': config['split_sha256'], 'training_data': 'shared train only',
                  'sklearn_version': version('scikit-learn')}
        model_path = directory / 'model.joblib'
        joblib.dump(bundle, model_path, compress=3)
        # Фіксуємо й зберігаємо вибір перед виділенням ознак і оцінюванням на тесті.
        test_features, test_feature_cost = features_with_cache(test_images, 'official_test', root, log)
        validation_pred = model.predict(train_features[split['validation']])
        test_X = np.asarray(test_features[split['test']])
        model.predict(test_X[:128])  # untimed warmup
        model_only_times = []
        with threadpool_limits(limits=1), MemorySampler() as eval_memory:
            for _ in range(3):
                tick = time.perf_counter()
                test_pred = model.predict(test_X)
                model_only_times.append(time.perf_counter() - tick)
            batch_indices = split['test'][:512]
            predict_images(bundle, test_images[batch_indices[:8]])  # warmup
            end_to_end_times = []
            for _ in range(3):
                tick = time.perf_counter()
                batch_pred = predict_images(bundle, test_images[batch_indices])
                end_to_end_times.append(time.perf_counter()-tick)
            np.testing.assert_array_equal(batch_pred, test_pred[:len(batch_indices)])
        results = {'test': metrics(test_labels[split['test']], test_pred, classes),
                   'validation': metrics(train_labels[split['validation']], validation_pred, classes),
                   'selected_by_cv': selected,
                   'resources': {'cv_search_and_refit_seconds': search_seconds,
                       'final_refit_seconds': float(search.refit_time_),
                       'preprocessing_official_train': train_feature_cost,
                       'preprocessing_official_test': test_feature_cost,
                       'training_preprocessing_per_image_ms': train_feature_cost['extraction_seconds'] / len(train_images) * 1000,
                       'sampled_peak_fit_tree_rss_mb': fit_memory.peak / 2**20,
                       'sampled_peak_eval_tree_rss_mb': eval_memory.peak / 2**20,
                       'memory_method': 'Sum process-tree RSS every 100ms; includes data, CV workers and shared-page double-counting; not hardware RAM/VRAM.',
                       'model_bytes': model_path.stat().st_size,
                       'svm_coefficients_and_intercepts': int(model.named_steps['svm'].coef_.size + model.named_steps['svm'].intercept_.size),
                       'selected_feature_count': int(model.named_steps['svm'].coef_.shape[1]),
                       'svm_iterations': int(model.named_steps['svm'].n_iter_),
                       'gpu_used': False, 'cuda_allocated_bytes': 0,
                       'model_only_inference': {'device': 'CPU', 'batch_size': len(test_X), 'repetitions': 3,
                           'seconds': model_only_times, 'median_ms_per_image': float(np.median(model_only_times) / len(test_X) * 1000)},
                       'end_to_end_inference': {'device': 'CPU', 'batch_size': len(batch_indices), 'repetitions': 3,
                           'includes': 'fixed HOG/RGB extraction, selection, scaling and prediction; excludes file decoding/resizing',
                           'seconds': end_to_end_times, 'median_ms_per_image': float(np.median(end_to_end_times) / len(batch_indices) * 1000)}}}
        save_json(directory / 'metrics.json', results)
        per_class = results['test']['classification_report']
        write_csv(directory / 'per_class_metrics.csv', [{'class': c, **per_class[c]} for c in classes])
        for name, truth, pred in [('test', test_labels, test_pred), ('validation', train_labels, validation_pred)]:
            write_csv(directory / f'{name}_predictions.csv', [
                {'original_index': int(index), 'true_label': int(truth[index]), 'predicted_label': int(label),
                 'true_class': classes[int(truth[index])], 'predicted_class': classes[int(label)]}
                for index, label in zip(split[name], pred)])
        plot_results(directory, classes, test_labels[split['test']], test_pred, test_images[split['test']], rows)
        summary = {'model': 'HOG+RGB+LinearSVC' if selected['use_color'] else 'HOG+LinearSVC',
                   'test_accuracy': results['test']['accuracy'], 'test_macro_f1': results['test']['f1_macro'],
                   'validation_accuracy': results['validation']['accuracy'],
                   'cv_macro_f1_mean': selected['cv_f1_macro_mean'], 'cv_macro_f1_std': selected['cv_f1_macro_std'],
                   'final_refit_seconds': results['resources']['final_refit_seconds'],
                   'search_and_refit_seconds': search_seconds, 'peak_fit_tree_rss_mb': fit_memory.peak / 2**20,
                   'model_mb': model_path.stat().st_size / 2**20,
                   'end_to_end_ms_per_image': results['resources']['end_to_end_inference']['median_ms_per_image']}
        write_csv(directory / 'summary.csv', [summary])
        report = f'''# Перша модель: HOG + Linear SVM

Повний запуск CIFAR-10. Train: {len(split['train'])}; validation: {len(split['validation'])}; test: {len(split['test'])}.
Дублікатів між цими split немає за SHA-256; близькі дублікати можуть залишатися. Деталі: data/audit.json.

- Обрано C={selected['C']}; колірні гістограми: {selected['use_color']}.
- 3-fold CV macro-F1: {selected['cv_f1_macro_mean']:.4f} ± {selected['cv_f1_macro_std']:.4f} (std по folds).
- Validation accuracy: {results['validation']['accuracy']:.4f}.
- Test accuracy: {results['test']['accuracy']:.4f}; macro-F1: {results['test']['f1_macro']:.4f}.
- Фінальний refit: {search.refit_time_:.2f} с; повний CV + refit: {search_seconds:.2f} с.
- Розмір моделі: {model_path.stat().st_size/2**20:.2f} MiB.
- Peak RAM: {fit_memory.peak/2**20:.1f} MiB (100ms sampling, сума RSS процесів; shared pages можуть рахуватися двічі).
- Повний inference без читання файлів: {summary['end_to_end_ms_per_image']:.3f} ms/зображення, batch 512, CPU.
- GPU для classical ML не використовується.

Гіперпараметри вибрано лише за CV всередині train. StandardScaler fit виконується окремо в кожному fold.
Після фіксації конфігурації модель навчена на всьому спільному train; validation не додано до train.
Test використано для фінальної оцінки, а не вибору параметрів. Це один test-результат, без штучного test std.
Кешовані ознаки не мають навчених параметрів; для нових зображень extract_features повторює ті самі операції.

Сильні/слабкі класи наведено у per_class_metrics.csv, випадкові помилки - error_examples.png.
Це baseline; висновок про перевагу CNN/transformer можливий лише після їхнього навчання.
'''
        (directory / 'RESULTS_UA.md').write_text(report, encoding='utf-8')
        status.update(status='complete', finished=datetime.now(ZoneInfo('Europe/Kiev')).isoformat(),
                      elapsed_seconds=time.perf_counter()-started, model_sha256=file_hash(model_path))
        save_json(directory / 'status.json', status)
        (root / 'results' / 'latest_classical.txt').write_text(str(directory), encoding='utf-8')
        log(f'COMPLETE accuracy={results["test"]["accuracy"]:.6f}, macro_F1={results["test"]["f1_macro"]:.6f}; {directory}')
        return directory
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        save_json(directory / 'status.json', status)
        log('FAILED: ' + repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    run(parser.parse_args().root)
