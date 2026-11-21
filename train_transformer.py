"""Fine-tune pinned DeiT-Tiny on shared CIFAR-10 splits, then evaluate frozen models."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
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

import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score

from cnn import save_checkpoint, set_seed
from prepare_data import ROOT, file_hash, save_json
from train_baseline import MemorySampler, write_csv
from train_cnn import configure_gpu, merge_resource_peaks, recover_best, now
from vit import ARCHITECTURE, MODEL_NAME, MEAN, STD, build_model, download_pretrained, prepare_batch, strict_evaluation


def atomic_json(path, value):
    path = Path(path)
    partial = path.with_suffix(path.suffix+'.part')
    save_json(partial, value)
    partial.replace(path)


def make_optimizer(model):
    head = list(model.head.parameters())
    head_ids = {id(parameter) for parameter in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    return torch.optim.AdamW([{'params': backbone, 'lr': 5e-5}, {'params': head, 'lr': 5e-4}], weight_decay=0.05)


def make_scheduler(optimizer, epochs, steps_per_epoch):
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=steps_per_epoch)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, (epochs-1)*steps_per_epoch))
    return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[steps_per_epoch])


def profile(batch_size=128):
    configure_gpu()
    path, provenance = download_pretrained()
    with np.load(ROOT/'data/dataset.npz') as data:
        images = data['train_images']
    with np.load(ROOT/'data/split_indices.npz') as split:
        indices = split['train']
    raw = torch.from_numpy(images[indices[:batch_size]].copy()).permute(0, 3, 1, 2).to('cuda')
    labels = torch.arange(batch_size, device='cuda') % 10
    set_seed(42)
    model = build_model(path).to('cuda').train()
    optimizer = make_optimizer(model)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = nn.functional.cross_entropy(model(prepare_batch(raw, augment=True)), labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = nn.functional.cross_entropy(model(prepare_batch(raw, augment=True)), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    torch.cuda.synchronize()
    step_seconds = (time.perf_counter()-tick)/10
    allocated, reserved = torch.cuda.max_memory_allocated()/2**20, torch.cuda.max_memory_reserved()/2**20
    model.eval()
    with strict_evaluation(), torch.inference_mode():
        model(prepare_batch(raw))
        torch.cuda.synchronize()
        tick = time.perf_counter()
        for _ in range(3):
            model(prepare_batch(raw))
        torch.cuda.synchronize()
        eval_seconds = (time.perf_counter()-tick)/3
    result = {'gpu': torch.cuda.get_device_name(0), 'batch_size': batch_size, 'training_step_seconds': step_seconds,
        'training_images_per_second': batch_size/step_seconds,
        'estimated_training_epoch_seconds': step_seconds*math.ceil(45000/batch_size),
        'estimated_validation_seconds': eval_seconds*math.ceil(5000/batch_size),
        'peak_training_cuda_allocated_mib': allocated, 'peak_training_cuda_reserved_mib': reserved,
        'pretrained_revision': provenance['revision'],
        'scope': 'Throwaway repeated training-only batch, synthetic labels, 3 warmup + 10 training steps. Strict FP32 eval timing is a repeated training-image batch, not test. Excludes loading, full dataset residency and checkpoint I/O.'}
    atomic_json(ROOT/'results/transformer_gpu_profile.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


@torch.inference_mode()
def evaluate(model, raw, labels, indices, batch_size):
    model.eval()
    loss_sum = torch.zeros((), device='cuda')
    predictions = []
    with strict_evaluation():
        for batch in indices.split(batch_size):
            output = model(prepare_batch(raw[batch]))
            loss_sum += nn.functional.cross_entropy(output, labels[batch], reduction='sum')
            predictions.append(output.argmax(1))
    pred = torch.cat(predictions).cpu().numpy()
    truth = labels[indices].cpu().numpy()
    return float(loss_sum.item()/len(indices)), pred, truth


def train_epoch(model, optimizer, scheduler, raw, labels, indices, batch_size):
    model.train()
    order = indices[torch.randperm(len(indices), device='cuda')]
    loss_sum, correct = torch.zeros((), device='cuda'), torch.zeros((), device='cuda')
    for batch in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(prepare_batch(raw[batch], augment=True))
            loss = nn.functional.cross_entropy(output, labels[batch])
        if not torch.isfinite(loss).item():
            raise FloatingPointError('Non-finite transformer training loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        scheduler.step()
        loss_sum += loss.detach()*len(batch)
        correct += (output.argmax(1) == labels[batch]).sum()
    return float(loss_sum.item()/len(indices)), float(correct.item()/len(indices))


def train_seed(directory, config, raw, labels, train_idx, val_idx, pretrained_path, log):
    seed = config['seed']
    target = directory/f'seed_{seed}'
    target.mkdir(exist_ok=True)
    set_seed(seed)
    model = build_model(pretrained_path).to('cuda')
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer, config['epochs'], math.ceil(len(train_idx)/config['batch_size']))
    best, best_checkpoint, history, epoch_start, elapsed_prior = -1., None, [], 1, 0.
    previous_peaks = {'rss_mib': 0., 'allocated_mib': 0., 'reserved_mib': 0., 'complete': True}
    if (target/'last.pt').exists():
        last = torch.load(target/'last.pt', map_location='cpu', weights_only=True)
        if last['config'] != config:
            raise ValueError('Resume config differs; refusing to change experiment.')
        model.load_state_dict(last['model_state'])
        optimizer.load_state_dict(last['optimizer_state'])
        scheduler.load_state_dict(last['scheduler_state'])
        best_checkpoint = recover_best(last, target/'best.pt')
        best, history, epoch_start, elapsed_prior = last['best_validation_f1'], last['history'], last['epoch']+1, last['training_seconds']
        previous_peaks = last['resource_peaks']
        torch.set_rng_state(last['torch_rng_state'])
        torch.cuda.set_rng_state(last['cuda_rng_state'])
        del last
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with MemorySampler() as memory:
        for epoch in range(epoch_start, config['epochs']+1):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            lrs = [group['lr'] for group in optimizer.param_groups]
            train_loss, train_accuracy = train_epoch(model, optimizer, scheduler, raw, labels, train_idx, config['batch_size'])
            torch.cuda.synchronize()
            train_seconds = time.perf_counter()-tick
            val_loss, pred, truth = evaluate(model, raw, labels, val_idx, config['batch_size'])
            val_f1 = float(f1_score(truth, pred, average='macro'))
            history.append({'epoch': epoch, 'backbone_lr_start': lrs[0], 'head_lr_start': lrs[1],
                'train_loss': train_loss, 'train_accuracy_augmented': train_accuracy, 'validation_loss': val_loss,
                'validation_accuracy': float(np.mean(pred == truth)), 'validation_f1_macro': val_f1, 'train_seconds': train_seconds})
            metadata = {'architecture': ARCHITECTURE, 'seed': seed, 'epoch': epoch, 'mean': MEAN, 'std': STD,
                        'classes': config['classes'], 'config': config, 'validation_f1_macro': val_f1}
            if val_f1 > best:
                best = val_f1
                best_checkpoint = {**metadata, 'model_state': {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}}
                save_checkpoint(target/'best.pt', best_checkpoint)
            peaks = merge_resource_peaks(previous_peaks, {'rss_mib': memory.peak/2**20,
                'allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                'reserved_mib': torch.cuda.max_memory_reserved()/2**20, 'complete': True})
            save_checkpoint(target/'last.pt', {**metadata, 'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
                'best_checkpoint': best_checkpoint, 'best_validation_f1': best, 'history': history, 'resource_peaks': peaks,
                'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state(),
                'training_seconds': elapsed_prior+time.perf_counter()-started})
            write_csv(target/'history.csv', history)
            atomic_json(target/'history.json', history)
            log(f'seed={seed} epoch={epoch}/{config["epochs"]} loss={train_loss:.4f} val_acc={history[-1]["validation_accuracy"]:.4f} val_F1={val_f1:.4f} train_s={train_seconds:.2f}')
    torch.cuda.synchronize()
    peaks = merge_resource_peaks(previous_peaks, {'rss_mib': memory.peak/2**20,
        'allocated_mib': torch.cuda.max_memory_allocated()/2**20, 'reserved_mib': torch.cuda.max_memory_reserved()/2**20, 'complete': True})
    resources = {'training_wall_seconds': elapsed_prior+time.perf_counter()-started,
        'train_epoch_seconds_sum': sum(row['train_seconds'] for row in history),
        'sampled_peak_process_tree_rss_mib': peaks['rss_mib'], 'peak_cuda_allocated_mib': peaks['allocated_mib'],
        'peak_cuda_reserved_mib': peaks['reserved_mib'], 'peak_measurement_complete': peaks['complete'],
        'parameters_total': sum(p.numel() for p in model.parameters()),
        'parameters_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'best_validation_f1': best, 'checkpoint_bytes': (target/'best.pt').stat().st_size,
        'scope': 'Local fine-tuning incl validation/checkpoint I/O; excludes original ImageNet pretraining. Peak VRAM includes resident train images. Tree RSS sampled 100ms may overcount shared pages.'}
    atomic_json(target/'training_resources.json', resources)
    del model, optimizer, scheduler, best_checkpoint
    gc.collect()
    torch.cuda.empty_cache()
    return resources


def run(epochs=10, batch_size=128, resume=None):
    configure_gpu()
    path, provenance = download_pretrained()
    if resume:
        directory = Path(resume)
        config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
        if json.loads((directory/'status.json').read_text(encoding='utf-8'))['status'] == 'complete':
            raise ValueError('Run already complete; use verify_transformer.py.')
    else:
        directory = ROOT/'results'/datetime.now(ZoneInfo('Europe/Kiev')).strftime('%Y%m%d_%H%M%S_transformer_deit')
        directory.mkdir(exist_ok=False)
        config = None
    def log(message):
        line = now()+' '+message
        print(line, flush=True)
        with (directory/'run.log').open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
    status = {'status': 'training', 'started': now()}
    atomic_json(directory/'status.json', status)
    try:
        dataset_hash, split_hash = file_hash(ROOT/'data/dataset.npz'), file_hash(ROOT/'data/split_indices.npz')
        original = json.loads((ROOT/'data/provenance.json').read_text(encoding='utf-8'))
        if dataset_hash != original['dataset_sha256'] or split_hash != original['split_sha256']:
            raise ValueError('Data or shared split changed.')
        with np.load(ROOT/'data/dataset.npz') as data:
            images, labels, classes = data['train_images'], data['train_labels'], data['classes'].tolist()
        with np.load(ROOT/'data/split_indices.npz') as split:
            train_indices, val_indices = split['train'], split['validation']
        if config is None:
            config = {'architecture': ARCHITECTURE, 'model_name': MODEL_NAME, 'pretrained': provenance,
                'epochs': epochs, 'batch_size': batch_size, 'seeds': [42, 123, 2026],
                'optimizer': 'AdamW', 'backbone_lr': 5e-5, 'head_lr': 5e-4, 'weight_decay': 0.05,
                'scheduler': 'one-epoch linear warmup from 0.1x, then cosine, stepped each batch',
                'gradient_clip_norm': 1., 'training_precision': 'BF16 autocast, FP32 weights',
                'evaluation_precision': 'strict FP32, TF32 disabled, deterministic algorithms and cuDNN; no augmentation',
                'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                'checkpoint_selection': 'earliest maximum validation macro-F1',
                'mean': MEAN, 'std': STD, 'normalization_source': 'published pretrained ImageNet model constants; no fit on target test',
                'preprocessing': 'Per-image reflect-padded crop/flip at 32x32 for train only, bicubic direct resize to224 and clamp[0,1], then ImageNet normalization. No center crop. No details added by upsampling.',
                'classes': classes, 'dataset_sha256': dataset_hash, 'split_sha256': split_hash,
                'sizes': {'train': len(train_indices), 'validation': len(val_indices), 'test': 10000},
                'gpu': torch.cuda.get_device_name(0), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'python': sys.version,
                'packages': {name: version(name) for name in ['torch', 'torchvision', 'timm', 'huggingface-hub', 'safetensors', 'numpy', 'scikit-learn']}}
            atomic_json(directory/'config.json', config)
        elif config['dataset_sha256'] != dataset_hash or config['split_sha256'] != split_hash or config['pretrained'] != provenance:
            raise ValueError('Cannot resume with different data or pretrained weights.')
        code = directory/'code'
        code.mkdir(exist_ok=True)
        for name in ['vit.py', 'train_transformer.py', 'transformer_results.py', 'verify_transformer.py',
                     'predict_transformer.py', 'test_vit.py', 'cnn.py', 'train_cnn.py', 'train_baseline.py',
                     'baseline.py', 'prepare_data.py', 'requirements-transformer-lock.txt']:
            if (ROOT/name).exists() and not (code/name).exists():
                shutil.copy2(ROOT/name, code/name)
        raw = torch.from_numpy(images).permute(0, 3, 1, 2).to('cuda')
        y = torch.from_numpy(labels).to('cuda')
        train_idx, val_idx = torch.from_numpy(train_indices).to('cuda'), torch.from_numpy(val_indices).to('cuda')
        resources = {}
        for seed in config['seeds']:
            resource_path = directory/f'seed_{seed}/training_resources.json'
            if resource_path.exists():
                resources[seed] = json.loads(resource_path.read_text(encoding='utf-8'))
                log(f'Seed {seed} already complete; preserving it.')
            else:
                log(f'Starting seed {seed}: {config["epochs"]} epochs, {len(train_indices)} training images')
                resources[seed] = train_seed(directory, {**config, 'seed': seed}, raw, y, train_idx, val_idx, path, log)
            last = torch.load(directory/f'seed_{seed}/last.pt', map_location='cpu', weights_only=True)
            recover_best(last, directory/f'seed_{seed}/best.pt')
            del last
        atomic_json(directory/'all_selections_frozen_before_test.json', {'time': now(), 'seeds': {
            str(seed): {'checkpoint_sha256': file_hash(directory/f'seed_{seed}/best.pt'),
                        'best_validation_f1': resources[seed]['best_validation_f1']} for seed in config['seeds']}})
        del raw, y, train_idx, val_idx
        gc.collect()
        torch.cuda.empty_cache()
        status.update(status='evaluating_frozen_checkpoints')
        atomic_json(directory/'status.json', status)
        from transformer_results import finalize
        aggregate = finalize(directory, config, resources, log)
        status.update(status='complete', finished=now())
        atomic_json(directory/'status.json', status)
        (ROOT/'results/latest_transformer.txt').write_text(str(directory), encoding='utf-8')
        log('COMPLETE '+json.dumps(aggregate))
        return directory
    except BaseException as error:
        status.update(status='failed', error=repr(error), updated=now())
        atomic_json(directory/'status.json', status)
        log('FAILED '+repr(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.profile_only:
        profile(args.batch_size)
    else:
        if args.epochs < 2 or args.batch_size < 1:
            parser.error('Use epochs>=2 and positive batch size.')
        run(args.epochs, args.batch_size, args.resume)
