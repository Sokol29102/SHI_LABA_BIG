"""Pinned DeiT-Tiny backbone, CIFAR preprocessing, and strict evaluation policy."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import json
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.nn import functional as F
import timm

from cnn import prepare_batch as cifar_augment
from prepare_data import ROOT, file_hash, save_json

ARCHITECTURE = 'deit_tiny_cifar_transfer_v1'
MODEL_NAME = 'deit_tiny_patch16_224.fb_in1k'
REPOSITORY = 'timm/' + MODEL_NAME
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def download_pretrained():
    os.environ.setdefault('HF_HOME', str(ROOT/'.hf_cache'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    from huggingface_hub import HfApi, hf_hub_download
    directory = ROOT/'data/pretrained/deit_tiny'
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory/'provenance.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        for name, digest in saved['sha256'].items():
            if file_hash(directory/name) != digest:
                raise ValueError('Pinned pretrained artifact changed: '+name)
        return directory/'model.safetensors', saved
    info = HfApi().model_info(REPOSITORY, token=False)
    revision = info.sha
    files = {}
    names = {item.rfilename for item in info.siblings}
    for name in ['model.safetensors', 'config.json', 'README.md']:
        if name not in names:
            if name != 'README.md':
                raise ValueError('Upstream repository missing '+name)
            continue
        downloaded = hf_hub_download(REPOSITORY, name, revision=revision, local_dir=directory, token=False)
        files[name] = file_hash(downloaded)
    saved = {'repository': REPOSITORY, 'revision': revision, 'sha256': files,
             'license': 'apache-2.0 (published model card)', 'pretraining_dataset': 'ImageNet-1K',
             'source': 'https://huggingface.co/'+REPOSITORY, 'accessed': '2026-11-11'}
    save_json(manifest, saved)
    return directory/'model.safetensors', saved


def build_model(pretrained_path=None):
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=10)
    if pretrained_path is not None:
        path = Path(pretrained_path)
        if path.suffix == '.safetensors':
            from safetensors.torch import load_file
            state = load_file(str(path))
        else:
            state = torch.load(path, map_location='cpu', weights_only=True)
        backbone = {key: value for key, value in state.items() if not key.startswith('head.')}
        missing = model.load_state_dict(backbone, strict=False)
        if set(missing.missing_keys) != {'head.weight', 'head.bias'} or missing.unexpected_keys:
            raise ValueError('Incomplete or incompatible pretrained backbone: '+str(missing))
    return model


def prepare_batch(raw, mean=MEAN, std=STD, augment=False):
    x = cifar_augment(raw, [0.]*3, [1.]*3, augment=augment)
    x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
    m = torch.as_tensor(mean, device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.as_tensor(std, device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
    return ((x-m)/s).contiguous()


@contextmanager
def strict_evaluation():
    previous = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic,
                torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision(),
                torch.are_deterministic_algorithms_enabled())
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic = previous[:2]
        torch.backends.cudnn.allow_tf32 = previous[2]
        torch.set_float32_matmul_precision(previous[3])
        torch.use_deterministic_algorithms(previous[4])


def load_model(path, device='cuda'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['architecture'] != ARCHITECTURE:
        raise ValueError('Checkpoint architecture is not the expected DeiT-Tiny.')
    model = build_model()
    model.load_state_dict(checkpoint['model_state'])
    return model.to(device).eval(), checkpoint


@torch.inference_mode()
def predict_raw(model, raw, mean=MEAN, std=STD, batch_size=128):
    model.eval()
    result = []
    with strict_evaluation():
        for batch in raw.split(batch_size):
            result.append(model(prepare_batch(batch, mean, std)).argmax(1))
    return torch.cat(result).cpu().numpy()


if __name__ == '__main__':
    path, metadata = download_pretrained()
    print(path)
    print(json.dumps(metadata, indent=2))
