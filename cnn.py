"""CIFAR-compatible ResNet-18 and its exact train/inference preprocessing."""
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

ARCHITECTURE = 'resnet18_cifar_v1'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model():
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    model.maxpool = nn.Identity()
    return model


def normalization_stats(images, indices):
    sums, squares, count = np.zeros(3), np.zeros(3), 0
    for start in range(0, len(indices), 1024):
        x = images[indices[start:start + 1024]].astype(np.float64) / 255
        sums += x.sum(axis=(0, 1, 2))
        squares += (x*x).sum(axis=(0, 1, 2))
        count += x.shape[0] * x.shape[1] * x.shape[2]
    if count == 0:
        raise ValueError('Training indices cannot be empty.')
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean*mean, 1e-12))
    return mean.tolist(), std.tolist()


def prepare_batch(raw, mean, std, augment=False):
    if raw.ndim != 4 or tuple(raw.shape[1:]) != (3, 32, 32) or raw.dtype != torch.uint8:
        raise ValueError('Expected uint8 NCHW RGB images with shape (N, 3, 32, 32).')
    x = raw.float().div(255)
    if augment:
        padded = F.pad(x, (4, 4, 4, 4), mode='reflect').permute(0, 2, 3, 1)
        offsets = torch.randint(0, 9, (len(x), 2), device=x.device)
        rows = torch.arange(32, device=x.device)[None, :, None] + offsets[:, 0, None, None]
        cols = torch.arange(32, device=x.device)[None, None, :] + offsets[:, 1, None, None]
        batch = torch.arange(len(x), device=x.device)[:, None, None]
        x = padded[batch, rows, cols].permute(0, 3, 1, 2)
        flip = torch.rand(len(x), 1, 1, 1, device=x.device) < 0.5
        x = torch.where(flip, x.flip(-1), x)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=x.device).view(1, 3, 1, 1)
    return ((x - mean_t) / std_t).contiguous(memory_format=torch.channels_last)


def save_checkpoint(path, checkpoint):
    path = Path(path)
    partial = path.with_suffix(path.suffix + '.part')
    torch.save(checkpoint, partial)
    partial.replace(path)


def load_model(path, device='cuda'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['architecture'] != ARCHITECTURE:
        raise ValueError('Checkpoint is not the expected CIFAR ResNet-18 architecture.')
    model = build_model()
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device=device, memory_format=torch.channels_last).eval()
    return model, checkpoint


@torch.inference_mode()
def predict_raw(model, raw, mean, std, batch_size=256):
    model.eval()
    predictions = []
    for start in range(0, len(raw), batch_size):
        predictions.append(model(prepare_batch(raw[start:start+batch_size], mean, std)).argmax(1))
    return torch.cat(predictions).cpu().numpy()
