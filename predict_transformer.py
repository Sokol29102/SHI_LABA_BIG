import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from vit import load_model, predict_raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('images', type=Path, nargs='+')
    args = parser.parse_args()
    model, checkpoint = load_model(args.model, args.device)
    images = []
    for path in args.images:
        with Image.open(path) as image:
            images.append(np.asarray(image.convert('RGB').resize((32, 32), Image.Resampling.BILINEAR)))
    raw = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(args.device)
    pred = predict_raw(model, raw, checkpoint['mean'], checkpoint['std'])
    for path, label in zip(args.images, pred):
        print(f'{path}\t{checkpoint["classes"][int(label)]}')


if __name__ == '__main__':
    main()
