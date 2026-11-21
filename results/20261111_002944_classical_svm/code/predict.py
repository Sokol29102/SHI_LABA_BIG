"""Predict local RGB images using the saved classical model."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from baseline import predict_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True, help='Path to a trusted local model.joblib')
    parser.add_argument('images', nargs='+', type=Path)
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    images = []
    for path in args.images:
        with Image.open(path) as image:
            images.append(np.asarray(image.convert('RGB').resize((32, 32), Image.Resampling.BILINEAR)))
    labels = predict_images(bundle, np.stack(images))
    for path, label in zip(args.images, labels):
        print(f'{path}\t{bundle["classes"][int(label)]}')


if __name__ == '__main__':
    main()
