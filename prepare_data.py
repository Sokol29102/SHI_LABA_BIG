"""Download verified CIFAR-10 and preserve shared leakage-safe splits."""
import hashlib
import json
import pickle
import tarfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
OFFICIAL_MD5 = 'c58f30108f718f92721af3b95e74349a'


def file_hash(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def image_hashes(images):
    return np.array([hashlib.sha256(image.tobytes()).hexdigest() for image in images])


def clean_and_split(train_images, train_labels, test_images, test_labels, seed=42):
    """Test hashes take priority; ambiguous labels and repeated images are removed.

    All indices refer to the original official arrays. No learned preprocessing
    or hyperparameter decision uses test labels.
    """
    train_hash = image_hashes(train_images)
    test_hash = image_hashes(test_images)
    removed = []

    def unique_unambiguous(hashes, labels, source):
        groups = defaultdict(list)
        for i, digest in enumerate(hashes):
            groups[str(digest)].append(i)
        retained, conflicts = [], 0
        for digest, indices in groups.items():
            ambiguous = len({int(labels[i]) for i in indices}) > 1
            if ambiguous:
                conflicts += 1
            else:
                retained.append(indices[0])
            for index in (indices if ambiguous else indices[1:]):
                removed.append({'source': source, 'original_index': index,
                                'reason': 'conflicting_labels' if ambiguous else 'exact_duplicate',
                                'sha256': digest})
        return np.array(retained, dtype=np.int64), conflicts

    test_indices, test_conflicts = unique_unambiguous(test_hash, test_labels, 'test')
    train_indices, train_conflicts = unique_unambiguous(train_hash, train_labels, 'train')
    # Порівнюємо з усіма офіційними тестовими зображеннями, навіть вилученими під час контролю якості.
    test_set = set(test_hash)
    overlaps = [int(i) for i in train_indices if train_hash[i] in test_set]
    for i in overlaps:
        removed.append({'source': 'train', 'original_index': i,
                        'reason': 'official_test_overlap', 'sha256': str(train_hash[i])})
    train_indices = np.array([i for i in train_indices if train_hash[i] not in test_set], dtype=np.int64)
    train_indices, validation = train_test_split(
        train_indices, test_size=0.1, stratify=np.asarray(train_labels)[train_indices], random_state=seed)
    audit = {
        'seed': seed, 'official_train': len(train_images), 'official_test': len(test_images),
        'train': len(train_indices), 'validation': len(validation), 'test': len(test_indices),
        'train_conflicting_hash_groups': train_conflicts,
        'test_conflicting_hash_groups': test_conflicts,
        'train_test_overlap_removed_train': len(overlaps),
        'removed': removed,
        'policy': 'Keep first exact duplicate per official partition, drop conflicting-label groups, '
                  'remove all remaining official-test hashes from train, then stratified 90/10 split.',
        'near_duplicates': 'Not automatically removed; exact-hash cleaning cannot detect crops, flips or near-identical scenes.',
    }
    return {'train': np.sort(train_indices), 'validation': np.sort(validation),
            'test': np.sort(test_indices), 'train_hashes': train_hash, 'test_hashes': test_hash, 'audit': audit}


def prepare(root=ROOT):
    directory = Path(root) / 'data'
    directory.mkdir(exist_ok=True)
    archive = directory / 'cifar-10-python.tar.gz'
    if not archive.exists():
        partial = archive.with_suffix('.gz.part')
        print('Downloading official CIFAR-10 archive...', flush=True)
        tick, count, last = time.perf_counter(), 0, 0
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open('wb') as stream:
            final_url = response.url
            for block in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(block)
                count += len(block)
                if count - last >= 10 * 1024 * 1024:
                    print(f'Downloaded {count / 2**20:.0f} MiB in {time.perf_counter()-tick:.1f}s', flush=True)
                    last = count
        if file_hash(partial, 'md5') != OFFICIAL_MD5:
            raise ValueError('Official archive MD5 mismatch; .part retained for inspection, not loaded.')
        partial.replace(archive)
        save_json(directory / 'download.json', {'url': URL, 'resolved_url': final_url})
    if file_hash(archive, 'md5') != OFFICIAL_MD5:
        raise ValueError('Existing archive fails the official MD5 check.')
    # Після перевірки контрольної суми читаємо лише очікувані файли, не розпаковуючи архів на диск.
    arrays = {}
    with tarfile.open(archive, 'r:gz') as tar:
        def load_member(name):
            with tar.extractfile('cifar-10-batches-py/' + name) as stream:
                return pickle.load(stream, encoding='bytes')
        train = [load_member(f'data_batch_{i}') for i in range(1, 6)]
        test = load_member('test_batch')
        meta = load_member('batches.meta')
        arrays['train_images'] = np.concatenate([p[b'data'] for p in train]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
        arrays['train_labels'] = np.concatenate([p[b'labels'] for p in train]).astype(np.int64)
        arrays['test_images'] = test[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()
        arrays['test_labels'] = np.array(test[b'labels'], dtype=np.int64)
        arrays['classes'] = np.array([c.decode('utf-8') for c in meta[b'label_names']])
    assert arrays['train_images'].shape == (50000, 32, 32, 3)
    assert arrays['test_images'].shape == (10000, 32, 32, 3)
    for source in ['train', 'test']:
        assert arrays[source + '_images'].dtype == np.uint8
        assert np.array_equal(np.unique(arrays[source + '_labels']), np.arange(10))
    split = clean_and_split(arrays['train_images'], arrays['train_labels'], arrays['test_images'], arrays['test_labels'])
    split_path = directory / 'split_indices.npz'
    # Зберігаємо наявний поділ, щоб усі наступні моделі використовували ті самі індекси.
    if split_path.exists():
        with np.load(split_path) as old:
            for key in ['train', 'validation', 'test']:
                if not np.array_equal(old[key], split[key]):
                    raise ValueError('Existing shared split differs; refusing to replace it.')
    else:
        np.savez(split_path, **{k: split[k] for k in ['train', 'validation', 'test']})
    dataset_path = directory / 'dataset.npz'
    if not dataset_path.exists():
        np.savez(dataset_path, **arrays)
    np.savez(directory / 'image_hashes.npz', train=split['train_hashes'], test=split['test_hashes'])
    audit = split['audit']
    audit['class_names'] = arrays['classes'].tolist()
    audit['class_counts'] = {name: np.bincount(arrays['test_labels' if name == 'test' else 'train_labels'][split[name]], minlength=10).tolist()
                             for name in ['train', 'validation', 'test']}
    save_json(directory / 'audit.json', audit)
    save_json(directory / 'provenance.json', {
        'source': 'https://www.cs.toronto.edu/~kriz/cifar.html', 'download_url': URL,
        'retrieved_date': '2026-11-11', 'official_md5': OFFICIAL_MD5,
        'archive_sha256': file_hash(archive), 'dataset_sha256': file_hash(dataset_path),
        'split_sha256': file_hash(split_path),
        'citation': 'Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images. 2009.',
        'license': 'No explicit dataset license identified on the official download page; cite source and do not redistribute the dataset.',
    })
    print('Dataset ready:', {k: len(split[k]) for k in ['train', 'validation', 'test']}, flush=True)
    return arrays, split


if __name__ == '__main__':
    prepare()
