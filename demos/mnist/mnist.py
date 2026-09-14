# Data access + visualization for the MLP demo. Downloads the real MNIST digits
# (28x28 = 784 pixels, flattened and scaled to [0,1]) on first use and caches the
# raw IDX files under ./data. Each function rebuilds the same seeded subset and
# returns one piece, so train/test/labels stay consistent. Futhark trains; Python
# draws the confusion matrix.

import gzip
import os
import struct
import urllib.request

import numpy as np

_K, _D = 10, 784  # 10 classes, 28*28 pixels

# Subset sizes kept small so the full-batch epoch loop stays snappy in the demo.
# Bump these (up to 60000 train / 10000 test) for a fuller run.
_N_TRAIN, _N_TEST = 3000, 1000

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(name):
    fname = _FILES[name]
    path = os.path.join(_DATA_DIR, fname)
    if not os.path.exists(path):
        os.makedirs(_DATA_DIR, exist_ok=True)
        req = urllib.request.Request(_MIRROR + fname, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    return path


def _read_images(name):
    with gzip.open(_download(name), "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, magic
        buf = f.read(n * rows * cols)
    x = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows * cols)
    return (x.astype(np.float32) / 255.0)


def _read_labels(name):
    with gzip.open(_download(name), "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, magic
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int32)


def _dataset():
    Xtr = _read_images("train_images")
    ytr = _read_labels("train_labels")
    Xte = _read_images("test_images")
    yte = _read_labels("test_labels")

    # Deterministic subset so every function call agrees on the same split.
    rng = np.random.default_rng(0)
    itr = rng.choice(len(Xtr), size=min(_N_TRAIN, len(Xtr)), replace=False)
    ite = rng.choice(len(Xte), size=min(_N_TEST, len(Xte)), replace=False)
    return (
        np.ascontiguousarray(Xtr[itr]),
        np.ascontiguousarray(ytr[itr]),
        np.ascontiguousarray(Xte[ite]),
        np.ascontiguousarray(yte[ite]),
    )


def train_x():
    return _dataset()[0]


def train_y():
    ytr = _dataset()[1]
    Y = np.zeros((len(ytr), _K), np.float32)
    Y[np.arange(len(ytr)), ytr] = 1.0
    return Y


def test_x():
    return _dataset()[2]


def test_labels():
    return _dataset()[3].astype(np.int32)


def init_w1():
    return np.random.default_rng(1).normal(0.0, 0.05, size=(_D, 32)).astype(np.float32)


def init_w2():
    return np.random.default_rng(2).normal(0.0, 0.1, size=(32, _K)).astype(np.float32)


def plot_confusion(true_labels, preds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.asarray(true_labels).astype(int)
    p = np.asarray(preds).astype(int)
    cm = np.zeros((_K, _K), int)
    for a, b in zip(t, p):
        cm[a, b] += 1
    acc = float((t == p).mean())
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(_K))
    ax.set_yticks(range(_K))
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("MLP confusion matrix (Futhark train)  acc=%.1f%%" % (100 * acc))
    for i in range(_K):
        for j in range(_K):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path
