# Data access + visualization for the MLP demo. No network here, so we synthesize
# a deterministic 10-class dataset (a random template per class + noise) that
# stands in for MNIST-style digits. Each function rebuilds the same seeded split
# and returns one piece, so train/test/labels stay consistent. Futhark trains;
# Python draws the confusion matrix.

import numpy as np

_K, _D = 10, 64


def _dataset():
    rng = np.random.default_rng(0)
    templates = rng.normal(0.0, 1.0, size=(_K, _D)).astype(np.float32)

    def gen(nper):
        xs, ys = [], []
        for c in range(_K):
            xs.append(templates[c] + rng.normal(0.0, 0.6, size=(nper, _D)).astype(np.float32))
            ys += [c] * nper
        return np.concatenate(xs, 0).astype(np.float32), np.array(ys, np.int32)

    Xtr, ytr = gen(80)
    Xte, yte = gen(20)
    return Xtr, ytr, Xte, yte


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
    return np.random.default_rng(1).normal(0.0, 0.1, size=(64, 32)).astype(np.float32)


def init_w2():
    return np.random.default_rng(2).normal(0.0, 0.1, size=(32, 10)).astype(np.float32)


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
