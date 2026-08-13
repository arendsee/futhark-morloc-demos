# Data preparation + visualization for the k-means demo. Python synthesizes the
# point cloud and initial centroids (seeded, so runs are deterministic) and draws
# the final clustering; Futhark does the per-iteration distance/assignment/mean
# work. Heavy imports are lazy.


def make_points(n):
    import numpy as np
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 6.0]], dtype=np.float32)
    per = max(1, n // len(centers))
    pts = [rng.normal(c, 0.8, size=(per, 2)) for c in centers]
    return np.concatenate(pts, axis=0).astype(np.float32)


def init_centroids():
    import numpy as np
    # deliberately offset from the true blob centers so iterations visibly move
    return np.array([[1.0, 1.0], [5.0, 4.0], [1.0, 5.0]], dtype=np.float32)


def plot_clusters(pts, labels, cent, path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = np.asarray(pts)
    labels = np.asarray(labels)
    cent = np.asarray(cent)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="tab10", s=12, alpha=0.7)
    ax.scatter(cent[:, 0], cent[:, 1], c="black", marker="X", s=200,
               edgecolors="white", linewidths=1.5, zorder=3)
    ax.set_title("k-means: Futhark compute, Python visualization")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path
