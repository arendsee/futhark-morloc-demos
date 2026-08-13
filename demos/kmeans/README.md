# k-means clustering (Futhark + Python)

k-means where the heavy per-iteration work runs in Futhark and the **convergence
loop lives in morloc**. Each Futhark `step` returns both the new centroids and the
maximum centroid movement as one tuple; morloc iterates until that shift is small.

- **Futhark** (`kernels.fut`): distance matrix, argmin assignment, per-cluster means,
  and the shift metric. Two entries (`step`, `assign`) composed.
- **Python** (`kmeans.py`): synthesize the point cloud + initial centroids (seeded),
  draw the final clustering (data prep + viz).
- **morloc** (`main.loc`): the Lloyd loop, testing the scalar shift returned by `step`;
  labels come back as a `Vector n I32`.

## Run

```bash
morloc make -o nexus main.loc
./nexus cluster 300 kmeans.png
```

`kmeans.png` shows the points colored by cluster with the converged centroids.
