# MLP training (Futhark + Python)

A small neural network (64 -> 32 -> 10) trained by gradient descent. Every matmul
-- forward, backprop, weight update, and prediction -- runs in Futhark; morloc owns
the epoch loop; Python supplies the data and draws a confusion matrix.

- **Futhark** (`kernels.fut`): `train_step` does one full forward+backward+SGD update
  and returns `(W1', W2', loss)` as one tuple (forward pass computed once); `predict`
  returns argmax classes.
- **Python** (`mnist.py`): synthesize a deterministic 10-class dataset (stand-in for
  MNIST digits) and plot the confusion matrix (data access + viz).
- **morloc** (`main.loc`): the epoch loop, threading the two weight matrices
  (`Matrix 64 32 F32`, `Matrix 32 10 F32`) and dropping the per-step loss.

The `matmul :: Matrix m k -> Matrix k n -> Matrix m n` dimension linkage type-checks
the network shapes across the Python/Futhark boundary at compile time.

## Run

```bash
morloc make -o nexus main.loc
./nexus train 150 mnist.png     # 150 epochs
```

`mnist.png` is the confusion matrix over a held-out test set.
