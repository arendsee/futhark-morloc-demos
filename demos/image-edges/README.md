# Image edge detection (Futhark + Python)

A real image is loaded in Python (Pillow), edge-detected with a Sobel stencil in
Futhark, and written back as a PNG. The grayscale image crosses the language
boundary as a `Matrix` in **both** directions.

- **Futhark** (`kernels.fut`): the per-pixel Sobel gradient stencil.
- **Python** (`image.py`): decode/encode + grayscale conversion (data access + viz).
- **morloc** (`main.loc`): the `<IO>` pipeline and the type-safe `Matrix h w F32` boundary.

## Run

```bash
morloc make -o nexus main.loc
# sample.png is a small synthetic test image; any grayscale-able image works
./nexus edges sample.png edges.png
```

`edges.png` is the Sobel edge map of the input.
