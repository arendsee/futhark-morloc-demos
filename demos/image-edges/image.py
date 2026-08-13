def load_gray(path):
    import numpy as np
    from PIL import Image
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0

def save_png(mat, path):
    import numpy as np
    from PIL import Image
    a = np.asarray(mat, dtype=np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi > lo:
        a = (a - lo) / (hi - lo)
    Image.fromarray((a * 255.0).astype("uint8")).save(path)
    return path
