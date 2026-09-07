
from pathlib import Path
import numpy as np
from PIL import Image
import colorsys

SRC = Path()
DST = Path()
DST.mkdir(parents=True, exist_ok=True)


MAP = {}

WHITE = np.array([255, 255, 255], dtype=np.uint8)
BLACK = np.array([0, 0, 0], dtype=np.uint8)
BLUE  = np.array([0, 0, 255], dtype=np.uint8)
RED   = np.array([255, 0, 0], dtype=np.uint8)


def rgb_to_hsv_np(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    d = mx - mn
    v = mx
    s = np.where(mx > 0, d / np.where(mx == 0, 1, mx), 0)
    h = np.zeros_like(v)
    mask = d > 0
    # red-dominant
    rmask = mask & (mx == r)
    gmask = mask & (mx == g)
    bmask = mask & (mx == b)
    h[rmask] = ((g[rmask] - b[rmask]) / d[rmask]) % 6
    h[gmask] = ((b[gmask] - r[gmask]) / d[gmask]) + 2
    h[bmask] = ((r[bmask] - g[bmask]) / d[bmask]) + 4
    h = h * 60.0 
    return h, s, v


def classify(img_rgb):
   
    h, s, v = rgb_to_hsv_np(img_rgb)
    out = np.zeros(img_rgb.shape[:2], dtype=np.uint8)  # default white

   
    black = v < 0.35
  
    red = ((h < 25) | (h > 335)) & (s > 0.45) & (v > 0.25)
   
    blue = (h > 160) & (h < 260) & (s > 0.35) & (v > 0.25)

  
    out[blue] = 2
    out[red] = 3
    out[black] = 1
    return out


def cleanup(idx):
   
    from scipy import ndimage
    cleaned = idx.copy()
   
    for label in (1, 2, 3):
        mask = idx == label
        lbl, n = ndimage.label(mask)
        if n == 0:
            continue
        sizes = ndimage.sum(mask, lbl, range(1, n + 1))
        small = np.where(sizes < 6)[0] + 1
        if len(small):
            drop = np.isin(lbl, small)
            cleaned[drop & mask] = 0

  
    struct = ndimage.generate_binary_structure(2, 2)  # 3x3
    closed_masks = {}
    for label, iters in ((2, 1), (3, 1), (1, 2)):  # black gets slightly stronger close
        m = cleaned == label
        m = ndimage.binary_closing(m, structure=struct, iterations=iters)
        closed_masks[label] = m

    out = np.zeros_like(cleaned)
    out[closed_masks[2]] = 2
    out[closed_masks[3]] = 3
    out[closed_masks[1]] = 1
    return out


def render(idx):
    out = np.full((*idx.shape, 3), 255, dtype=np.uint8)
    out[idx == 1] = BLACK
    out[idx == 2] = BLUE
    out[idx == 3] = RED
    return out


def process(src_path, dst_path):
    img = np.array(Image.open(src_path).convert("RGB"))
    idx = classify(img)
    try:
        idx = cleanup(idx)
    except ImportError:
        pass
    out = render(idx)
    Image.fromarray(out).save(dst_path)
    # stats
    total = idx.size
    for name, v in [("white", 0), ("black", 1), ("blue", 2), ("red", 3)]:
        c = int((idx == v).sum())
        print(f"  {name:6} {c:>8}  ({c/total*100:.2f}%)")


for src_name, dst_name in MAP.items():
    src = SRC / src_name
    dst = DST / dst_name
    print(f"{src.name} -> {dst}")
    process(src, dst)

print("done, output in", DST)
