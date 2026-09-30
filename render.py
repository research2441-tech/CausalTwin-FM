"""Procedural canopy image renderer (48x48 RGB, float in [0,1]).

Visible content: soil background, leaf blobs whose cover follows LAI, leaf tone
following nitrogen/water status (chlorosis is a lesion look-alike), tan target
lesions with a dark ring for the infectious class, dark necrotic patches for
removed tissue, insect-feeding holes as distractors, and camera nuisances
(white balance, exposure, blur, sensor noise).  Latent infections are
invisible by construction.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

S = 48
YY, XX = np.mgrid[0:S, 0:S].astype(np.float32)
CAMERAS = {  # region -> (rgb gains, gamma)
    "HYD": (np.array([1.00, 1.00, 1.00]), 1.00),
    "COR": (np.array([1.04, 1.00, 0.95]), 0.97),
    "TUN": (np.array([1.12, 0.97, 0.86]), 0.88),   # unseen camera profile
}


def _disk(img, cy, cx, r, col, soft=0.6):
    d = np.sqrt((YY - cy) ** 2 + (XX - cx) ** 2)
    a = np.clip((r - d) / soft + 0.5, 0, 1)[..., None]
    img *= (1 - a)
    img += a * col


def render(rng, lai, I, R, chlor, holes_rate, region):
    img = np.empty((S, S, 3), np.float32)
    soil = np.array([0.45, 0.33, 0.22]) * (0.9 + 0.2 * rng.random())
    img[:] = soil + 0.05 * rng.standard_normal((S, S, 1))
    fc = 1 - np.exp(-0.75 * lai)
    leafmask = np.zeros((S, S), np.float32)
    nleaf = int(4 + 40 * fc)
    for _ in range(nleaf):
        if leafmask.mean() > fc:
            break
        cy, cx = rng.random(2) * S
        a, b = 3 + 5 * rng.random(), 3 + 5 * rng.random()
        th = rng.random() * np.pi
        yy, xx = YY - cy, XX - cx
        u = (xx * np.cos(th) + yy * np.sin(th)) / a
        v = (-xx * np.sin(th) + yy * np.cos(th)) / b
        leafmask = np.maximum(leafmask, np.clip((1 - (u * u + v * v)) * 3, 0, 1))
    green = np.array([0.22, 0.52, 0.18]) * (1 - chlor) + np.array([0.62, 0.62, 0.22]) * chlor
    shade = 0.85 + 0.3 * gaussian_filter(rng.random((S, S)), 3)
    leafcol = green[None, None, :] * shade[..., None]
    img = img * (1 - leafmask[..., None]) + leafcol * leafmask[..., None]
    leaf_px = np.argwhere(leafmask > 0.7)
    if len(leaf_px):
        area = len(leaf_px)
        n_i = rng.poisson(area * I / 6.0)
        n_r = rng.poisson(area * R / 10.0)
        n_h = rng.poisson(holes_rate)
        for _ in range(min(n_r, 120)):
            cy, cx = leaf_px[rng.integers(len(leaf_px))]
            _disk(img, cy, cx, 1.2 + 1.2 * rng.random(), np.array([0.20, 0.12, 0.07]))
        for _ in range(min(n_i, 160)):
            cy, cx = leaf_px[rng.integers(len(leaf_px))]
            r = 0.9 + 0.8 * rng.random()
            _disk(img, cy, cx, r + 0.7, np.array([0.30, 0.20, 0.10]))
            _disk(img, cy, cx, r, np.array([0.70, 0.55, 0.33]))
        for _ in range(n_h):
            cy, cx = leaf_px[rng.integers(len(leaf_px))]
            _disk(img, cy, cx, 0.8 + 0.8 * rng.random(), soil * 1.1)
    gains, gamma = CAMERAS[region]
    expo = 0.8 + 0.4 * rng.random()
    img = np.clip(img * gains * expo, 0, 1) ** gamma
    sig = rng.random() * 0.8
    if sig > 0.3:
        img = gaussian_filter(img, (sig, sig, 0))
    img = img + 0.02 * rng.standard_normal(img.shape)
    return np.clip(img, 0, 1).astype(np.float32)


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(2, 5, figsize=(10, 4))
    cases = [(0.3, 0, 0), (1.5, 0.005, 0), (2.5, 0.02, 0.01), (3, 0.06, 0.04), (3, 0.15, 0.2)]
    for j, (l, i, r) in enumerate(cases):
        ax[0, j].imshow(render(rng, l, i, r, 0.1, 2, "COR")); ax[0, j].set_title(f"{i+r:.3f}")
        ax[1, j].imshow(render(rng, l, i, r, 0.5, 6, "TUN"))
    for a in ax.flat: a.axis("off")
    plt.savefig("/tmp/claude-0/-home-claude/411fb3e8-ad7b-5e28-82db-77c456506249/scratchpad/render_test.png", dpi=100)
