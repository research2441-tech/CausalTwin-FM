"""Masked-autoencoder pretraining of a compact ViT canopy encoder (JAX), then a
frozen-feature lesion-severity head with heteroscedastic output.

Pretraining uses unlabelled images of the training and calibration seasons
only; test, warming and geographic images are never seen during pretraining.
Outputs data/emb.npz with per-image embedding, severity mean and log-sd.
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

P_SZ, IMG, D, DEPTH, HEADS = 6, 48, 64, 4, 4
NTOK = (IMG // P_SZ) ** 2
PDIM = P_SZ * P_SZ * 3


def patchify(x):  # (B,48,48,3)->(B,64,108)
    B = x.shape[0]
    x = x.reshape(B, IMG // P_SZ, P_SZ, IMG // P_SZ, P_SZ, 3).transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(B, NTOK, PDIM)


def dense(key, i, o):
    return dict(w=jax.random.normal(key, (i, o)) * np.sqrt(2.0 / (i + o)), b=jnp.zeros(o))


def block_init(key, d):
    k = jax.random.split(key, 4)
    return dict(qkv=dense(k[0], d, 3 * d), o=dense(k[1], d, d), f1=dense(k[2], d, 2 * d),
                f2=dense(k[3], 2 * d, d), g1=jnp.ones(d), g2=jnp.ones(d))


def ln(x, g):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * g


def block(p, x):
    B, N, d = x.shape
    h = ln(x, p["g1"])
    qkv = h @ p["qkv"]["w"] + p["qkv"]["b"]
    q, k, v = jnp.split(qkv.reshape(B, N, 3, HEADS, d // HEADS), 3, axis=2)
    q, k, v = q[:, :, 0], k[:, :, 0], v[:, :, 0]
    a = jax.nn.softmax(jnp.einsum("bnhd,bmhd->bhnm", q, k) / np.sqrt(d // HEADS), -1)
    o = jnp.einsum("bhnm,bmhd->bnhd", a, v).reshape(B, N, d)
    x = x + o @ p["o"]["w"] + p["o"]["b"]
    h = ln(x, p["g2"])
    return x + jax.nn.gelu(h @ p["f1"]["w"] + p["f1"]["b"]) @ p["f2"]["w"] + p["f2"]["b"]


def init(key):
    k = jax.random.split(key, 12)
    return dict(emb=dense(k[0], PDIM, D), pos=0.02 * jax.random.normal(k[1], (NTOK, D)),
                enc=[block_init(k[2 + i], D) for i in range(DEPTH)], gE=jnp.ones(D),
                dec_in=dense(k[7], D, D), mask_tok=jnp.zeros(D),
                dpos=0.02 * jax.random.normal(k[8], (NTOK, D)),
                dec=[block_init(k[9 + i], D) for i in range(2)], out=dense(k[11], D, PDIM))


def encode_tokens(p, patches):
    x = patches @ p["emb"]["w"] + p["emb"]["b"] + p["pos"]
    for b in p["enc"]:
        x = block(b, x)
    return ln(x, p["gE"])


def embed(p, imgs):
    return encode_tokens(p, patchify(imgs)).mean(1)


def mae_loss(p, imgs, key, ratio=0.75):
    pt = patchify(imgs)
    B = pt.shape[0]
    nkeep = int(NTOK * (1 - ratio))
    perm = jax.vmap(lambda k: jax.random.permutation(k, NTOK))(jax.random.split(key, B))
    keep = perm[:, :nkeep]
    x = pt @ p["emb"]["w"] + p["emb"]["b"] + p["pos"]
    xk = jnp.take_along_axis(x, keep[..., None], 1)
    for b in p["enc"]:
        xk = block(b, xk)
    xk = ln(xk, p["gE"]) @ p["dec_in"]["w"] + p["dec_in"]["b"]
    full = jnp.broadcast_to(p["mask_tok"], (B, NTOK, D))
    full = full.at[jnp.arange(B)[:, None], keep].set(xk) + p["dpos"]
    for b in p["dec"]:
        full = block(b, full)
    rec = full @ p["out"]["w"] + p["out"]["b"]
    tgt = (pt - pt.mean(-1, keepdims=True)) / (pt.std(-1, keepdims=True) + 1e-3)
    m = jnp.ones((B, NTOK)).at[jnp.arange(B)[:, None], keep].set(0.0)
    return jnp.sum(((rec - tgt) ** 2).mean(-1) * m) / jnp.sum(m)


def pretrain(imgs, steps=3000, bs=128, seed=0, log=None):
    key = jax.random.PRNGKey(seed)
    p = init(key)
    sched = optax.warmup_cosine_decay_schedule(0, 1.5e-3, 200, steps, 1e-5)
    opt = optax.adamw(sched, weight_decay=0.05)
    st = opt.init(p)

    @jax.jit
    def step(p, st, x, k):
        l, g = jax.value_and_grad(mae_loss)(p, x, k)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    t0 = time.time(); hist = []
    for s in range(steps):
        idx = rng.integers(0, len(imgs), bs)
        x = imgs[idx].astype(np.float32) / 255.0
        if rng.random() < 0.5:
            x = x[:, :, ::-1]
        key, k = jax.random.split(key)
        p, st, l = step(p, st, jnp.asarray(x), k)
        if s % 100 == 0:
            hist.append((s, float(l)))
            print(s, float(l), "%.0fs" % (time.time() - t0), flush=True)
    return p, hist


def embed_all(p, imgs, bs=512):
    f = jax.jit(lambda x: embed(p, x))
    out = []
    for i in range(0, len(imgs), bs):
        out.append(np.asarray(f(jnp.asarray(imgs[i:i + bs].astype(np.float32) / 255.0))))
    return np.concatenate(out)


# ------------------------------------------------------------ severity head
def to_z(s):
    return np.log(s + 0.005)


def fit_head(E, s, seed=0, steps=3000):
    """2-layer MLP on frozen embeddings; Gaussian NLL on log-severity."""
    mu, sd = E.mean(0), E.std(0) + 1e-6
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    p = dict(l1=dense(k[0], E.shape[1], 64), l2=dense(k[1], 64, 64), o=dense(k[2], 64, 2))

    def f(p, x):
        h = jax.nn.gelu(x @ p["l1"]["w"] + p["l1"]["b"])
        h = jax.nn.gelu(h @ p["l2"]["w"] + p["l2"]["b"])
        o = h @ p["o"]["w"] + p["o"]["b"]
        return o[:, 0], jnp.clip(o[:, 1], -4, 2)

    def loss(p, x, y):
        m, ls = f(p, x)
        return jnp.mean(0.5 * ((y - m) / jnp.exp(ls)) ** 2 + ls)

    opt = optax.adam(1e-3); st = opt.init(p)
    X = jnp.asarray((E - mu) / sd); Y = jnp.asarray(to_z(s))

    @jax.jit
    def step(p, st, idx):
        l, g = jax.value_and_grad(loss)(p, X[idx], Y[idx])
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    for i in range(steps):
        p, st, l = step(p, st, jnp.asarray(rng.integers(0, len(E), 256)))
    pred = jax.jit(lambda x: f(p, x))
    return lambda Enew: tuple(np.asarray(a) for a in pred(jnp.asarray((Enew - mu) / sd)))


def color_features(imgs):
    """Handcrafted colour-index baseline features (fraction of tan / dark / green px)."""
    x = imgs.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    exg = 2 * g - r - b
    tan = ((r > 0.5) & (g > 0.38) & (b < 0.4) & (r - g > 0.08)).mean((1, 2))
    dark = ((r + g + b) < 0.5).mean((1, 2))
    green = (exg > 0.12).mean((1, 2))
    return np.stack([tan, dark, green, exg.mean((1, 2)), r.mean((1, 2)), g.mean((1, 2)),
                     b.mean((1, 2)), tan / (green + 0.02), dark / (green + 0.02)], -1)


if __name__ == "__main__":
    import json
    Z = np.load("bench.npz")
    imgs = Z["img"]; split = Z["split"]
    nF, nD = imgs.shape[:2]
    flat = imgs.reshape(-1, 48, 48, 3)
    fsplit = np.repeat(split, nD)
    pre = np.isin(fsplit, ["train", "cal"])
    p, hist = pretrain(flat[pre])
    E = embed_all(p, flat)
    C = color_features(flat)
    s = Z["S_expert"].reshape(-1)
    tr = fsplit == "train"
    res = {}
    out = {}
    for name, feats in [("mae", E), ("color", C)]:
        head = fit_head(feats[tr], s[tr])
        m, ls = head(feats)
        out[name + "_mu"] = m.reshape(nF, nD); out[name + "_ls"] = ls.reshape(nF, nD)
        strue = Z["S"][:, Z["img_days"]].reshape(-1)
        for sp in ["train", "cal", "test", "warm", "geo"]:
            k = fsplit == sp
            pr = np.exp(m[k]) - 0.005
            res[f"{name}_{sp}_mae"] = float(np.mean(np.abs(pr - strue[k])))
            res[f"{name}_{sp}_r_log"] = float(np.corrcoef(m[k], to_z(strue[k]))[0, 1])
    print(json.dumps(res, indent=1))
    np.savez_compressed("emb.npz", emb=E.reshape(nF, nD, -1).astype(np.float32),
                        color=C.reshape(nF, nD, -1).astype(np.float32), **out)
    json.dump(dict(pretrain_hist=hist, head=res), open("encoder.json", "w"), indent=1)
