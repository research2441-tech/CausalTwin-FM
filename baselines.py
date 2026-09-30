"""Learned baselines: GRU seq2seq, Transformer (TFT-style known-future inputs),
CRN-style balanced GRU (gradient reversal on future treatment), static
single-time-point multimodal MLP, gradient-boosted trees, persistence."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import jax
import jax.numpy as jnp
import optax
from common import HORIZONS, EPS
import seqdata as SD

HID = 64


def dense(key, i, o, s=1.0):
    return dict(w=jax.random.normal(key, (i, o)) * s * np.sqrt(1.0 / i), b=jnp.zeros(o))


def lin(p, x):
    return x @ p["w"] + p["b"]


def gru_init(key, i, h):
    k = jax.random.split(key, 2)
    return dict(x=dense(k[0], i, 3 * h), h=dense(k[1], h, 3 * h))


def gru_cell(p, h, x):
    gx = lin(p["x"], x); gh = lin(p["h"], h)
    zr = jax.nn.sigmoid(gx[..., :2 * HID] + gh[..., :2 * HID])
    zg, rg = zr[..., :HID], zr[..., HID:]
    n = jnp.tanh(gx[..., 2 * HID:] + rg * gh[..., 2 * HID:])
    return (1 - zg) * n + zg * h


# ------------------------------------------------------------ GRU / CRN
def gru_init_all(key, fp, ff, crn=False):
    k = jax.random.split(key, 6)
    p = dict(enc=gru_init(k[0], fp, HID), dec=gru_init(k[1], ff, HID), out=dense(k[2], HID, 2))
    if crn:
        p["adv"] = dense(k[3], HID + ff - 2, 1)
    return p


def gru_apply(p, past, mask, fut, crn_rep=False):
    B = past.shape[0]
    h0 = jnp.zeros((B, HID))

    def enc(h, xm):
        x, m = xm
        hn = gru_cell(p["enc"], h, x)
        return jnp.where(m[:, None] > 0, hn, h), None

    h, _ = jax.lax.scan(enc, h0, (past.transpose(1, 0, 2), mask.T))

    def dec(h, x):
        h = gru_cell(p["dec"], h, x)
        return h, h

    _, hs = jax.lax.scan(dec, h, fut.transpose(1, 0, 2))
    hs = hs.transpose(1, 0, 2)
    o = lin(p["out"], hs)
    return o[..., 0], jnp.clip(o[..., 1], -3, 1.5), h


@jax.custom_vjp
def grad_rev(x):
    return x


grad_rev.defvjp(lambda x: (x, None), lambda _, g: (-g,))


# ------------------------------------------------------------ Transformer
def tf_init(key, fp, ff, d=64, L=2):
    k = jax.random.split(key, 4 + 4 * L)
    p = dict(ip=dense(k[0], fp, d), if_=dense(k[1], ff, d), pos=0.02 * jax.random.normal(k[2], (SD.WIN + SD.HMAX, d)),
             out=dense(k[3], d, 2), L=[])
    for l in range(L):
        kk = k[4 + 4 * l:8 + 4 * l]
        p["L"].append(dict(qkv=dense(kk[0], d, 3 * d), o=dense(kk[1], d, d), f1=dense(kk[2], d, 2 * d),
                           f2=dense(kk[3], 2 * d, d)))
    return p


def _ln(x):
    return (x - x.mean(-1, keepdims=True)) / jnp.sqrt(x.var(-1, keepdims=True) + 1e-5)


def tf_apply(p, past, mask, fut, heads=4):
    x = jnp.concatenate([lin(p["ip"], past), lin(p["if_"], fut)], 1) + p["pos"]
    m = jnp.concatenate([mask, jnp.ones(fut.shape[:2])], 1)
    B, N, d = x.shape
    for l in p["L"]:
        h = _ln(x)
        q, k, v = jnp.split(lin(l["qkv"], h).reshape(B, N, 3, heads, d // heads), 3, 2)
        q, k, v = q[:, :, 0], k[:, :, 0], v[:, :, 0]
        s = jnp.einsum("bnhd,bmhd->bhnm", q, k) / np.sqrt(d // heads)
        s = jnp.where(m[:, None, None, :] > 0, s, -1e9)
        a = jax.nn.softmax(s, -1)
        x = x + lin(l["o"], jnp.einsum("bhnm,bmhd->bnhd", a, v).reshape(B, N, d))
        x = x + lin(l["f2"], jax.nn.gelu(lin(l["f1"], _ln(x))))
    o = lin(p["out"], _ln(x[:, SD.WIN:]))
    return o[..., 0], jnp.clip(o[..., 1], -3, 1.5), None


# ------------------------------------------------------------ training
def fit_seq(kind, past, mask, fut, y, spray, seed=0, steps=3000, bs=256, lr=2e-3, lam=0.5):
    key = jax.random.PRNGKey(seed)
    fp, ff = past.shape[-1], fut.shape[-1]
    if kind == "transformer":
        p = tf_init(key, fp, ff); ap = tf_apply
    else:
        p = gru_init_all(key, fp, ff, crn=(kind == "crn")); ap = gru_apply
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr)); st = opt.init(p)
    P_, M_, F_, Y_, S_ = map(jnp.asarray, (past, mask, fut, y, spray))

    def loss(p, i, sign):
        mu, ls, h = ap(p, P_[i], M_[i], F_[i])
        l = jnp.mean(0.5 * ((Y_[i] - mu) / jnp.exp(ls)) ** 2 + ls)
        if kind == "crn":
            # adversary predicts future treatment from the history representation
            rep = grad_rev(h) if sign else jax.lax.stop_gradient(h)
            fcov = F_[i][:, 0, :-2 - 5 - 2]  # weather part of first future step (no treatment)
            logit = lin(p["adv"], jnp.concatenate([rep, F_[i][:, 0, :ff - 2]], -1))[:, 0]
            tgt = (S_[i].sum(1) > 0).astype(jnp.float32)
            bce = jnp.mean(optax.sigmoid_binary_cross_entropy(logit, tgt))
            l = l + lam * bce
        return l

    @jax.jit
    def upd(p, st, i):
        l, g = jax.value_and_grad(loss)(p, i, True)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(seed)
    for s in range(steps):
        p, st, l = upd(p, st, jnp.asarray(rng.integers(0, len(past), bs)))
    f = jax.jit(lambda a, b_, c: ap(p, a, b_, c)[:2])

    def predict(past, mask, fut, bsz=1024):
        mus, lss = [], []
        for i in range(0, len(past), bsz):
            m, l = f(jnp.asarray(past[i:i + bsz]), jnp.asarray(mask[i:i + bsz]), jnp.asarray(fut[i:i + bsz]))
            mus.append(np.asarray(m)); lss.append(np.asarray(l))
        return np.concatenate(mus), np.exp(np.concatenate(lss))
    return predict


def daily_targets(b, idx, sdays):
    return np.stack([np.stack([np.log(b.S[idx, s + 1:s + 1 + SD.HMAX] + EPS)], 0)[0] for s in sdays], 1)


# ------------------------------------------------------------ static MLP
def fit_mlp(X, y, seed=0, steps=3000):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    p = dict(a=dense(k[0], X.shape[1], 64), b=dense(k[1], 64, 64), o=dense(k[2], 64, 2 * y.shape[1]))
    nh = y.shape[1]

    def f(p, x):
        h = jax.nn.gelu(lin(p["a"], x)); h = jax.nn.gelu(lin(p["b"], h)); o = lin(p["o"], h)
        return o[:, :nh], jnp.clip(o[:, nh:], -3, 1.5)

    def loss(p, x, t):
        m, ls = f(p, x)
        return jnp.mean(0.5 * ((t - m) / jnp.exp(ls)) ** 2 + ls)

    opt = optax.adam(1e-3); st = opt.init(p)
    Xj = jnp.asarray((X - mu) / sd); Yj = jnp.asarray(y)

    @jax.jit
    def upd(p, st, i):
        l, g = jax.value_and_grad(loss)(p, Xj[i], Yj[i])
        u, st = opt.update(g, st)
        return optax.apply_updates(p, u), st

    rng = np.random.default_rng(seed)
    for s in range(steps):
        p, st = upd(p, st, jnp.asarray(rng.integers(0, len(X), 256)))
    g = jax.jit(lambda x: f(p, x))
    return lambda Xn: tuple(np.asarray(a) if i == 0 else np.exp(np.asarray(a))
                            for i, a in enumerate(g(jnp.asarray((Xn - mu) / sd))))


# ------------------------------------------------------------ GBM
def fit_gbm(X, y, groups, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    models, res_models = [], []
    for h in range(y.shape[1]):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          random_state=seed).fit(X, y[:, h])
        # out-of-fold residual magnitudes -> scale model
        oof = np.zeros(len(X))
        ug = np.unique(groups); folds = np.array_split(np.random.default_rng(seed).permutation(ug), 3)
        for fo in folds:
            te = np.isin(groups, fo)
            mm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                               random_state=seed).fit(X[~te], y[~te, h])
            oof[te] = np.abs(y[te, h] - mm.predict(X[te]))
        r = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed).fit(X, oof)
        models.append(m); res_models.append(r)

    def predict(Xn):
        mu = np.stack([m.predict(Xn) for m in models], -1)
        sd = np.stack([np.maximum(r.predict(Xn), 0.05) for r in res_models], -1) * 1.2533
        return mu, sd
    return predict
