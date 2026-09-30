"""Amortised abduction for CausalTwin-FM.

A recurrent inference network reads ONLY pre-decision history (images through
the frozen foundation encoder, IoT, weather, management up to the origin) and
returns a Gaussian posterior over the exogenous twin state at the origin:
latent-tissue fractions (L1, L2), infectious and necrotic tissue (I, R), and
the field latents (theta, iota).  Future management never enters the encoder;
it acts only through the twin's typed ports during propagation, which is what
keeps counterfactual queries interventional.  Twin dynamics are frozen (stage 1
parameters); only the encoder is trained (stage 2) with a mixture likelihood
of expert-scored severities over the following 21 days.
"""
import numpy as np
import jax
import jax.numpy as jnp
import optax
from common import EPS
import seqdata as SD
import baselines as BL
import twin as TW
import run_twin as RT

NLAT = 6


def enc_init(key, fp):
    k = jax.random.split(key, 3)
    return dict(enc=BL.gru_init(k[0], fp, BL.HID), h1=BL.dense(k[1], BL.HID, 64),
                out=BL.dense(k[2], 64, 2 * NLAT, s=0.1))


def enc_apply(p, past, mask):
    B = past.shape[0]

    def f(h, xm):
        x, m = xm
        hn = BL.gru_cell(p["enc"], h, x)
        return jnp.where(m[:, None] > 0, hn, h), None

    h, _ = jax.lax.scan(f, jnp.zeros((B, BL.HID)), (past.transpose(1, 0, 2), mask.T))
    o = BL.lin(p["out"], jax.nn.gelu(BL.lin(p["h1"], h)))
    base = jnp.array([-7.0, -7.0, -6.0, -6.0, 0.0, 0.0])
    return o[:, :NLAT] + base, jnp.clip(o[:, NLAT:] - 1.0, -4.0, 1.0)


def sample_state(mu, ls, psi, key, K):
    eps = jax.random.normal(key, mu.shape[:-1] + (K, NLAT))
    zz = mu[..., None, :] + jnp.exp(ls)[..., None, :] * eps
    L1, L2, I, R = [jnp.exp(jnp.clip(zz[..., i], -30.0, 0.0)) for i in range(4)]
    tot = L1 + L2 + I + R
    sc = jnp.where(tot > 0.95, 0.95 / tot, 1.0)
    L1, L2, I, R = L1 * sc, L2 * sc, I * sc, R * sc
    H = 1.0 - (L1 + L2 + I + R)
    s = jnp.stack([H, L1, L2, I, R, jnp.broadcast_to(psi[..., None], H.shape)], -1)
    return s, jnp.clip(zz[..., 4], -5.0, 5.0), jnp.clip(zz[..., 5], -8.0, 8.0)


def origin_arrays(bench, cfg, p, idx, sdays, mode="forecast", action=None):
    past, mask, _, _ = SD.windows(bench, idx, sdays, mode=mode, action=action, feat=cfg["feat"])
    if not cfg["images"]:
        past = past.copy(); past[..., 11:22] = 0.0       # image flag, head mean, embedding PCA, recency
    if not cfg["iot"]:
        past = past.copy(); past[..., 5:9] = 0.0
    H = 21
    fut = RT.build_future(bench, cfg, idx, sdays, H, mode=mode, action=action)
    off = TW.iot_offset(bench, cfg, p, idx, sdays) if cfg["iot"] else np.zeros((len(idx), len(sdays)))
    drv = TW.drivers(bench, cfg, p, idx, sdays, fut, H, off)
    psi = TW.psi_at(bench, p, idx, sdays)
    B, O = len(idx), len(sdays)
    drv = {k: v.reshape((B * O,) + v.shape[2:]) for k, v in drv.items()}
    return past, mask, drv, psi.reshape(-1).astype(np.float32)


def train_encoder(bench, cfg, p, idx, seed=0, steps=2500, K=16, bs=128, sxi=0.2, joint=False):
    sd = bench.origins
    past, mask, drv, psi = origin_arrays(bench, cfg, p, idx, sd)
    # expert labels at origin and following image days (+3..+21)
    lab = np.log(bench.S_expert[idx] + EPS)
    O = len(sd)
    Y = np.stack([lab[:, bench.oidx + k] for k in range(8)], -1).reshape(len(idx) * O, 8).astype(np.float32)
    key = jax.random.PRNGKey(seed + 100)
    q = dict(e=enc_init(key, past.shape[-1]), logs=jnp.log(0.3))
    if joint:   # ablation: dynamics/ports re-estimated jointly with the encoder
        q["p"] = jax.tree_util.tree_map(jnp.asarray, p)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)); st = opt.init(q)
    Pj, Mj, PSj, Yj = map(jnp.asarray, (past, mask, psi, Y))
    Dj = {k: jnp.asarray(v) for k, v in drv.items()}
    pj = jax.tree_util.tree_map(jnp.asarray, p)
    steps_idx = jnp.array([2, 5, 8, 11, 14, 17, 20])   # rollout index -> day tau+3..tau+21

    def loss(q, i, key):
        mu, ls = enc_apply(q["e"], Pj[i], Mj[i])
        k1, k2 = jax.random.split(key)
        s0, th, io = sample_state(mu, ls, PSj[i], k1, K)
        d = {k: v[i][:, None] if k != "resist" else v[i][:, None] for k, v in Dj.items()}
        d = {k: (v if k == "resist" else v) for k, v in d.items()}
        drv_i = {k: Dj[k][i] for k in Dj}
        drv_i = {k: (v[:, None] if k == "resist" else v[:, None, :]) for k, v in drv_i.items()}
        S = propagate_batch(q["p"] if joint else pj, cfg, s0, th, io, drv_i, sxi, k2)
        vis = jnp.concatenate([TW.visible(s0)[None], TW.visible(S[steps_idx])], 0)   # (8,B,K)
        pred = jnp.log(vis + EPS).transpose(1, 0, 2)                     # (B,8,K)
        so = jnp.exp(q["logs"])
        ll = -0.5 * ((Yj[i][..., None] - pred) / so) ** 2 - q["logs"]
        mix = jax.scipy.special.logsumexp(ll, -1) - jnp.log(K)
        return -jnp.mean(mix)

    @jax.jit
    def upd(q, st, i, key):
        l, g = jax.value_and_grad(loss)(q, i, key)
        u, st = opt.update(g, st, q)
        return optax.apply_updates(q, u), st, l

    rng = np.random.default_rng(seed)
    hist = []
    for s in range(steps):
        key, k = jax.random.split(key)
        q, st, l = upd(q, st, jnp.asarray(rng.integers(0, len(past), bs)), k)
        if s % 250 == 0:
            hist.append((s, float(l)))
    if joint:
        p_new = jax.tree_util.tree_map(np.asarray, q.pop("p"))
        return q, hist, p_new
    return q, hist


def propagate_batch(p, cfg, s0, th, io, drv, sxi, key):
    """s0: (B,K,6); drv arrays (B,1,H) / resist (B,1)."""
    H = drv["T"].shape[-1]

    def body(c, h):
        s, key = c
        key, k = jax.random.split(key)
        x = dict(T=drv["T"][..., h], W=drv["W"][..., h], spray=drv["spray"][..., h], r=drv["r"][..., h],
                 dd10=drv["dd10"][..., h], t=drv["t"][..., h], resist=drv["resist"])
        s = TW.step(p, cfg, s, x, th, io, sxi * jax.random.normal(k, th.shape))
        return (s, key), s

    _, S = jax.lax.scan(body, (s0, key), jnp.arange(H))
    return S


def predict(bench, cfg, p, q, idx, sdays, mode="forecast", action=None, K=64, sxi=0.2, seed=0,
            steps=None, batch=2048):
    """Returns vis samples (B,O,len(steps),K)."""
    past, mask, drv, psi = origin_arrays(bench, cfg, p, idx, sdays, mode, action)
    pj = jax.tree_util.tree_map(jnp.asarray, p)
    f = jax.jit(lambda a, b_: enc_apply(q["e"], a, b_))
    outs = []
    key = jax.random.PRNGKey(seed + 5)
    for i in range(0, len(past), batch):
        sl = slice(i, i + batch)
        mu, ls = f(jnp.asarray(past[sl]), jnp.asarray(mask[sl]))
        s0, th, io = sample_state(mu, ls, jnp.asarray(psi[sl]), key, K)
        d = {k: jnp.asarray(v[sl])[:, None] if k == "resist" else jnp.asarray(v[sl])[:, None, :]
             for k, v in drv.items()}
        S = propagate_batch(pj, cfg, s0, th, io, d, sxi, jax.random.fold_in(key, i))
        vis = np.asarray(TW.visible(S))[np.asarray(steps)]        # (len,B,K)
        outs.append(vis.transpose(1, 0, 2))
    v = np.concatenate(outs)
    return v.reshape(len(idx), len(sdays), len(steps), K)


def forecast(bench, cfg, p, q, idx, sig_h, sxi=0.2, seed=0):
    vis = predict(bench, cfg, p, q, idx, bench.origins, steps=[6, 13, 20], sxi=sxi, seed=seed)
    w = np.ones(vis.shape[:2] + (vis.shape[-1],))
    m, s = RT.summarize(vis, w, sig_h)
    raw = np.sqrt(np.maximum(s ** 2 - np.asarray(sig_h)[None, None] ** 2, 0))
    return m, s, raw


def counterfactual(bench, cfg, p, q, idx, sxi=0.2, seed=0):
    import pathosim as P
    res = []
    for a in P.ACTIONS:
        vis = predict(bench, cfg, p, q, idx, RT.CF_SAVE, mode="actual", action=a, steps=[20], sxi=sxi,
                      seed=seed + 11)
        res.append(vis[:, :, 0].mean(-1))
    return np.stack(res, 2)
