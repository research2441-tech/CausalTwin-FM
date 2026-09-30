"""Forecast and counterfactual inference with the twin."""
import numpy as np
import pathosim as P
from common import HORIZONS, EPS, z
import twin as TW

CF_SAVE = [d - 1 for d in P.CF_DAYS]


def build_future(bench, cfg, idx, sdays, H, mode="forecast", action=None):
    inp = TW.Inputs(bench, cfg)
    d = inp.past(idx)
    F = inp.feat(d, cfg["ports"])
    sdays = np.asarray(sdays)
    B, O = len(idx), len(sdays)
    past_feat = np.stack([F[:, s - TW.KLAG + 1:s + 1] for s in sdays], 1)
    past_dd = np.stack([d["dd10"][:, s - TW.KLAG + 1:s + 1] for s in sdays], 1)
    qall = inp.qimg(idx)
    qimg = qall[:, sdays]
    fut = dict(past_feat=past_feat, past_dd=past_dd, qimg=qimg)
    if mode == "forecast":
        oi = [list(bench.origins).index(int(s)) for s in sdays]
        wf = bench.wxf[idx][:, oi, :H]
        fut.update(Tmin=wf[..., 0], Tmax=wf[..., 1], P=wf[..., 2], Ww=wf[..., 4])
    else:
        wx = bench.wx[idx]
        g = lambda c: np.stack([wx[:, s + 1:s + 1 + H, c] for s in sdays], 1)
        fut.update(Tmin=g(0), Tmax=g(1), P=g(2), Ww=g(4))
    spray = np.stack([bench.spray[idx, s + 1:s + 1 + H] for s in sdays], 1)
    sprk = np.stack([bench.static[idx, 4][:, None] * bench.irr[idx, s + 1:s + 1 + H] for s in sdays], 1)
    if action is not None:
        spray = np.zeros_like(spray)
        if action in ("spray_now", "spray_twice"):
            spray[..., 0] = 1
        if action == "spray_twice":
            spray[..., 10] = 1
        if action == "drip_only":
            sprk = np.zeros_like(sprk)
    fut.update(spray=spray.astype(np.float32), sprk=sprk.astype(np.float32))
    return fut


def summarize(vis, w, sig_h):
    """particle mixture -> Gaussian on z scale: mean, sd (adds model-error sd)."""
    zv = np.log(vis + EPS)
    w = w / w.sum(-1, keepdims=True)
    m = np.einsum("bohk,bok->boh", zv, w)
    v = np.einsum("bohk,bok->boh", (zv - m[..., None]) ** 2, w)
    return m, np.sqrt(v + np.asarray(sig_h)[None, None] ** 2)


def forecast(bench, cfg, p, idx, sxi=0.25, sig_h=(0.3, 0.4, 0.5), K=96, seed=0, batch=64, prior=None):
    mus, sds, raw = [], [], []
    for i in range(0, len(idx), batch):
        ii = idx[i:i + batch]
        pf = TW.particle_filter(bench, cfg, p, ii, K=K, sxi=sxi, seed=seed, save_days=bench.origins,
                                prior=None if prior is None else prior[i:i + batch])
        fut = build_future(bench, cfg, ii, bench.origins, max(HORIZONS))
        vis, w = TW.rollout(bench, cfg, p, ii, pf, bench.origins, fut, max(HORIZONS), sxi=sxi, seed=seed + 1)
        vis = vis[:, :, [h - 1 for h in HORIZONS]]
        m, s = summarize(vis, w, sig_h)
        mus.append(m); sds.append(s)
        raw.append(np.sqrt(np.maximum(s ** 2 - np.asarray(sig_h)[None, None] ** 2, 0)))
    return np.concatenate(mus), np.concatenate(sds), np.concatenate(raw)


def counterfactual(bench, cfg, p, idx, sxi=0.25, K=96, seed=0, batch=64, prior=None):
    """Returns predicted S at (d+7, d+14, d+20) for each decision day and action:
    array (B, n_days, n_actions, 3) of particle-mean severity (median of mixture on z)."""
    out = []
    for i in range(0, len(idx), batch):
        ii = idx[i:i + batch]
        pf = TW.particle_filter(bench, cfg, p, ii, K=K, sxi=sxi, seed=seed, save_days=CF_SAVE,
                                prior=None if prior is None else prior[i:i + batch])
        res = []
        for a in P.ACTIONS:
            fut = build_future(bench, cfg, ii, CF_SAVE, P.CF_WIN, mode="actual", action=a)
            vis, w = TW.rollout(bench, cfg, p, ii, pf, CF_SAVE, fut, P.CF_WIN, sxi=sxi, seed=seed + 7)
            vis = vis[:, :, [7, 14, 20]]
            w = w / w.sum(-1, keepdims=True)
            res.append(np.einsum("bohk,bok->boh", vis, w))
        out.append(np.stack(res, 2))
    return np.concatenate(out)
