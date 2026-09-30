"""Train/evaluate learned and simple baselines.  usage: python exp_base.py SEED [names]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, time
import numpy as np
from common import Bench, HORIZONS, EPS
import seqdata as SD
import baselines as BL
import pathosim as P

SPLITS = ["cal", "test", "warm", "geo"]
CF_S = [d - 1 for d in P.CF_DAYS]
HI = [h - 1 for h in HORIZONS]


def static_X(b, idx, sdays, mode="forecast", action=None):
    D = SD.daily_past(b, idx)
    days = b.img_days
    T = SD.tab_features(b, idx, sdays, mode, action)
    rows = []
    for o, s in enumerate(sdays):
        j = np.searchsorted(days, s, side="right") - 1
        rows.append(np.concatenate([b.emb[idx, j], D[:, s, :9], b.static[idx], T[:, o, 9:13]], -1))
    return np.stack(rows, 1)


def main(seed, names):
    b = Bench()
    tr = b.idx("train")
    O = len(b.origins)
    for name in names:
        t0 = time.time(); out = {}
        if name in ("gru", "transformer", "crn"):
            P_, M_, F_, S_ = SD.windows(b, tr, b.origins)
            Y = BL.daily_targets(b, tr, b.origins).reshape(len(P_), -1)
            pred = BL.fit_seq(name, P_, M_, F_, Y, S_, seed=seed, steps=3000)
            for sp in SPLITS:
                idx = b.idx(sp)
                Pt, Mt, Ft, _ = SD.windows(b, idx, b.origins)
                mu, sd = pred(Pt, Mt, Ft)
                out[f"{sp}_mu"] = mu[:, HI].reshape(len(idx), O, 3)
                out[f"{sp}_sd"] = sd[:, HI].reshape(len(idx), O, 3)
                if sp != "cal":
                    cf = []
                    for a in P.ACTIONS:
                        Pt, Mt, Ft, _ = SD.windows(b, idx, CF_S, mode="actual", action=a)
                        m, _ = pred(Pt, Mt, Ft)
                        cf.append(np.exp(m[:, 20].reshape(len(idx), len(CF_S))) - EPS)
                    out[f"{sp}_cf"] = np.stack(cf, 2)
        elif name in ("gbm", "static_mlp"):
            if name == "gbm":
                X = SD.tab_features(b, tr, b.origins).reshape(len(tr) * O, -1)
            else:
                X = static_X(b, tr, b.origins).reshape(len(tr) * O, -1)
            y = SD.targets(b, tr, b.origins).reshape(-1, 3)
            groups = np.repeat(b.year[tr] * 10 + (b.region[tr] == "HYD"), O)
            model = BL.fit_gbm(X, y, groups, seed=seed) if name == "gbm" else BL.fit_mlp(X, y, seed=seed)
            fx = SD.tab_features if name == "gbm" else static_X
            for sp in SPLITS:
                idx = b.idx(sp)
                mu, sd = model(fx(b, idx, b.origins).reshape(len(idx) * O, -1))
                out[f"{sp}_mu"] = mu.reshape(len(idx), O, 3); out[f"{sp}_sd"] = sd.reshape(len(idx), O, 3)
                if sp != "cal":
                    cf = []
                    for a in P.ACTIONS:
                        m, _ = model(fx(b, idx, CF_S, "actual", a).reshape(len(idx) * len(CF_S), -1))
                        cf.append(np.exp(m[:, 2].reshape(len(idx), len(CF_S))) - EPS)
                    out[f"{sp}_cf"] = np.stack(cf, 2)
        elif name == "persistence":
            ytr = SD.targets(b, tr, b.origins)
            m_tr = b.hmu[tr][:, b.oidx]
            sd_h = np.sqrt(np.mean((ytr - m_tr[..., None]) ** 2, (0, 1)))
            for sp in SPLITS:
                idx = b.idx(sp)
                m = b.hmu[idx][:, b.oidx]
                out[f"{sp}_mu"] = np.repeat(m[..., None], 3, -1)
                out[f"{sp}_sd"] = np.broadcast_to(sd_h, out[f"{sp}_mu"].shape).copy()
                if sp != "cal":
                    j = [np.searchsorted(b.img_days, s, side="right") - 1 for s in CF_S]
                    base = np.exp(b.hmu[idx][:, j]) - EPS
                    out[f"{sp}_cf"] = np.repeat(base[..., None], len(P.ACTIONS), -1)
        np.savez_compressed(f"pred_{name}_s{seed}.npz", **out)
        print(name, seed, "done %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    seed = int(sys.argv[1])
    main(seed, sys.argv[2:] or ["persistence", "gbm", "static_mlp", "gru", "transformer", "crn"])
