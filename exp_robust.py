"""Robustness: IoT missingness and image revisit interval at test time
(twin seed 0 vs GRU and GBM retrained with seed 0)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, pickle, copy
import numpy as np
from common import Bench, HORIZONS, EPS, unz
import twin_amort as TA
import seqdata as SD
import baselines as BL
from evaluate import ztrue, cf_metrics
from scipy import stats
import pathosim as P

b0 = Bench()
tr = b0.idx("train"); te = b0.idx("test")
O = len(b0.origins)
d = pickle.load(open("params_twin_s0.pkl", "rb"))
R0 = np.load("pred_twin_s0.npz")
import twin as TW, run_twin as RT
dpf = pickle.load(open("params_abl_pf_s0.pkl", "rb"))
Rpf = np.load("pred_abl_pf_s0.npz")
sig = R0["sig"]; sxi = float(R0["sxi"])
# baselines
P_, M_, F_, S_ = SD.windows(b0, tr, b0.origins)
Y = BL.daily_targets(b0, tr, b0.origins).reshape(len(P_), -1)
gru = BL.fit_seq("gru", P_, M_, F_, Y, S_, seed=0, steps=3000)
X = SD.tab_features(b0, tr, b0.origins).reshape(len(tr) * O, -1)
gb = BL.fit_gbm(X, SD.targets(b0, tr, b0.origins).reshape(-1, 3),
                np.repeat(b0.year[tr] * 10 + (b0.region[tr] == "HYD"), O), seed=0)


def crps(zt, mu, sd):
    z = (zt - mu) / sd
    return float(np.mean(sd * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))))


def evaluate(b, tag):
    zt = ztrue(b, te); st = unz(zt); out = {}
    mu, sd, _ = TA.forecast(b, d["cfg"], d["params"], d["q"], te, sig, sxi=sxi, seed=0)
    cf = TA.counterfactual(b, d["cfg"], d["params"], d["q"], te, sxi=sxi, seed=0)
    out["twin"] = dict(mae=100 * float(np.mean(np.abs(unz(mu) - st))), crps=crps(zt, mu, sd),
                       cf=cf_metrics(b, "test", {"test_cf": cf})["eff_rmse"])
    pri = TW.region_prior(b, tr, dpf["lat"], te)
    mu, sd, _ = RT.forecast(b, dpf["cfg"], dpf["params"], te, sxi=float(Rpf["sxi"]), sig_h=Rpf["sig"], seed=0, prior=pri)
    cfp = RT.counterfactual(b, dpf["cfg"], dpf["params"], te, sxi=float(Rpf["sxi"]), seed=0, prior=pri)[..., 2]
    out["twin_pf"] = dict(mae=100 * float(np.mean(np.abs(unz(mu) - st))), crps=crps(zt, mu, sd),
                          cf=cf_metrics(b, "test", {"test_cf": cfp})["eff_rmse"])
    Pt, Mt, Ft, _ = SD.windows(b, te, b.origins)
    m, s = gru(Pt, Mt, Ft)
    m = m[:, [6, 13, 20]].reshape(len(te), O, 3); s = s[:, [6, 13, 20]].reshape(len(te), O, 3)
    cfg_ = []
    for a in P.ACTIONS:
        Pt, Mt, Ft, _ = SD.windows(b, te, [dd - 1 for dd in P.CF_DAYS], mode="actual", action=a)
        mm, _ = gru(Pt, Mt, Ft)
        cfg_.append(np.exp(mm[:, 20].reshape(len(te), 3)) - EPS)
    out["gru"] = dict(mae=100 * float(np.mean(np.abs(unz(m) - st))), crps=crps(zt, m, s),
                      cf=cf_metrics(b, "test", {"test_cf": np.stack(cfg_, 2)})["eff_rmse"])
    m, s = gb(SD.tab_features(b, te, b.origins).reshape(len(te) * O, -1))
    m = m.reshape(len(te), O, 3); s = s.reshape(len(te), O, 3)
    out["gbm"] = dict(mae=100 * float(np.mean(np.abs(unz(m) - st))), crps=crps(zt, m, s))
    print(tag, json.dumps(out), flush=True)
    return out


res = {}
rng = np.random.default_rng(123)
for p in [0.08, 0.3, 0.6, 0.9]:
    b = copy.copy(b0); b.miss = b0.miss.copy()
    b.miss[te] = (rng.random(b0.miss[te].shape) < p).astype(np.float32)
    res[f"miss_{p}"] = evaluate(b, f"miss {p}")
for r in [2, 3]:
    b = copy.copy(b0)
    b.hmu = b0.hmu.copy(); b.emb = b0.emb.copy(); b.hls = b0.hls.copy()
    b.img_keep = np.ones(b0.hmu.shape, bool)
    for j in range(len(b0.img_days)):
        b.img_keep[te, j] = (j % r == 0)
        k = (j // r) * r
        b.hmu[te, j] = b0.hmu[te, k]; b.emb[te, j] = b0.emb[te, k]; b.hls[te, j] = b0.hls[te, k]
    res[f"revisit_{3 * r}"] = evaluate(b, f"revisit {3 * r} d")
json.dump(res, open("robust.json", "w"), indent=1)
