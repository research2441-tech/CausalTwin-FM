"""Computational cost: parameters, training time, inference latency (single CPU core)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import time, json, pickle
import numpy as np
import jax
from common import Bench
import twin as TW, twin_amort as TA, run_twin as RT
import seqdata as SD, baselines as BL
import encoder_mae as EM

b = Bench(); tr = b.idx("train"); te = b.idx("test"); O = len(b.origins)
nparam = lambda t: int(sum(np.size(x) for x in jax.tree_util.tree_leaves(t)))
res = {}
res["mae_encoder_params"] = nparam(EM.init(jax.random.PRNGKey(0)))
cfg = dict(TW.DEFAULT)
t = time.time(); p, _, lat = TW.train(b, cfg, tr, steps=1500); t1 = time.time() - t
p = jax.tree_util.tree_map(np.asarray, p)
t = time.time(); q, _ = TA.train_encoder(b, cfg, p, tr); t2 = time.time() - t
q = jax.tree_util.tree_map(np.asarray, q)
TA.forecast(b, cfg, p, q, te[:8], (0.3, 0.3, 0.3))            # warm-up / compile
t = time.time(); TA.forecast(b, cfg, p, q, te, (0.3, 0.3, 0.3)); tf = time.time() - t
t = time.time(); TA.counterfactual(b, cfg, p, q, te); tc = time.time() - t
res["twin"] = dict(params_twin=nparam(p), params_encoder=nparam(q), train_s=t1 + t2, stage1_s=t1, stage2_s=t2,
                   ms_per_forecast=1000 * tf / (len(te) * O), ms_per_cf_query=1000 * tc / (len(te) * 3 * 4))
pd = TW.init_params(jax.random.PRNGKey(0), dict(cfg, attn="none"))
cfgp = dict(cfg, attn="none")
RT.forecast(b, cfgp, pd, te[:8])
t = time.time(); RT.forecast(b, cfgp, pd, te); tp = time.time() - t
res["process_da"] = dict(params=17, train_s=0.0, ms_per_forecast=1000 * tp / (len(te) * O))
P_, M_, F_, S_ = SD.windows(b, tr, b.origins)
Y = BL.daily_targets(b, tr, b.origins).reshape(len(P_), -1)
Pt, Mt, Ft, _ = SD.windows(b, te, b.origins)
for kind in ["gru", "crn", "transformer"]:
    steps = 3000
    t = time.time(); pred = BL.fit_seq(kind, P_, M_, F_, Y, S_, steps=steps); tt = time.time() - t
    pred(Pt[:64], Mt[:64], Ft[:64])
    t = time.time(); pred(Pt, Mt, Ft); ti = time.time() - t
    k = jax.random.PRNGKey(0)
    pp = BL.tf_init(k, P_.shape[-1], F_.shape[-1]) if kind == "transformer" else \
        BL.gru_init_all(k, P_.shape[-1], F_.shape[-1], crn=(kind == "crn"))
    res[kind] = dict(params=nparam(pp), train_s=tt, ms_per_forecast=1000 * ti / len(Pt))
    print(kind, res[kind], flush=True)
X = SD.tab_features(b, tr, b.origins).reshape(len(tr) * O, -1)
t = time.time(); gb = BL.fit_gbm(X, SD.targets(b, tr, b.origins).reshape(-1, 3),
                                 np.repeat(b.year[tr] * 10 + (b.region[tr] == "HYD"), O)); tg = time.time() - t
Xt = SD.tab_features(b, te, b.origins).reshape(len(te) * O, -1)
t = time.time(); gb(Xt); ti = time.time() - t
res["gbm"] = dict(params=None, train_s=tg, ms_per_forecast=1000 * ti / len(Xt))
json.dump(res, open("cost.json", "w"), indent=1)
print(json.dumps(res, indent=1))
