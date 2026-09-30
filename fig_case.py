"""Fig: case study of one temporal-test field - forecast fan and action queries."""
import pickle
import numpy as np
from figstyle import plt, COL_W, LS, MK
from common import Bench, EPS
import twin_amort as TA
import pathosim as P

b = Bench()
te = b.idx("test")
d = pickle.load(open("params_twin_s0.pkl", "rb"))
R = np.load("pred_twin_s0.npz")
sig = R["sig"]; sxi = float(R["sxi"])
# choose a field with a clear epidemic and a sizeable spray effect at day 65
eff = b.cf[te][:, 1, 1, 2] - b.cf[te][:, 1, 0, 2]
pcf = R["test_cf"][:, 1]
err = np.abs((pcf[:, 1] - pcf[:, 0]) - eff)
cand = np.where(np.abs(eff) > 0.01)[0]
n_loc = cand[np.argsort(err[cand])[len(cand) // 2]]
n = te[n_loc]
orig = 45
vis = TA.predict(b, d["cfg"], d["params"], d["q"], np.array([n]), [orig], steps=list(range(21)), K=256,
                 sxi=sxi, seed=3)[0, 0]            # (21, K)
zz = np.log(vis + EPS)
sh = np.interp(np.arange(1, 22), [7, 14, 21], sig)
rng = np.random.default_rng(0)
zs = zz + sh[:, None] * rng.standard_normal(zz.shape)
q05, q50, q95 = [100 * (np.exp(np.quantile(zs, q, 1)) - EPS) for q in (0.05, 0.5, 0.95)]
fig, ax = plt.subplots(2, 1, figsize=(COL_W, 4.6), gridspec_kw=dict(height_ratios=[1.1, 1]))
days = np.arange(b.T)
a = ax[0]
a.plot(days, 100 * b.S[n], color="0", lw=1.0, label="True visible severity")
a.plot(days, 100 * b.Lat[n], color="0.55", lw=0.9, ls=":", label="True latent tissue (unseen)")
a.plot(b.img_days, 100 * (np.exp(b.hmu[n]) - EPS), "o", ms=2.3, mfc="white", mec="0.35", mew=0.6,
       label="Image-derived estimate")
fd = orig + np.arange(1, 22)
a.fill_between(fd, q05, q95, color="0.8", lw=0, label="90% predictive band")
a.plot(fd, q50, color="0.25", ls="--", lw=1.0, label="Twin median forecast")
for t_ in np.where(b.spray[n] > 0)[0]:
    a.axvline(t_, color="0.6", lw=0.5, ls="-.")
a.axvline(orig, color="0", lw=0.6)
a.set_yscale("symlog", linthresh=1); a.set_xlim(0, 139); a.set_ylim(0, None)
a.set_ylabel("Severity (%)"); a.set_xlabel("Days after sowing")
a.set_title(f"(a) Forecast issued on day {orig} (vertical dash-dot: recorded sprays)", loc="left", fontsize=7.6)
a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=5.8, ncol=3, handlelength=1.8,
         columnspacing=0.8)
# (b) action queries at decision day 65
di = 1; dday = P.CF_DAYS[di]
pr = {}
for ai, act in enumerate(P.ACTIONS):
    v = TA.predict(b, d["cfg"], d["params"], d["q"], np.array([n]), [dday - 1], mode="actual", action=act,
                   steps=list(range(21)), K=256, sxi=sxi, seed=11)[0, 0]
    pr[act] = 100 * v.mean(-1)
labels = {"no_spray": "No spray", "spray_now": "Spray now", "spray_twice": "Spray + day 10",
          "drip_only": "Drip, no spray"}
a = ax[1]
fd = dday + np.arange(21)
from matplotlib.lines import Line2D
hs = []
for ai, act in enumerate(P.ACTIONS):
    col = ["0", "0.3", "0.5", "0.65"][ai]
    a.plot(fd, pr[act], ls=LS[ai], color=col, lw=1.0)
    a.plot([dday + 7, dday + 14, dday + 20], 100 * b.cf[n, di, ai], MK[ai], ms=3.4, mfc="white", mec=col, mew=0.8)
    hs.append(Line2D([], [], ls=LS[ai], color=col, lw=1.0, marker=MK[ai], ms=3.4, mfc="white", mec=col,
                     label=labels[act]))
a.set_xlabel("Days after sowing"); a.set_ylabel("Severity (%)")
a.set_title(f"(b) Action queries on day {dday}", loc="left", fontsize=7.6)
a.xaxis.set_major_locator(plt.MultipleLocator(5))
a.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=5.8, ncol=2, handlelength=2.6,
         columnspacing=1.0, title="lines: twin prediction; markers: exact replay", title_fontsize=5.8)
fig.tight_layout(h_pad=0.6)
fig.savefig("fig_case.png")
print("field", n, b.region[n], b.year[n], "effect", eff[n_loc])
