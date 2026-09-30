"""Fig: counterfactual effect recovery (spray-now effect at d+20, temporal test)."""
import json
import numpy as np
from figstyle import plt, COL_W, HATCH
from common import Bench

b = Bench()
te = b.idx("test")
tru = b.cf[te][..., 2]
et = 100 * (tru[..., 1] - tru[..., 0]).ravel()
names = [("twin", "CausalTwin-FM"), ("gru", "GRU seq2seq"), ("crn", "CRN (balanced)"), ("gbm", "Gradient boosting")]
fig, ax = plt.subplots(2, 2, figsize=(COL_W, 3.55))
lim = (-12, 4)
for k, (m, lab) in enumerate(names):
    a = ax.flat[k]
    ps = []
    for s in range(3):
        try:
            R = np.load(f"pred_{m}_s{s}.npz"); ps.append(R["test_cf"])
        except FileNotFoundError:
            pass
    pr = np.mean(ps, 0)
    ep = 100 * (pr[..., 1] - pr[..., 0]).ravel()
    a.scatter(et, ep, s=4, c="0.25", alpha=0.45, lw=0, rasterized=True)
    a.plot(lim, lim, color="0.5", lw=0.7, ls="--")
    a.axhline(0, color="0.75", lw=0.5); a.axvline(0, color="0.75", lw=0.5)
    rmse = np.sqrt(np.mean((ep - et) ** 2)); sgn = np.mean(np.sign(ep[np.abs(et) > 0.2]) == np.sign(et[np.abs(et) > 0.2]))
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_title(f"({'abcd'[k]}) {lab}", loc="left")
    a.text(0.97, 0.05, f"RMSE {rmse:.2f} pp\nsign {100 * sgn:.0f}%", transform=a.transAxes, ha="right",
           va="bottom", fontsize=6.3, bbox=dict(fc="white", ec="0.7", lw=0.4, pad=1.5))
    if k >= 2:
        a.set_xlabel("True effect (pp)")
    if k % 2 == 0:
        a.set_ylabel("Predicted effect (pp)")
    a.tick_params(length=2)
fig.tight_layout(h_pad=0.9, w_pad=0.8)
fig.savefig("fig_cf.png")
print("ok")
