"""Fig. 3: testbed characterisation from the real weather traces and simulated epidemics."""
import numpy as np
from figstyle import plt, GRAYS, LS, COL_W
from common import Bench
from pathosim import beta_T

b = Bench()
fig, ax = plt.subplots(2, 2, figsize=(COL_W, 3.6))
names = {"HYD": "Hyderabad", "COR": "Córdoba", "TUN": "Tunis"}
days = np.arange(b.T)
# (a) seasonal infection-favourability (generic efficiency) by region, hist scen
for k, r in enumerate(["HYD", "COR", "TUN"]):
    m = (b.region == r) & (b.scen == "hist")
    wx = b.wx[m]
    T = 0.5 * (wx[..., 0] + wx[..., 1]); W = wx[..., 4]
    fT = beta_T(T, 10, 25, 35); e = fT / (1 + np.exp(-(W - np.minimum(8 / np.maximum(fT, 1e-3), 48)) / 2))
    e7 = np.array([np.convolve(x, np.ones(7) / 7, mode="same") for x in e])
    ax[0, 0].plot(days, e7.mean(0), color=GRAYS[k], ls=LS[k], label=names[r])
ax[0, 0].set_xlabel("Days after sowing"); ax[0, 0].set_ylabel("7-day mean index")
ax[0, 0].set_title("(a) Infection favourability", loc="left")
# (b) warming delta: paired season means (historical vs SSP5-8.5 delta)
for k, r in enumerate(["HYD", "COR"]):
    xs, ys = [], []
    for y in np.unique(b.year[(b.region == r) & (b.split == "test")]):
        vals = []
        for sc in ["test", "warm"]:
            m = (b.region == r) & (b.year == y) & (b.split == sc)
            wx = b.wx[m][0]; T = 0.5 * (wx[:, 0] + wx[:, 1]); W = wx[:, 4]
            fT = beta_T(T, 10, 25, 35); e = fT / (1 + np.exp(-(W - np.minimum(8 / np.maximum(fT, 1e-3), 48)) / 2))
            vals.append(e.mean())
        xs.append(vals[0]); ys.append(vals[1])
    ax[0, 1].plot(xs, ys, ["o", "s"][k], color=GRAYS[k], ms=4, mfc="white" if k else GRAYS[k], label=names[r])
lim = [0.1, 0.55]
ax[0, 1].plot(lim, lim, color="0.6", lw=0.6, ls=":")
ax[0, 1].set_xlim(lim); ax[0, 1].set_ylim(lim)
ax[0, 1].set_xlabel("Historical season mean"); ax[0, 1].set_ylabel("Warming-delta season mean")
ax[0, 1].set_title("(b) Climate-shift replay", loc="left")
# (c) severity trajectories: median and IQR per region
for k, r in enumerate(["HYD", "COR", "TUN"]):
    m = (b.region == r) & (b.scen == "hist")
    q = np.quantile(100 * b.S[m], [0.25, 0.5, 0.75], 0)
    ax[1, 0].plot(days, q[1], color=GRAYS[k], ls=LS[k], label=names[r])
    ax[1, 0].fill_between(days, q[0], q[2], color=GRAYS[k], alpha=0.15, lw=0)
ax[1, 0].set_yscale("symlog", linthresh=1); ax[1, 0].set_xlabel("Days after sowing")
ax[1, 0].set_ylabel("Visible severity (%)"); ax[1, 0].set_title("(c) Epidemic trajectories", loc="left")
# (d) confounding: sprays vs final severity, factual; and causal effect of spray_now
m = b.scen == "hist"
ns = b.spray[m].sum(1); fin = 100 * b.S[m][:, -1]
lv = np.arange(0, 6)
med = [np.median(fin[ns == v]) if (ns == v).sum() > 5 else np.nan for v in lv]
ax[1, 1].plot(lv, med, "o-", color="0", ms=3.5, label="Observed (factual)")
eff = 100 * (b.cf[m][:, :, 1, 2] - b.cf[m][:, :, 0, 2]).mean(1)
effm = [np.median(eff[ns == v]) if (ns == v).sum() > 5 else np.nan for v in lv]
ax[1, 1].plot(lv, effm, "s--", color="0.5", ms=3.5, label="Causal effect of spray")
ax[1, 1].axhline(0, color="0.3", lw=0.5)
ax[1, 1].set_xlabel("Sprays applied in season"); ax[1, 1].set_ylabel("Severity (%)")
ax[1, 1].set_title("(d) Confounding", loc="left")
for a in ax.flat:
    a.tick_params(length=2)
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)
h2, l2 = ax[1, 1].get_legend_handles_labels()
ax[0, 1].legend(loc="lower right", fontsize=6.5, handletextpad=0.3, borderaxespad=0.2)
fig.legend(h2, l2, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), handlelength=2.2)
fig.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=1.2, w_pad=1.0)
fig.savefig("fig3_testbed.png")
print("saved")
