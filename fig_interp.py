"""Fig: what the twin identified - efficiency surface, thermal-time attention, splash residual."""
import pickle, json
import numpy as np
import jax.numpy as jnp
from figstyle import plt, COL_W, GRAYS
from common import Bench
import twin as TW
from pathosim import beta_T

b = Bench()
te = b.idx("test")
fig = plt.figure(figsize=(COL_W, 4.1))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.55, wspace=0.45)
ax_a = fig.add_subplot(gs[0, :]); ax_b = fig.add_subplot(gs[1, 0]); ax_c = fig.add_subplot(gs[1, 1])
# (a) efficiency surface
T = np.linspace(8, 38, 121); W = np.linspace(0, 24, 97)
TT, WW = np.meshgrid(T, W)
fT = beta_T(TT, 12, 27, 35); true = fT / (1 + np.exp(-(WW - np.minimum(6 / np.maximum(fT, 1e-3), 48)) / 1.5))
ests = []
stats = {}
for s in range(3):
    try:
        d = pickle.load(open(f"params_twin_s{s}.pkl", "rb"))
    except FileNotFoundError:
        continue
    p = d["params"]
    ests.append(np.asarray(TW.efficiency({k: jnp.asarray(v) for k, v in p.items()}, jnp.asarray(TT), jnp.asarray(WW))))
est = np.mean(ests, 0)
tn = true / true.max(); en = est / est.max()
lv = [0.1, 0.3, 0.5, 0.7, 0.9]
c1 = ax_a.contour(TT, WW, tn, levels=lv, colors="0.55", linestyles="--", linewidths=0.8)
c2 = ax_a.contour(TT, WW, en, levels=lv, colors="0", linestyles="-", linewidths=0.9)
ax_a.set_xlim(10, 38); ax_a.set_xlabel("Temperature (°C)"); ax_a.set_ylabel("Wetness (h)")
ax_a.set_title("(a) Normalised infection efficiency", loc="left")
from matplotlib.lines import Line2D
la = [Line2D([], [], color="0", lw=0.9), Line2D([], [], color="0.55", ls="--", lw=0.8)]
ll = ["(a) identified by twin", "(a) generator, unknown to twin"]
# (b) attention mass vs thermal lag, (c) residual after heavy rain
d = pickle.load(open("params_twin_s0.pkl", "rb"))
p = {k: jnp.asarray(v) for k, v in d["params"].items()}
cfg = d["cfg"]
inp = TW.Inputs(b, cfg)
dd = inp.past(te)
feat = jnp.asarray(inp.feat(dd, True)); qim = jnp.asarray(inp.qimg(te))
A, lag = TW.attention_weights(p, cfg, feat, qim, jnp.asarray(dd["dd10"], jnp.float32))
A = np.asarray(A)[:, 15:]; lag = 100 * np.asarray(lag)[:, 15:]
bins = np.arange(0, 181, 20)
rain = np.asarray(b.wx[te, :, 2] > 5)
idx = np.arange(b.T)[:, None] - np.arange(1, 11)[None]
rk = np.concatenate([np.zeros((len(te), 10), bool), rain], 1)[:, idx + 10][:, 15:]
for flag, lab, ls, mk in [(True, "Key day with rain > 5 mm", "-", "o"), (False, "Key day without rain", "--", "s")]:
    m_ = rk == flag
    mids, vals = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = m_ & (lag >= lo) & (lag < hi)
        if sel.sum() > 200:
            mids.append((lo + hi) / 2); vals.append(A[sel].mean())
    ax_b.plot(mids, vals, ls=ls, marker=mk, ms=2.8, color="0" if flag else "0.5", label=lab)
ax_b.axhline(0.1, color="0.7", lw=0.5, ls=":")
ax_b.set_xlabel("Thermal lag (degree-days)"); ax_b.set_ylabel("Attention weight")
ax_b.set_title("(b) Attention by lag", loc="left")
r = np.asarray(TW.residual(p, cfg, feat, qim, jnp.asarray(dd["dd10"], jnp.float32)))
P = b.wx[te, :, 2]
after = np.zeros_like(P, bool)
for k in range(1, 4):
    after[:, k:] |= P[:, :-k] > 5
vals = [r[:, 20:][after[:, 20:]].mean(), r[:, 20:][~after[:, 20:]].mean()]
dv = vals[0] - vals[1]
ax_c.bar([0, 1], [dv, np.log(1.8)], width=0.55, color=["0.25", "0.85"], edgecolor="0", lw=0.5,
         hatch=["", "////"])
ax_c.set_xticks([0, 1]); ax_c.set_xticklabels(["twin\n$\\Delta r$", "generator\nln 1.8"])
ax_c.set_xlim(-0.6, 1.6)
ax_c.axhline(0, color="0", lw=0.5)
ax_c.set_ylabel("Log-rate uplift after rain"); ax_c.set_title("(c) Splash effect", loc="left")
h, l = ax_b.get_legend_handles_labels()
l = ["(b) " + x.lower() for x in l]
fig.legend(la + h, ll + l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=6.1,
           handlelength=2, columnspacing=0.9)
fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.215)
fig.savefig("fig_interp.png")
json.dump(dict(res_after_rain=float(vals[0]), res_other=float(vals[1]), delta=float(dv)), open("interp.json", "w"))
print(vals)
