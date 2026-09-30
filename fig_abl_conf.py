"""Figs: ablation deltas and conformal calibration."""
import json
import numpy as np
from figstyle import plt, COL_W, HATCH

S = json.load(open("summary.json"))
ABL = [("abl_no_attn", "− attention residual"), ("abl_calendar", "calendar-time attention"),
       ("abl_no_ports", "generic action entry"), ("abl_joint", "joint re-estimation"),
       ("abl_pf", "particle-filter abduction"), ("abl_pf_no_obsop", "PF, no obs. operator"),
       ("abl_no_images", "− image channel"), ("abl_color", "colour indices"), ("abl_no_iot", "− IoT channel")]
ABL = [a for a in ABL if a[0] in S]
fig, ax = plt.subplots(1, 2, figsize=(COL_W, 2.55), sharey=True)
REFM = "twin_s01" if "twin_s01" in S else "twin"
ref_m = S[REFM]["test"]["mae_21"]["mean"]; ref_c = S[REFM]["test"]["cf_eff_rmse"]["mean"]
y = np.arange(len(ABL))[::-1]
for k, (key, ref, lab) in enumerate([("mae_21", ref_m, "(a) Δ MAE@21 (pp)"), ("cf_eff_rmse", ref_c, "(b) Δ effect RMSE (pp)")]):
    a = ax[k]
    v = np.array([S[m]["test"][key]["mean"] - ref for m, _ in ABL])
    e = np.array([S[m]["test"][key]["sd"] for m, _ in ABL])
    a.barh(y, v, xerr=e, height=0.6, color=["0.3" if x > 0 else "0.8" for x in v], edgecolor="0", lw=0.4,
           error_kw=dict(lw=0.6, capsize=1.5))
    a.axvline(0, color="0", lw=0.6)
    a.set_title(lab, loc="left", fontsize=7.3)
    a.tick_params(length=2, labelsize=6.3)
    a.grid(axis="y", visible=False)
ax[0].set_yticks(y); ax[0].set_yticklabels([l for _, l in ABL], fontsize=6.3)
fig.text(0.62, 0.015, "positive = worse than full model (dark), negative = better (light)", ha="center", fontsize=6)
fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=0.6)
fig.savefig("fig_ablation.png")

# conformal
C = S["twin"]["conformal"]
meth = [("split", "Split"), ("local", "Exposure-localised"), ("aci_season", "ACI within season"),
        ("split_aci_region", "Season-chained ACI")]
fig, ax = plt.subplots(1, 2, figsize=(COL_W, 2.45))
x = np.arange(3); w = 0.19
for k, (mk, lab) in enumerate(meth):
    cov = [C[sp][f"{mk}_21"]["cov"] for sp in ["test", "warm", "geo"]]
    wid = [C[sp][f"{mk}_21"]["width_pp"] for sp in ["test", "warm", "geo"]]
    ax[0].bar(x + (k - 1.5) * w, cov, w * 0.92, color="white" if k else "0.85", edgecolor="0", lw=0.5,
              hatch=["", "////", "....", "xxxx"][k], label=lab)
    ax[1].bar(x + (k - 1.5) * w, wid, w * 0.92, color="white" if k else "0.85", edgecolor="0", lw=0.5,
              hatch=["", "////", "....", "xxxx"][k])
ax[0].axhline(0.9, color="0", lw=0.7, ls="--")
ax[0].set_ylim(0.6, 1.0); ax[0].set_ylabel("Coverage at 21 d")
ax[1].set_ylabel("Median width (pp)")
for a, t in zip(ax, ["(a) Coverage (nominal 0.90)", "(b) Interval width"]):
    a.set_xticks(x); a.set_xticklabels(["Temporal", "Warming", "Geographic"], fontsize=6.5)
    a.set_title(t, loc="left", fontsize=7.3); a.tick_params(length=2, labelsize=6.3)
    a.grid(axis="x", visible=False)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, fontsize=5.9, bbox_to_anchor=(0.5, 0.0), handlelength=1.6,
           columnspacing=0.8)
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.8)
fig.savefig("fig_conformal.png")
print("ok")
