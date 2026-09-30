"""Fig: factual forecast skill vs horizon across partitions (mean +/- sd over seeds)."""
import json
import numpy as np
from figstyle import plt, COL_W

S = json.load(open("summary.json"))
H = [7, 14, 21]
models = [("twin", "CausalTwin-FM (amortised)", "0", "-", "o"), ("abl_pf", "CausalTwin-FM (filter)", "0", ":", "o"),
          ("gru", "GRU", "0.35", "--", "s"),
          ("crn", "CRN", "0.35", ":", "^"), ("transformer", "Transformer", "0.55", "-.", "D"),
          ("gbm", "Gradient boosting", "0.55", (0, (5, 1, 1, 1, 1, 1)), "v"),
          ("process_da", "Process model + PF", "0.7", "--", "P"), ("persistence", "Persistence", "0.7", ":", "x")]
fig, ax = plt.subplots(2, 3, figsize=(COL_W, 3.5), sharex=True)
for c, sp in enumerate(["test", "warm", "geo"]):
    for r, met in enumerate(["mae", "crps"]):
        a = ax[r, c]
        for m, lab, col, ls, mk in models:
            if m not in S or sp not in S[m]:
                continue
            d = S[m][sp]
            mu = [d[f"{met}_{h}"]["mean"] for h in H]; sd = [d[f"{met}_{h}"]["sd"] for h in H]
            a.errorbar(H, mu, yerr=sd, color=col, ls=ls, marker=mk, ms=3, lw=0.9, capsize=1.5, elinewidth=0.6,
                       mfc=col if m == "twin" else "white", label=lab)
        a.set_xticks(H)
        a.tick_params(length=2, labelsize=6.3)
        if r == 0:
            a.set_title({"test": "(a) Temporal", "warm": "(b) Warming", "geo": "(c) Geographic"}[sp], loc="left",
                        fontsize=7.5)
        if r == 1:
            a.set_xlabel("Horizon (d)", fontsize=7)
    ax[0, 0].set_ylabel("MAE (pp)", fontsize=7); ax[1, 0].set_ylabel("CRPS (log scale)", fontsize=7)
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=5.8, bbox_to_anchor=(0.5, 0.0), handlelength=2.6,
           columnspacing=0.8)
fig.tight_layout(rect=(0, 0.16, 1, 1), h_pad=0.5, w_pad=0.5)
fig.savefig("fig_skill.png")
print("ok")
