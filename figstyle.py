"""Shared grayscale publication style."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 400,
    "axes.grid": True, "grid.color": "0.88", "grid.linewidth": 0.4, "legend.frameon": False,
    "mathtext.fontset": "stix",
})
GRAYS = ["0.0", "0.35", "0.55", "0.7", "0.82"]
LS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
MK = ["o", "s", "^", "D", "v", "P", "X", "<"]
HATCH = ["", "////", "....", "xxxx", "\\\\\\\\", "++++", "oooo", "----"]
COL_W = 3.45   # inches, one column
FULL_W = 7.0
