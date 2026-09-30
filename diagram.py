"""Tiny box-and-arrow diagram engine with automatic layout checks:
box overlap, text overflow, arrow endpoints on the intended box borders,
arrow segments crossing unrelated boxes, and text-text collisions."""
import numpy as np
from figstyle import plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D


class Diagram:
    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis("off")
        self.W, self.H = w, h
        self.boxes = {}; self.arrows = []; self.texts = []; self.groups = {}

    def box(self, name, x, y, w, h, text, fc="white", lw=0.8, ls="-", size=6.5, bold=False, ec="0",
            valign="center", group=False):
        self.boxes[name] = (x, y, w, h)
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=ec, lw=lw, ls=ls,
                           zorder=1 if group else 2)
        self.ax.add_patch(p)
        if group:
            self.groups[name] = True
        if text:
            ty = y + h / 2 if valign == "center" else y + h - 0.06
            t = self.ax.text(x + w / 2, ty, text, ha="center", va="center" if valign == "center" else "top",
                             fontsize=size, fontweight="bold" if bold else "normal", zorder=5, linespacing=1.15)
            self.texts.append((t, name))

    def label(self, x, y, text, size=6, ha="center", style="normal", owner=None):
        t = self.ax.text(x, y, text, fontsize=size, ha=ha, va="center", style=style, zorder=6,
                         bbox=dict(fc="white", ec="none", pad=0.4))
        self.texts.append((t, owner))

    def anchor(self, name, side, frac=0.5):
        x, y, w, h = self.boxes[name]
        return {"l": (x, y + frac * h), "r": (x + w, y + frac * h),
                "b": (x + frac * w, y), "t": (x + frac * w, y + h)}[side]

    def arrow(self, src, sside, dst, dside, via=(), sfrac=0.5, dfrac=0.5, ls="-", lw=0.8, color="0",
              label=None, lpos=None, lsize=5.8):
        p0 = self.anchor(src, sside, sfrac); p1 = self.anchor(dst, dside, dfrac)
        pts = [p0] + list(via) + [p1]
        xs, ys = zip(*pts)
        self.ax.add_line(Line2D(xs[:-1] + (xs[-1],), ys, color=color, lw=lw, ls=ls, zorder=3,
                                solid_capstyle="butt"))
        # arrow head on last segment
        (xa, ya), (xb, yb) = pts[-2], pts[-1]
        L = max(np.hypot(xb - xa, yb - ya), 1e-9); back = min(0.07, 0.9 * L)
        self.ax.annotate("", xy=(xb, yb), xytext=(xb - back * (xb - xa) / L, yb - back * (yb - ya) / L),
                         arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=color, lw=lw,
                                         shrinkA=0, shrinkB=0, ls="-"), zorder=4)
        self.arrows.append(dict(src=src, dst=dst, pts=pts, ls=ls))
        if label:
            lx, ly = lpos if lpos else ((pts[-2][0] + pts[-1][0]) / 2, (pts[-2][1] + pts[-1][1]) / 2)
            self.label(lx, ly, label, size=lsize, owner=("arrow", len(self.arrows) - 1))

    # ------------------------------------------------------------ checks
    def check(self):
        errs = []
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        names = list(self.boxes)
        # box overlap (non-group boxes)
        for i, a in enumerate(names):
            for b_ in names[i + 1:]:
                if a in self.groups or b_ in self.groups:
                    continue
                if _ov(self.boxes[a], self.boxes[b_], 0.02):
                    errs.append(f"box overlap {a} / {b_}")
        # text within own box, no text-text overlap
        tb = []
        for t, owner in self.texts:
            bb = t.get_window_extent(r)
            (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            tb.append(((x0, y0, x1 - x0, y1 - y0), owner, t.get_text()))
            if isinstance(owner, str) and owner in self.boxes:
                bx = self.boxes[owner]
                if x0 < bx[0] + 0.01 or x1 > bx[0] + bx[2] - 0.01 or y0 < bx[1] + 0.005 or y1 > bx[1] + bx[3] - 0.005:
                    if owner not in self.groups:
                        errs.append(f"text overflow in {owner}: {t.get_text()[:30]!r}")
            if x0 < 0 or y0 < 0 or x1 > self.W or y1 > self.H:
                errs.append(f"text outside canvas {t.get_text()[:30]!r}")
        for i in range(len(tb)):
            for j in range(i + 1, len(tb)):
                if _ov(tb[i][0], tb[j][0], 0.0):
                    errs.append(f"text collision {tb[i][2][:25]!r} / {tb[j][2][:25]!r}")
        # free labels must not sit on boxes they don't belong to
        for (bb, owner, txt) in tb:
            if isinstance(owner, tuple):
                for n, bx in self.boxes.items():
                    if n in self.groups:
                        continue
                    if _ov(bb, bx, -0.005):
                        errs.append(f"label {txt[:25]!r} overlaps box {n}")
        # labels must not sit on arrows other than their own
        for (bb, owner, txt) in tb:
            for k, a in enumerate(self.arrows):
                if owner == ("arrow", k):
                    continue
                for p, q in zip(a["pts"][:-1], a["pts"][1:]):
                    if _seg_box(p, q, bb, shrink=-0.01):
                        errs.append(f"label {txt[:25]!r} sits on arrow {a['src']}->{a['dst']}")
        # arrow segments crossing boxes other than src/dst (groups allowed)
        for k, a in enumerate(self.arrows):
            for n, bx in self.boxes.items():
                if n in (a["src"], a["dst"]) or n in self.groups:
                    continue
                for p, q in zip(a["pts"][:-1], a["pts"][1:]):
                    if _seg_box(p, q, bx):
                        errs.append(f"arrow {a['src']}->{a['dst']} crosses {n}")
            # endpoints on borders
            for pt, n in ((a["pts"][0], a["src"]), (a["pts"][-1], a["dst"])):
                if not _on_border(pt, self.boxes[n]):
                    errs.append(f"arrow endpoint not on border of {n}")
            # segments axis aligned or deliberate
        # arrow-arrow overlap (collinear shared segments)
        segs = []
        for k, a in enumerate(self.arrows):
            for p, q in zip(a["pts"][:-1], a["pts"][1:]):
                segs.append((k, p, q))
        for i in range(len(segs)):
            for j in range(i + 1, len(segs)):
                if segs[i][0] == segs[j][0]:
                    continue
                if _collinear_overlap(segs[i][1], segs[i][2], segs[j][1], segs[j][2]):
                    errs.append(f"arrows {self.arrows[segs[i][0]]['src']}->{self.arrows[segs[i][0]]['dst']} and "
                                f"{self.arrows[segs[j][0]]['src']}->{self.arrows[segs[j][0]]['dst']} overlap")
        return errs

    def save(self, path):
        errs = self.check()
        self.fig.savefig(path, dpi=400)
        return errs


def _ov(a, b, m):
    return not (a[0] + a[2] <= b[0] + m or b[0] + b[2] <= a[0] + m or a[1] + a[3] <= b[1] + m or b[1] + b[3] <= a[1] + m)


def _on_border(pt, bx, tol=1e-6):
    x, y = pt; x0, y0, w, h = bx
    inx = x0 - tol <= x <= x0 + w + tol; iny = y0 - tol <= y <= y0 + h + tol
    return (inx and iny) and (abs(x - x0) < tol or abs(x - x0 - w) < tol or abs(y - y0) < tol or abs(y - y0 - h) < tol)


def _seg_box(p, q, bx, shrink=0.015):
    x0, y0, w, h = bx
    x0 += shrink; y0 += shrink; w -= 2 * shrink; h -= 2 * shrink
    for t in np.linspace(0, 1, 60):
        x = p[0] + t * (q[0] - p[0]); y = p[1] + t * (q[1] - p[1])
        if x0 < x < x0 + w and y0 < y < y0 + h:
            return True
    return False


def _collinear_overlap(p1, q1, p2, q2, tol=0.01):
    # horizontal
    if abs(p1[1] - q1[1]) < tol and abs(p2[1] - q2[1]) < tol and abs(p1[1] - p2[1]) < tol:
        a0, a1 = sorted([p1[0], q1[0]]); b0, b1 = sorted([p2[0], q2[0]])
        return min(a1, b1) - max(a0, b0) > tol
    if abs(p1[0] - q1[0]) < tol and abs(p2[0] - q2[0]) < tol and abs(p1[0] - p2[0]) < tol:
        a0, a1 = sorted([p1[1], q1[1]]); b0, b1 = sorted([p2[1], q2[1]])
        return min(a1, b1) - max(a0, b0) > tol
    return False
