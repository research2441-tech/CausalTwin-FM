"""Fig. 1: CausalTwin-FM system architecture (full page width)."""
from diagram import Diagram

D = Diagram(7.0, 3.5)
G = "0.93"
# inputs
D.box("img", 0.06, 2.95, 1.3, 0.44, "Canopy images\n(48×48 RGB, every 3 d)")
D.box("iot", 0.06, 2.39, 1.3, 0.44, "IoT microclimate (canopy\ntemperature, wetness, soil)")
D.box("wx", 0.06, 1.83, 1.3, 0.44, "Weather history\n(station trace)")
D.box("mg", 0.06, 1.27, 1.3, 0.44, "Management history\n(sprays, irrigation, N)")
D.box("fut", 0.06, 0.62, 1.3, 0.5, "Future drivers: weather\nforecast + action plan $a$")
# encoders
D.box("enc", 1.62, 2.95, 1.3, 0.44, "Frozen MAE-ViT canopy\nencoder + severity head", fc=G)
D.box("abd", 1.62, 1.27, 1.3, 1.42, "Amortised abduction\nencoder $q_\\phi$\n(pre-decision history)\n\nposterior over\n$L^{(1)},L^{(2)},I,R,\\theta,\\iota$\nat origin $\\tau$", fc=G)
# twin group
D.box("twin", 3.2, 0.62, 1.95, 2.77, "Compartment-anchored twin", fc="0.975", group=True, valign="top", bold=True,
      size=6.6)
D.box("chain", 3.3, 2.58, 1.75, 0.47, "$H\\rightarrow L^{(1)}\\rightarrow L^{(2)}\\rightarrow I\\rightarrow R$\nthermal-time latency")
D.box("att", 3.3, 1.98, 1.75, 0.42, "Thermal-time cross-modal\nattention residual $r_t$")
D.box("obs", 3.3, 1.42, 1.75, 0.38, "Observation operator $g(I,R)$")
D.box("ports", 3.3, 0.72, 1.75, 0.5, "Typed intervention ports:\nresidue $\\pi_t$, wetness $W_t$")
# outputs
D.box("roll", 5.42, 2.75, 1.5, 0.62, "Rollouts: factual forecast\nand action queries do($a$)")
D.box("conf", 5.42, 1.85, 1.5, 0.62, "Season-chained\nadaptive conformal\n+ OOD flag", fc=G)
D.box("out", 5.42, 0.8, 1.5, 0.75, "Outputs: 21-day intervals,\nonset risk, action\neffects, expert\nreferral flag")
# losses
D.box("l2", 1.62, 0.05, 1.3, 0.42, "Stage-2 loss: 21-day\nmixture likelihood", ls="--")
D.box("l1", 3.2, 0.05, 1.95, 0.42, "Stage-1 loss: season likelihood\n(expert labels + image operator)", ls="--")

# forward arrows
D.arrow("img", "r", "enc", "l")
D.arrow("enc", "b", "abd", "t")
D.arrow("iot", "r", "abd", "l", dfrac=(2.61 - 1.27) / 1.42)
D.arrow("wx", "r", "abd", "l", dfrac=(2.05 - 1.27) / 1.42)
D.arrow("mg", "r", "abd", "l", dfrac=(1.49 - 1.27) / 1.42)
D.arrow("fut", "r", "ports", "l", sfrac=0.5, dfrac=(0.87 - 0.72) / 0.5, label="enters only through ports",
        lpos=(2.62, 0.99))
D.arrow("abd", "r", "chain", "l", sfrac=(2.52 - 1.27) / 1.42, dfrac=(2.93 - 2.58) / 0.47,
        via=[(3.08, 2.52), (3.08, 2.93)])
D.arrow("att", "t", "chain", "b")
D.arrow("chain", "l", "obs", "l", sfrac=(2.68 - 2.58) / 0.47, dfrac=0.5, via=[(3.25, 2.68), (3.25, 1.61)])
D.arrow("ports", "r", "chain", "r", sfrac=0.5, dfrac=(2.68 - 2.58) / 0.47, via=[(5.1, 0.97), (5.1, 2.68)])
D.arrow("chain", "r", "roll", "l", sfrac=(2.95 - 2.58) / 0.47, dfrac=(2.95 - 2.75) / 0.62)
D.arrow("roll", "b", "conf", "t")
D.arrow("conf", "b", "out", "t")
# reverse-mode gradients (dashed)
D.arrow("l1", "t", "twin", "b", ls="--", label="$\\nabla_\\vartheta$ (reverse mode)", lpos=(4.62, 0.545), lsize=5.6)
D.arrow("l2", "t", "abd", "b", ls="--", sfrac=0.25, dfrac=0.25, label="$\\nabla_\\phi$", lpos=(2.1, 0.66), lsize=5.6)
errs = D.save("fig1_architecture.png")
print("\n".join(errs) if errs else "layout checks passed")
