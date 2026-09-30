# CausalTwin-FM: compartment-anchored digital twin with pre-decision abduction

Code for the manuscript *Compartment-Anchored Digital Twin with Pre-Decision Abduction for Counterfactual
Cotton Disease Forecasting under Climate and Regional Shift*. Every file sits in this one folder;
generated data, results and figures are also written here.

## Requirements
Python 3.11, JAX 0.10 (CPU is enough), Optax 0.2.8, NumPy 2.4, SciPy 1.17, scikit-learn 1.8, pandas,
matplotlib. Install with:

    pip install "jax[cpu]" optax numpy scipy scikit-learn pandas matplotlib

## Data sources included in this folder
* `hyderabad_climate.txt`, `cordoba_climate.txt`, `tunis_climate.txt`: daily station weather distributed
  with AquaCrop-OSPy (Kelly and Foster, Agric. Water Manage. 2021).
* `723170TYA.CSV`: hourly TMY3 record of Greensboro, NC, distributed with pvlib; used only to
  calibrate the daily leaf-wetness model.
* `cmip6_*.csv`: CMIP6 daily output (CanESM5, MIROC6, NorESM2-LM; historical 1995-2014 and SSP5-8.5
  2060-2079), distributed with AquaCrop-OSPy; used for warming deltas.

## Pipeline (run in this order)
| Step | Script | Output |
|---|---|---|
| 1 | `python build_dataset.py` | `bench.npz`, `bench_meta.json` (benchmark: epidemics, images, IoT, exact counterfactual replays; seed 2026) |
| 2 | `python encoder_mae.py` | `emb.npz`, `encoder.json` (MAE-ViT pretraining on train+calibration images, severity heads) |
| 3 | `python exp_twin.py SEED` for SEED in 0 1 2 | `pred_twin_sSEED.npz`, ablations, filter mode (`abl_pf`), process model (`process_da`) |
| 4 | `python exp_base.py SEED` for SEED in 0 1 2 | persistence, gradient boosting, static network, GRU, Transformer, CRN |
| 5 | `python run_eval.py` | `summary.json` (all metrics, conformal variants, paired statistics) |
| 6 | `python exp_robust.py`, `python exp_cost.py` | `robust.json`, `cost.json` |
| 7 | `python fig_arch.py`, `fig_data.py`, `fig_skill.py`, `fig_cf.py`, `fig_abl_conf.py`, `fig_interp.py`, `fig_case.py` | figures (PNG, grayscale) |

`run_all.sh` runs the whole pipeline. On two CPU cores the full run takes several hours; a single
seed of the twin takes about 10 minutes.

## Module map
* `weather.py`: station traces, TMY3 wetness calibration, CMIP6 deltas.
* `pathosim.py`: ground-truth pathosystem generator with confounded management and common-random-number replays.
* `render.py`: canopy image renderer.
* `common.py`: benchmark access, forecast emulation, metrics.
* `twin.py`: compartment-anchored twin, stage-1 training, particle-filter abduction, drivers.
* `twin_amort.py`: amortised pre-decision abduction encoder (stage 2), forecasts and action queries.
* `run_twin.py`: future-driver construction and filter-mode forecasting.
* `seqdata.py`, `baselines.py`: baseline inputs and models.
* `conformal.py`, `evaluate.py`: exposure index, weighted quantiles, season-chained adaptive conformal, metrics and statistics.
* `diagram.py`, `figstyle.py`: figure engine with automatic overlap and arrow checks, and grayscale style.

## Reproducibility notes
The benchmark is deterministic given seed 2026. Learned models use seeds 0, 1 and 2. The only
data-dependent choices (process-noise scale, horizon error scale, conformal quantiles) use the
calibration seasons; test seasons are never used for selection.
