#!/bin/sh
set -e
python build_dataset.py
python encoder_mae.py
for s in 0 1 2; do python exp_twin.py $s; OMP_NUM_THREADS=1 python exp_base.py $s; done
python run_eval.py
OMP_NUM_THREADS=1 python exp_robust.py
OMP_NUM_THREADS=1 python exp_cost.py
for f in fig_arch fig_data fig_skill fig_cf fig_abl_conf fig_interp fig_case; do python $f.py; done
