#!/usr/bin/env bash
cd /home/snics/MER-DA
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=""
V=DAI/checks/DAI-20/post_tools_run_v3
nice -n 10 ./.venv/bin/python DAI/docs/tools_pending/dai20_posthoc_intercept.py --units DAI/checks/DAI-20/post_tools_run_v2/fake_eval/units --out $V/fake_new_posthoc > $V/fake_new.log 2>&1
echo "status=$?" >> $V/fake_new.log
touch $V/fake_new.done
