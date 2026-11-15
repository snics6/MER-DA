#!/usr/bin/env bash
cd /home/snics/MER-DA
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=""
V=DAI/checks/DAI-20/post_tools_run_v2/cases
one() { N=$1; L=$V/$N.log; nice -n 10 ./.venv/bin/python DAI/docs/tools_pending/dai20_posthoc_intercept.py --units $V/$N/units --out $V/$N/posthoc > $L 2>&1; echo "status=$? 出力=$(ls $V/$N/posthoc 2>/dev/null | wc -l)" >> $L; }
( one G_est; one H_lo; one I_lv; one P_swap ) &
( one J_noprod; one K_nosum; one O_fewrows ) &
wait
touch $V/all.done
