"""gate-code（評価の回の起動の台本）：実験 3 の eval_unit を，仮のターゲットの MOSEI の単位（T=AFEW, c=MOSEI）に当て，
GPU のメモリの最大と所要を測る．成績の値は出さない．results/ には書かない（eval_unit はファイルを書かない）．
使い方：MERDA_GPU_FRACTION=0.9 ./.venv/bin/python DAI/checks/DAI-eval/code_run/m1_dai3_mem.py <T> <c> <f> <r> [<f> <r> ...]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "exp"))
import torch
import dai1, dai3  # noqa: E402

ROOT = dai3.ROOT
out1 = ROOT / "results" / "dai1"
fz1 = json.load(open(out1 / "select" / "frozen.json"))
fz3 = json.load(open(ROOT / "results" / "dai3" / "frozen.json"))
T, c = sys.argv[1], sys.argv[2]
pairs = [(int(sys.argv[i]), int(sys.argv[i + 1])) for i in range(3, len(sys.argv), 2)]
GB = 1024 ** 3
for f, r in pairs:
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    sv = dai3.Saved(out1, "pseudo", T, c, f, r, fz1[T])
    sv.T = c                                 # 評価の回と同じく，テストのコーパスの部分集合と分岐を使う（評価の回は c = T）
    t1 = time.time()
    a1 = torch.cuda.max_memory_allocated() / GB
    R = dai3.eval_unit(sv, fz3, first_seed=(r == 0))
    t2 = time.time()
    print(f"{T}_{c}_f{f}_r{r}: Saved {t1-t0:.0f} 秒（最大 {a1:.2f} GB），eval_unit {t2-t1:.0f} 秒，"
          f"最大の確保 {torch.cuda.max_memory_allocated()/GB:.2f} GB，最大の予約 {torch.cuda.max_memory_reserved()/GB:.2f} GB，"
          f"識別器 {len(R['refit'])}，n_test {R['meta']['n_test']}，n_pool {R['meta']['n_pool']}，"
          f"解いた切片の問 {sum(x['n_problems'] for x in R['conv'])}", flush=True)
    del sv, R
    torch.cuda.empty_cache()
    print(f"  後の確保 {torch.cuda.memory_allocated()/GB:.2f} GB", flush=True)
