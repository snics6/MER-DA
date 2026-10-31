"""E30：非線形の分類器（2 層のパーセプトロン）で，ソースのみ・合併・D1（切片だけを直す）を，中心合わせの有無で学ぶ（GPU）。

予言と設定は `docs/exploratory-analyses.md` E30。多数の小さいネットを 1 回の計算に束ねて学ぶ（問ごとに行の重みが違うだけ）。
学習に使う行は，ソースの行と，どれかの問で使うターゲットの K 本だけ。テストと K 本は地図と同じ。

    ./.venv/bin/python exp/mlp_check.py --kind audio_wavlm [--target …] [--source …]

出力：`results/mlp_check_<kind>/<ターゲット>.npz`（`test_idx_r{r}`・`y_test_r{r}`，
`{src|m|d1}{c?}_wd{wd}_{ソース}[_K{K}]_r{r}`：テストの点数），`<ターゲット>.csv`（UAR，最終の損失），`provenance.json`
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exp"))
sys.path.insert(0, str(ROOT / "measure"))
import ablate  # noqa: E402
import map as M  # noqa: E402
import shift  # noqa: E402
from data import groups_for  # noqa: E402

KS = ("1", "5", "10")
WDS = (1e-4, 1e-2)
HIDDEN, STEPS, LR = 256, 500, 1e-3
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False          # 再現性のため TF32 は使わない


def balanced(y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """行の重み w（0/1）の中で感情を均等にする重み n/(k n_y)。"""
    out = torch.zeros_like(w)
    on = w > 0
    cnt = torch.bincount(y[on], minlength=4).to(w.dtype)
    out[on] = on.sum().to(w.dtype) / (4 * cnt[y[on]])
    return out


def train(Z: torch.Tensor, y: torch.Tensor, Wt: torch.Tensor, wds: torch.Tensor, seed: int):
    """Z (n,d)，y (n,)，Wt (B,n) 行の重み（感情を均等にした後），wds (B,)。束ねた 2 層のネットの重みを返す。"""
    B, n = Wt.shape
    d = Z.shape[1]
    g = torch.Generator(device=DEV).manual_seed(seed)
    b1, b2 = 1 / np.sqrt(d), 1 / np.sqrt(HIDDEN)
    W1 = ((torch.rand(B, d, HIDDEN, generator=g, device=DEV) * 2 - 1) * b1).requires_grad_(True)
    c1 = ((torch.rand(B, HIDDEN, generator=g, device=DEV) * 2 - 1) * b1).requires_grad_(True)
    W2 = ((torch.rand(B, HIDDEN, 4, generator=g, device=DEV) * 2 - 1) * b2).requires_grad_(True)
    c2 = ((torch.rand(B, 4, generator=g, device=DEV) * 2 - 1) * b2).requires_grad_(True)
    opt = torch.optim.Adam([W1, c1, W2, c2], lr=LR)
    S = Wt.sum(1)
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        L = forward(Z, W1, c1, W2, c2)                                      # (B, n, 4)
        nll = -torch.log_softmax(L, 2).gather(2, y.view(1, n, 1).expand(B, n, 1)).squeeze(2)
        loss_b = (Wt * nll).sum(1) / S + wds / 2 * ((W1 ** 2).sum((1, 2)) + (W2 ** 2).sum((1, 2)))
        loss_b.sum().backward()
        opt.step()
    return (W1.detach(), c1.detach(), W2.detach(), c2.detach()), loss_b.detach().cpu().numpy()


def forward(Z, W1, c1, W2, c2):
    B = W1.shape[0]
    H = torch.relu((Z @ W1.permute(1, 0, 2).reshape(Z.shape[1], -1)).view(Z.shape[0], B, HIDDEN).permute(1, 0, 2) + c1[:, None, :])
    return torch.bmm(H, W2) + c2[:, None, :]


def run_target(kind: str, tgt: M.Group, sources: list[M.Group], reps, out: Path) -> pd.DataFrame:
    Xt = tgt.X.float()
    yt = torch.as_tensor(np.array(tgt.y), device=DEV)
    split, saved, rows = {}, {}, []
    for r in reps:
        te = M.test_mask(tgt, r)
        order = M.kshot_order(tgt, np.flatnonzero(~te), r)
        split[r] = (te, {K: M.take_k(order, K) for K in KS}, np.unique(tgt.person[te]))
        saved[f"test_idx_r{r}"] = np.flatnonzero(te)
        saved[f"y_test_r{r}"] = tgt.y[te]
    ct_all = Xt.mean(0)
    for s in sources:
        t0 = time.perf_counter()
        groups = {}
        for r in reps:
            keep = ~np.isin(s.person, split[r][2])
            groups.setdefault(keep.tobytes(), []).append(r)
        for gi, (kb, rs) in enumerate(groups.items()):
            keep = np.frombuffer(kb, dtype=bool)
            Xs = M.rows(s.X, keep).float()
            ys = torch.as_tensor(np.array(s.y[keep]), device=DEV)
            ns = len(ys)
            mu, sd = Xs.mean(0), Xs.std(0, correction=0)
            sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
            used = np.unique(np.concatenate([split[r][1][K] for r in rs for K in KS]))
            pos = {int(i): j for j, i in enumerate(used)}
            for cen in (False, True):
                if cen:
                    Zs, Zt = (Xs - Xs.mean(0)) / sd, (Xt - ct_all) / sd
                else:
                    Zs, Zt = (Xs - mu) / sd, (Xt - mu) / sd
                Z = torch.cat([Zs, Zt[torch.as_tensor(used, device=DEV)]])
                y = torch.cat([ys, yt[torch.as_tensor(used, device=DEV)]])
                n = len(y)
                names, Wt, wds = [], [], []
                for wd in WDS:
                    w = torch.zeros(n, device=DEV); w[:ns] = 1
                    names.append(("src", wd, None, None)); Wt.append(balanced(y, w)); wds.append(wd)
                    for r in rs:
                        for K in KS:
                            w = torch.zeros(n, device=DEV); w[:ns] = 1
                            w[torch.as_tensor([ns + pos[int(i)] for i in split[r][1][K]], device=DEV)] = 1
                            names.append(("m", wd, r, K)); Wt.append(balanced(y, w)); wds.append(wd)
                seed = zlib.crc32(f"{kind}|{tgt.name}|{s.name}|{cen}|{gi}".encode())
                params, final = train(Z, y, torch.stack(Wt), torch.tensor(wds, device=DEV), seed)
                with torch.no_grad():
                    Lt = forward(Zt, *params).cpu().numpy()                     # (B, nt, 4)
                tag = "c" if cen else ""
                for b, (kindm, wd, r, K) in enumerate(names):
                    if kindm == "src":
                        for rr in rs:
                            te, idx, _ = split[rr]
                            L_te = Lt[b][te]
                            saved[f"src{tag}_wd{wd:g}_{s.name}_r{rr}"] = L_te.astype(np.float32)
                            rows.append({"target": tgt.name, "source": s.name, "rep": rr, "K": "", "method": f"src{tag}", "wd": wd,
                                         "uar": shift.uar(tgt.y[te], L_te.argmax(1)), "final_loss": float(final[b])})
                            for KK in KS:
                                bias = ablate.fit_class_bias(Lt[b][idx[KK]].astype(np.float64), tgt.y[idx[KK]])
                                Ld = L_te + bias
                                saved[f"d1{tag}_wd{wd:g}_{s.name}_K{KK}_r{rr}"] = Ld.astype(np.float32)
                                rows.append({"target": tgt.name, "source": s.name, "rep": rr, "K": KK, "method": f"d1{tag}", "wd": wd,
                                             "uar": shift.uar(tgt.y[te], Ld.argmax(1))})
                    else:
                        te = split[r][0]
                        L_te = Lt[b][te]
                        saved[f"m{tag}_wd{wd:g}_{s.name}_K{K}_r{r}"] = L_te.astype(np.float32)
                        rows.append({"target": tgt.name, "source": s.name, "rep": r, "K": K, "method": f"m{tag}", "wd": wd,
                                     "uar": shift.uar(tgt.y[te], L_te.argmax(1)), "final_loss": float(final[b])})
                del params, Lt
                torch.cuda.empty_cache()
        print(f"  {kind} {tgt.name} ← {s.name}：{time.perf_counter() - t0:.1f} 秒（束 {len(groups)}）", flush=True)
    np.savez_compressed(out / f"{tgt.name}.npz", **saved)
    d = pd.DataFrame(rows)
    d.to_csv(out / f"{tgt.name}.csv", index=False)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--target", nargs="+")
    ap.add_argument("--source", nargs="+")
    ap.add_argument("--reps", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()
    out = args.out_dir or ROOT / "results" / f"mlp_check_{args.kind}"
    out.mkdir(parents=True, exist_ok=True)
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    (out / "provenance.json").write_text(json.dumps({
        "script": "exp/mlp_check.py", "argv": sys.argv[1:], "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "K": list(KS), "reps": args.reps, "wds": WDS, "hidden": HIDDEN, "steps": STEPS, "lr": LR}, ensure_ascii=False, indent=1))
    groups = groups_for(args.kind)
    data = {g: M.load(g, args.kind) for g in groups}
    for t in args.target or groups:
        if (out / f"{t}.npz").exists():
            print(f"{args.kind} {t}：済み（飛ばす）", flush=True)
            continue
        t0 = time.perf_counter()
        run_target(args.kind, data[t], [data[s] for s in (args.source or groups) if s != t], args.reps, out)
        print(f"{args.kind} {t}：{time.perf_counter() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
