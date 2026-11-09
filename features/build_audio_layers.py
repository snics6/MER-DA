"""音声の**全層**の特徴を作る（DAI-13 の下調べ．2026-11-09）。

`build_audio.py` と同じ前処理（動画 → 16kHz の波形 → 先頭 60 秒 → 平均 0・標準偏差 1）で WavLM-large に通し，
最終層だけでなく **25 個の隠れ状態（CNN の出力に位置の畳み込みを足した入力 1 つ ＋ Transformer の 24 層）** のそれぞれを，
詰め物の部分を除いて時間方向に平均する（1 発話あたり 25 × 1024）。

- 最後の隠れ状態（添字 24）は `last_hidden_state` と同じものなので，`audio_wavlm_<コーパス>.npz` の `X` と一致するはずである
  （`--check` でその差を測る．fp32，同じモデル，同じ前処理）。
- モデルは凍結，微調整なし，fp32（`build_audio.py` と同じ）。評価モードなので layerdrop は働かず全部の層を通す。
- 出力：`data/cache/audio_wavlm_layers_<コーパス>.npz`（`X_layers` (n, 25, 1024) float32 と，`build_audio.py` と同じ列）。
  途中保存は `data/cache/parts/audio_wavlm_layers_<コーパス>_<番号>.npz`。
- **既存の特徴のファイルは書き換えない**（`data/README.md` 4 節の取り決めのとおり，特徴は別のファイルに足す）。

使い方:

    ./.venv/bin/python features/build_audio_layers.py --corpus AFEW --limit 20 --check   # 試走と照合
    ./.venv/bin/python features/build_audio_layers.py --corpus AFEW IEMOCAP MELD MOSEI    # 本番（4 コーパス）
    ./.venv/bin/python features/build_audio_layers.py --corpus AFEW IEMOCAP MELD MOSEI --merge
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from build_audio import SR, decode, normalize  # noqa: E402

MODEL = "microsoft/wavlm-large"
KIND = "audio_wavlm_layers"
N_STATES = 25             # 入力 1 ＋ 24 層
CHECK_TOL = 1e-3          # --check：最後の層の平均と既存の X の最大の差の許し（試走 24 発話で 1.66e−5）．超えればエラーで止まる


def encode_layers(model, wavs: list[np.ndarray], device: str, budget: int, max_size: int) -> np.ndarray:
    """波形の列 → (n, 25, 1024)。各隠れ状態を，詰め物の部分を除いて時間方向に平均する。"""
    out = np.zeros((len(wavs), N_STATES, model.config.hidden_size), dtype=np.float32)
    lengths = [len(w) for w in wavs]
    with torch.inference_mode():
        for idx in common.batches_by_budget(lengths, budget, max_size):
            n = max(lengths[i] for i in idx)
            x = torch.zeros(len(idx), n)
            for r, i in enumerate(idx):
                x[r, : lengths[i]] = torch.from_numpy(wavs[i])
            x = x.to(device)
            lens = torch.tensor([lengths[i] for i in idx], device=device)
            mask = (torch.arange(n, device=device)[None] < lens[:, None]).long()
            res = model(x, attention_mask=mask, output_hidden_states=True)
            hs = res.hidden_states                                             # 25 個の (B, T, D)
            if len(hs) != N_STATES:
                raise SystemExit(f"隠れ状態の数が {len(hs)}（{N_STATES} のはず）")
            if not torch.equal(hs[-1], res.last_hidden_state):
                raise SystemExit("最後の隠れ状態が last_hidden_state と違う（モデルの実装が前提と違う）")
            flen = model._get_feat_extract_output_lengths(lens).to(device)
            fmask = (torch.arange(hs[0].shape[1], device=device)[None] < flen[:, None]).float()
            den = fmask.sum(1, keepdim=True)
            H = torch.stack(hs, 1)                                             # (B, 25, T, D)
            pooled = (H * fmask[:, None, :, None]).sum(2) / den[:, :, None]     # (B, 25, D)
            out[idx] = pooled.float().cpu().numpy()
    return out


def process_chunk(df: pd.DataFrame, model, args) -> dict:
    n = len(df)
    dim = model.config.hidden_size
    X = np.full((n, N_STATES, dim), np.nan, dtype=np.float32)
    dur = np.full(n, np.nan, dtype=np.float32)
    trunc = np.zeros(n, dtype=bool)
    ok = np.zeros(n, dtype=bool)
    reason = np.array([""] * n, dtype=object)
    paths = common.media_paths(df, prefer_audio=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        decoded = list(pool.map(decode, paths))
    keep, wavs = [], []
    for i, (w, sec, tr) in enumerate(decoded):
        dur[i], trunc[i] = sec, tr
        if not paths[i]:
            reason[i] = "音声ファイルが無い"
            continue
        if len(w) < SR // 100:
            reason[i] = "音声を取り出せない"
            continue
        keep.append(i)
        wavs.append(normalize(w))
    if keep:
        X[keep] = encode_layers(model, wavs, args.device, args.budget * SR, args.max_batch)
        ok[keep] = True
    return {
        "X_layers": X, "ok": ok, "duration_sec": dur, "truncated": trunc,
        "fail_reason": np.asarray(reason, dtype=np.str_),
        **{c: df[c].astype(str).to_numpy(dtype=np.str_) for c in ("sample_id", "corpus", "split", "speaker_id", "dialog_type")},
        "label_4": df["label_4"].to_numpy(dtype=np.int8),
    }


def run_corpus(corpus: str, model, args) -> None:
    df = common.load_manifest(corpus)
    if args.limit:
        df = df.head(args.limit)
    print(f"\n=== {corpus}: {len(df)} 発話 ===", flush=True)
    t0 = time.perf_counter()
    for ci, start in enumerate(range(0, len(df), args.chunk)):
        part = common.part_path(KIND, corpus, ci)
        if part.exists() and not args.overwrite:
            print(f"  [{ci}] 済み: {part.name}", flush=True)
            continue
        sub = df.iloc[start : start + args.chunk].reset_index(drop=True)
        t1 = time.perf_counter()
        payload = process_chunk(sub, model, args)
        np.savez(part, **payload)
        dt = time.perf_counter() - t1
        print(f"  [{ci}] {len(sub)} 件 / 成功 {int(payload['ok'].sum())} / 切り詰め {int(payload['truncated'].sum())} / {dt:.1f} 秒 "
              f"({dt / max(len(sub), 1):.3f} 秒/件) → {part.name}", flush=True)
    print(f"{corpus} 完了: {time.perf_counter() - t0:.1f} 秒", flush=True)


def check_against_cache(corpus: str, limit: int) -> None:
    """最後の隠れ状態の平均（添字 24）が，`audio_wavlm_<コーパス>.npz` の X と一致するかを測る（同じ発話を sample_id で合わせる）。"""
    z = common.load_cache(KIND, corpus)
    ref = np.load(common.out_path("audio_wavlm", corpus), allow_pickle=False)
    pos = {s: i for i, s in enumerate(ref["sample_id"].astype(str))}
    ids = z["sample_id"].astype(str)
    if limit:
        ids = ids[:limit]
    rows = [(i, pos[s]) for i, s in enumerate(ids) if s in pos and z["ok"][i] and ref["ok"][pos[s]]]
    a = np.stack([z["X_layers"][i, -1] for i, _ in rows]); b = np.stack([ref["X"][j] for _, j in rows])
    d = np.abs(a - b)
    print(f"{corpus}: 照合 {len(rows)} 発話．最後の層の平均と既存の X の差：最大 {d.max():.2e}，平均 {d.mean():.2e}（値の平均の大きさ {np.abs(b).mean():.3e}）．"
          f"ok の一致 {int((z['ok'][:len(ids)] == ref['ok'][[pos[s] for s in ids]]).sum())}/{len(ids)}", flush=True)
    if d.max() > CHECK_TOL:
        raise SystemExit(f"{corpus}：最後の層の平均と既存の X の最大の差 {d.max():.2e} が {CHECK_TOL} を超える（同じ重み・同じ前処理で計算できていない）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", nargs="+", default=["AFEW", "IEMOCAP", "MELD", "MOSEI"])
    ap.add_argument("--limit", type=int, default=0, help="先頭から何件だけ処理するか（試走用）")
    ap.add_argument("--chunk", type=int, default=1000, help="途中保存の単位（25 層ぶんなので build_audio.py より小さい）")
    ap.add_argument("--budget", type=int, default=240, help="1 バッチに入れる音声の合計秒数の目安（25 層を持つぶん build_audio.py より小さい）")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--check", action="store_true", help="最後の層の平均を既存の特徴と照合する")
    args = ap.parse_args()
    if args.merge:
        for c in args.corpus:
            common.merge_npz(KIND, c)
        if args.check:
            for c in args.corpus:
                check_against_cache(c, args.limit)
        return
    from transformers import AutoModel
    model = AutoModel.from_pretrained(MODEL).to(args.device).eval()
    print(f"{MODEL} を {args.device} に載せた（fp32・凍結・微調整なし・全 {N_STATES} 隠れ状態）次元 {model.config.hidden_size}", flush=True)
    for c in args.corpus:
        run_corpus(c, model, args)
    if args.check:
        for c in args.corpus:
            check_against_cache(c, args.limit)


if __name__ == "__main__":
    main()
