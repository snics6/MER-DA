"""DAI 実験 2：勾配法の手法（xMUDA 型，DANN，MME，その同じ形の対照）と CORAL（設計は `DAI/docs/実験2_設計.md` 第 3.2 版の 1 節・2.2 節・2.3 節）。

段（--stage）：
  pseudo：`dai2.py` と同じ単位（T, c, 分割 f, K 本の抜き方の種 r），r ∈ {0, 1}．
  eval  ：評価のターゲット T をソース 3 つで回す．勾配法の手法（map・xmuda）は r ∈ {0, 1, 2}，CORAL（線形．初期化の種を持たない）は
          r ∈ {0, …, R−1}（R は実験 1 の固定のファイル）．流す条件は `dai2.check_eval_frozen`（`dai2.py` と同じ）．
  select：`dai2.py --stage select`（`dai2_select.py`）で，`dai2.py` の出力と合わせて選ぶ．

種の名前（設計 1 節．S24，C21，M20）：K 本の抜き方の種 r と初期化の種 s を分けて書き，勾配法の手法は s = r とする（1 単位に初期化 1 つ）．
s が決めるもの：判別器の初期値の乱数（torch.Generator，種 1000 + s），候補とソースのミニバッチの番号の乱数（種 5000 + s．同じ s の DANN・MME・
対照は同じミニバッチの列を通る），xMUDA 型のミニバッチの乱数（種 2000 + s）．行の seed_init に残す．

部分（--parts．単位ごとに別のファイル `<T>_<c>_f<f>_r<r>_<部分>.npz/.csv`）：
  map   ：DANN（Ganin ほか 2016），MME（Saito ほか 2019），同じ形の対照 mapctl．生徒 A，V，T，AVT．
          候補の標準化した特徴 z に写像 A（恒等で初期化）をかけ，頭 W・b（出発点で初期化）を載せる．予測は W A z + b．問の目的（設計 1 節
          「目的関数の形」．分子は E78 の `_train_map` と同じ）：[K 本の交差エントロピーの和（実験 1 と同じ釣り合いの重み）＋ ‖W − W0‖²/(2C′)
          ＋ ‖b − b0‖²/(2C_b) ＋ μ‖A − I‖² ＋ DA の項]（E78 と同じく S で割らない）．S（記録だけ）は K 本の重みの和 4K ＋ DA の項の重み（DANN・MME は λn，MME はさらに sw·n）．
          mapctl の S は 4K（K = 0 では 0 なので 1e−12 で下を切る．出発点で勾配が 0 なので動かない）．
          (C′, C_b) は実験 1 の (生徒, K) の値（K=0 は K=1 の値）．DA の項は候補 1 本あたりの重み（n × ミニバッチの平均）：
            DANN：領域の判別器（隠れ層 1 つ 256，ReLU）の 2 値の交差エントロピーの平均 × n．写像には逆の勾配を λ × (2/(1 + e^{−10p}) − 1)
                  （p は反復の進み 0→1．原論文の予定）の係数で返す．ソースのデータは DA の項にだけ使う．
            MME ：候補の予測のエントロピーの平均に −λn（頭は大きく，写像は逆の勾配で小さく）＋ ソースのラベルの交差エントロピーの平均 × sw·n
                  （E78 と同じく感情の釣り合いの重みを取らないミニバッチの平均．設計 2.3 節の m101）．
            mapctl：DA の項の重み 0（同じ写像・罰則・初期値・ミニバッチの列）．
          ソースの特徴は，各ソースの全発話の平均と標準偏差で標準化した座標（段 source と同じ）に同じ写像 A をかける．
  xmuda ：xMUDA 型の相互の模倣（Jaritz ほか 2020．E78 の `run_xmuda` と同じ形）．生徒 A，V，T を同時に学ぶ．各モダリティに写像 A_m（恒等で初期化，
          μ‖A_m − I‖²）と主の頭（出発点で初期化し引き戻し．W にだけかけ，W·A にはかけない）と模倣の頭（出発点の写しで初期化，罰則なし）を置く．
          模倣の頭は，候補のミニバッチの上で，残り 2 つのモダリティの主の頭の対数の確率の和を正規化したもの（勾配を止める）との KL を学ぶ．
          KL の重みは候補 1 本あたり（κ × n × ミニバッチの平均）．3 つのモダリティの項を足した目的（S = 3·4K ＋ 3κn は記録だけ．割らない）．
  coral ：CORAL（Sun ほか 2016）．ソースごとに，ソース（そのソースの全発話の統計で標準化）と候補（候補の統計で標準化）をそれぞれ中心化して合わせた
          散らばりの上位 r 次元の主成分の部分空間の中で，ソースの共分散（Ledoit–Wolf）を候補の共分散（Ledoit–Wolf）にそろえ（白色化して色付けし
          直す．部分空間に直交する方向は変えない），そろえたソースで段 source と同じ問（W0 = 0，切片に罰則なし，そのソースの段 source の C）を
          学び直して平均し，新しい出発点にする．そこから比較の基準と同じく K 本で学び直す（引き戻しは実験 1 の値）．K = 0 は新しい出発点．
          評価の回だけ，選んだ r の新しい出発点から引き戻しの格子 14 点の曲線も解く（coral_tether）．そろえない版を学び直して段 source の識別器と
          比べた相対差を `_coral_check` の行に残す（作り方の検算）．主成分は散らばりの行列の固有ベクトル（E78 の特異値分解と同じ部分空間）．

設計で決まっていない細部（E78 の `exp/compare_dai2.py` の値をそのまま使った）：
  ミニバッチ：候補 128 本とソース 128 本（BATCH，SRC_BATCH．毎反復，復元抽出の一様な番号）．判別器の隠れ層 256（DISC_H），初期値は正規 / √(入力の次元)
  を 1 つ引いて束の全部の問に写す．Adam（torch の既定：β = (0.9, 0.999)，ε = 1e−8，重みの減衰なし），学習率は反復のあいだ一定．
  束の大きさ（1 回にまとめて学ぶ問の数）は，map で特徴の次元 1,024 以下なら 30，それより大きければ 12．xmuda は (学習率, 反復) ごとに 30 問を 1 束．
  問の並び（束の組み方を決める）は E78 と同じ：(学習率, 反復) の中で DANN (λ, μ, K) → MME (λ, μ, sw, K) → 対照 (μ, K)．
E78 から変えた点：出発点と引き戻しを E39 から実験 1 の値（(生徒, K) ごとの (C′, C_b)）に．目的は E78 と同じく S で割らない（S で割ると Adam の ε = 1e−8 が S 倍に効き，試走で ‖A − I‖_F の最大が 1.56e3 から 126 に変わった．
S は記録だけに使う）．種を 0 だけで選ぶ形から s = r に．TF32 を切った
（`torch.backends.cuda.matmul.allow_tf32 = False`，cudnn も．float32 の行列積は IEEE の float32）．融合の生徒 AVT でも DANN・MME・対照を回す．
判別器の損失は DANN の問だけに，予測のエントロピーは全部の問に残す（束の中の別の手法の問に値を入れない）．CORAL は E39 の計量 ρ_M・C′・C_b を
選び直さず（実験 1 の値に固定），候補の数が 5r に満たない r を外す E78 の規則を置かない（設計の格子は 3 点）．
GPU の非決定性を抑える `torch.use_deterministic_algorithms` は使わない．再現の度合いは設計 6 節のとおり試走で 1 単位を 2 回解いて測る．

記録（設計 1 節）：問ごとに S，束の番号・束の中の位置・束の大きさ（items と行），‖A − I‖_F，‖W·A − W0‖_F，‖W − W0‖_F，判別器の損失
（DANN）と候補のミニバッチの予測のエントロピーの最初と最後の値（更新の前の値），xMUDA 型の模倣の KL の最初と最後の値．配備する W・b は
W·A と b（候補の統計で標準化した座標）を float32 で，テストの判定を int8 で残す．ミニバッチ・最適化・TF32 の設定は provenance_log.jsonl に残す．
xMUDA 型の対照（設計 2.3 節の 2-C）は map の mapctl（A・V・T）で，8 点すべてを仮のターゲットの回と評価の回で解く．

予算を超えたときの削り方（設計 2.3 節の m72）：--cut-mme-sw01 で MME の sw = 0.1 の点を外し，--cut-iter200 で反復 200 の点を外す．
削り方は行と npz に残し，評価の回は固定のファイルに書いた削り方で回す．
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.covariance import ledoit_wolf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dai1  # noqa: E402
import dai2  # noqa: E402
import dai_data as DD  # noqa: E402
import dai_solver as S  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False                          # 設計 1 節：TF32 を切る
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

F32 = torch.float32
DEV = S.DEV
KS_ALL = dai1.KS_ALL
MAP_LR, MAP_IT, MAP_LAM, MAP_MU, MME_SW = (1e-3, 1e-2), (200, 1000), (0.01, 0.1, 1.0), (1.0, 10.0), (0.0, 0.1)
XM_KAP = (0.1, 1.0, 10.0)
DIVIDE_BY_S = False                       # 勾配法の目的は E78 と同じく S で割らない（S は記録だけ．割ると Adam の ε が S 倍に効き，E78 の格子と学び方が変わる）
BATCH, SRC_BATCH, DISC_H = 128, 128, 256
CHUNK_SMALL, CHUNK_LARGE, CHUNK_DIM = 30, 12, 1024
ADAM = {"betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.0}
SEED_BASE = {"disc": 1000, "batch": 5000, "xmuda": 2000}
CORAL_R = (32, 64, 128)
GRAD_SEEDS_EVAL = (0, 1, 2)
GRIDS = {
    "dann": [("lr", MAP_LR), ("iters", MAP_IT), ("lam", MAP_LAM), ("mu", MAP_MU)],
    "mme": [("lr", MAP_LR), ("iters", MAP_IT), ("lam", MAP_LAM), ("mu", MAP_MU), ("sw", MME_SW)],
    "mapctl": [("lr", MAP_LR), ("iters", MAP_IT), ("mu", MAP_MU)],
    "xmuda": [("lr", MAP_LR), ("iters", MAP_IT), ("kap", XM_KAP), ("mu", MAP_MU)],
    "coral": [("r", CORAL_R)],
}
DEFAULTS = {                                                           # 選択なしの値（設計 2.2・2.3 節．xMUDA 型は置かない）
    "dann": {"lr": 1e-3, "iters": 1000, "lam": 0.1, "mu": 1.0},
    "mme": {"lr": 1e-3, "iters": 1000, "lam": 0.1, "mu": 1.0, "sw": 0.0},
    "mapctl": {"lr": 1e-3, "iters": 1000, "mu": 1.0},
    "coral": {"r": 64},
}
SETTINGS = {"batch": BATCH, "src_batch": SRC_BATCH, "disc_h": DISC_H, "chunk_small": CHUNK_SMALL, "chunk_large": CHUNK_LARGE,
            "chunk_dim": CHUNK_DIM, "adam": ADAM, "lr_schedule": "constant", "grl_schedule": "lam*(2/(1+exp(-10p))-1)",
            "seed_base": SEED_BASE, "seed_rule": "s = r", "tf32": False, "deterministic_algorithms": False, "coral_r": CORAL_R,
            "coral_pca": "eigh of scatter of [source centered; candidates centered]", "coral_cov": "sklearn ledoit_wolf(assume_centered=True)"}


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coef):
        ctx.coef = coef
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.coef[:, None, None] * g, None


def grl(x, coef):
    """x（問 × 本数 × 次元）の勾配を，問ごとに −coef 倍にして返す（E78 と同じ）。"""
    return _GRL.apply(x, coef)


def ce_rows(L, y):
    return -torch.log_softmax(L, 2).gather(2, y.view(1, -1, 1).expand(L.shape[0], -1, 1)).squeeze(2)


def ent_mean(L):
    lp = torch.log_softmax(L, 2)
    return -(lp.exp() * lp).sum(2).mean(1)


def kshot(u):
    """K 本の入れ子の和集合（K=25）の位置と真のラベル，K ごとの重み（実験 1 の釣り合いの重み．使わない行は 0）。"""
    pos = u.kpos[25]
    ylab = torch.as_tensor(u.ypool[pos], device=DEV, dtype=torch.long)
    pt = torch.as_tensor(pos, device=DEV)
    wK = {0: torch.zeros(len(pos), device=DEV, dtype=F32)}
    for K in DD.KS:
        wK[K] = u.gK(K)[pt].gather(1, ylab[:, None]).squeeze(1).to(F32)
    return pt, ylab, wK


def src_feats(u, kind):
    X, y = dai2.st_sources(u, kind, Path(dai2.RUN["dai1_dir"]))
    return X.to(F32), torch.as_tensor(y, device=DEV, dtype=torch.long)


def save_lin(store, key, Weff, b, Xt):
    """配備する W·A・b（候補の統計で標準化した座標．設計 1 節）とテストの判定を残す。戻り：テストの点数（float64）。"""
    W64, b64 = Weff.to(S.DT), b.to(S.DT)
    Z = S.logits(Xt, W64, b64)
    store[f"{key}_W"], store[f"{key}_b"] = dai1.f32(W64), dai1.f32(b64)
    store[f"{key}_pred"] = dai1.pred8(Z)
    return Z


# ------------------------------------------------------------------ DANN・MME・対照

def map_problems(cut_sw: bool, cut_it: bool) -> list[tuple]:
    """(方法, 学習率, 反復, λ, μ, sw, K) の全部の問（E78 の map_problems と同じ並び．種は単位で 1 つ）。"""
    its = tuple(i for i in MAP_IT if not (cut_it and i == 200))
    sws = (0.0,) if cut_sw else MME_SW
    out = []
    for lr in MAP_LR:
        for iters in its:
            out += [("dann", lr, iters, lam, mu, 0.0, K) for lam in MAP_LAM for mu in MAP_MU for K in KS_ALL]
            out += [("mme", lr, iters, lam, mu, sw, K) for lam in MAP_LAM for mu in MAP_MU for sw in sws for K in KS_ALL]
            out += [("mapctl", lr, iters, 0.0, mu, 0.0, K) for mu in MAP_MU for K in KS_ALL]
    return out


def train_map(Z, Zl, ylab, wl, Zs, ys, W0, b0, cw, cb, meth, lam, mu, sw, lr, iters, s):
    """1 つの束を Adam で学ぶ（E78 の `_train_map` と同じ式．引き戻しは問ごとの (C′, C_b)）。戻り：(W·A, b, 記録の list)。"""
    B = len(meth)
    n, d = Z.shape
    g = torch.Generator(device=DEV).manual_seed(SEED_BASE["disc"] + s)            # 判別器の初期値
    gb = torch.Generator(device=DEV).manual_seed(SEED_BASE["batch"] + s)          # ミニバッチ（同じ s の DANN・MME・対照で同じ列）
    I = torch.eye(d, dtype=F32, device=DEV)
    A = I.unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True)
    W = W0.unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True)
    b = b0.unsqueeze(0).repeat(B, 1).clone().requires_grad_(True)
    D1 = (torch.randn(1, d, DISC_H, generator=g, device=DEV, dtype=F32) / math.sqrt(d)).repeat(B, 1, 1).requires_grad_(True)
    c1 = torch.zeros(B, 1, DISC_H, device=DEV, dtype=F32, requires_grad=True)
    D2 = (torch.randn(1, DISC_H, 1, generator=g, device=DEV, dtype=F32) / math.sqrt(DISC_H)).repeat(B, 1, 1).requires_grad_(True)
    c2 = torch.zeros(B, 1, 1, device=DEV, dtype=F32, requires_grad=True)
    t = lambda v: torch.tensor(v, dtype=F32, device=DEV)  # noqa: E731
    lam, mu, sw, cw, cb = t(lam), t(mu), t(sw), t(cw), t(cb)
    is_dann = torch.tensor([m == "dann" for m in meth], device=DEV)
    is_mme = torch.tensor([m == "mme" for m in meth], device=DEV)
    zero = torch.zeros(B, dtype=F32, device=DEV)
    Sv = wl.sum(1) + torch.where(is_dann | is_mme, lam * n, zero) + torch.where(is_mme, sw * n, zero)   # 設計 1 節の S（K 本の重み ＋ 項の重み × n）
    Sd = Sv.clamp(min=1e-12)
    opt = torch.optim.Adam([A, W, b, D1, c1, D2, c2], lr=lr, **ADAM)
    tgt_dom = torch.cat([torch.ones(SRC_BATCH), torch.zeros(BATCH)]).to(DEV).view(1, -1, 1).expand(B, -1, 1)
    rec = {}
    nan = torch.full((B,), float("nan"), device=DEV)
    for step in range(iters):
        p = step / max(iters - 1, 1)
        sched = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        it = torch.randint(0, n, (BATCH,), generator=gb, device=DEV)
        is_ = torch.randint(0, len(Zs), (SRC_BATCH,), generator=gb, device=DEV)
        opt.zero_grad()
        loss = torch.zeros(B, dtype=F32, device=DEV)
        if len(Zl):
            Ll = torch.einsum("bne,bke->bnk", torch.einsum("nd,bed->bne", Zl, A), W) + b[:, None]
            loss = loss + (wl * ce_rows(Ll, ylab)).sum(1)
        loss = loss + ((W - W0) ** 2).sum((1, 2)) / (2 * cw) + ((b - b0) ** 2).sum(1) / (2 * cb) + ((A - I) ** 2).sum((1, 2)) * mu
        Ht = torch.einsum("nd,bed->bne", Z[it], A)
        ld = None
        if is_dann.any():
            Hs = torch.einsum("nd,bed->bne", Zs[is_], A)
            Hcat = grl(torch.cat([Hs, Ht], 1), lam * sched)
            o = torch.einsum("bnh,bho->bno", torch.relu(torch.einsum("bnd,bdh->bnh", Hcat, D1) + c1), D2) + c2
            ld = F.binary_cross_entropy_with_logits(o, tgt_dom, reduction="none").mean((1, 2))
            loss = loss + torch.where(is_dann, ld * n, torch.zeros_like(ld))
        if is_mme.any():
            Hm = ent_mean(torch.einsum("bne,bke->bnk", grl(Ht, torch.ones(B, device=DEV, dtype=F32)), W) + b[:, None])
            Ls = torch.einsum("bne,bke->bnk", torch.einsum("nd,bed->bne", Zs[is_], A), W) + b[:, None]
            cs = ce_rows(Ls, ys[is_]).mean(1)
            loss = loss + torch.where(is_mme, -lam * n * Hm + sw * n * cs, torch.zeros_like(Hm))
        if step in (0, iters - 1):                                               # DA の項が働いたかを測る量（最初と最後．更新の前の値）
            with torch.no_grad():
                ent = ent_mean(torch.einsum("bne,bke->bnk", Ht, W) + b[:, None])
                rec[step] = {"ent": ent.clone(), "dom": torch.where(is_dann, ld, nan) if ld is not None else nan.clone()}
        (loss / Sd if DIVIDE_BY_S else loss).sum().backward()                    # E78 と同じく S で割らない（設計 1 節．Adam の ε の効き方を E78 の格子に合わせる）
        opt.step()
        if not torch.isfinite(loss).all():
            raise FloatingPointError("dann/mme/mapctl の損失が NaN/Inf")
    with torch.no_grad():
        Weff = torch.einsum("bke,bed->bkd", W, A)
        an = ((A - I) ** 2).sum((1, 2)).sqrt()
        wad = ((Weff - W0) ** 2).sum((1, 2)).sqrt()
        wd = ((W - W0) ** 2).sum((1, 2)).sqrt()
        fin = torch.isfinite(Weff).all(2).all(1) & torch.isfinite(b).all(1)
        r0, r1 = rec[0], rec[iters - 1]
        diag = [dict(S=float(Sv[j]), anorm=float(an[j]), wa_dist=float(wad[j]), w_dist=float(wd[j]), ent_first=float(r0["ent"][j]), ent_last=float(r1["ent"][j]),
                     dom_first=float(r0["dom"][j]), dom_last=float(r1["dom"][j]), finite=bool(fin[j])) for j in range(B)]
        return Weff.detach().clone(), b.detach().clone(), diag


def pass_map(u, store, rows, tether, s_init, cut_sw, cut_it):
    probs_all = map_problems(cut_sw, cut_it)
    pt, ylab, wK = kshot(u)
    for s in DD.STUDENTS:
        Z, Zt = u.Xp[s].to(F32), u.Xt[s]
        Zl = Z[pt]
        Zs, ys = src_feats(u, s)
        W0, b0 = (x.to(F32) for x in u.start[s])
        d = Z.shape[1]
        step = CHUNK_SMALL if d <= CHUNK_DIM else CHUNK_LARGE
        groups = {}
        for j, c in enumerate(probs_all):
            groups.setdefault((c[1], c[2]), []).append(j)
        Ws, bs, items, rws, chunks = [None] * len(probs_all), [None] * len(probs_all), [None] * len(probs_all), [None] * len(probs_all), []
        for (lr, iters), idx in groups.items():
            for a in range(0, len(idx), step):
                ch = idx[a:a + step]
                cid = len(chunks)
                chunks.append(ch)
                t0 = time.time()
                cs = [probs_all[j] for j in ch]
                tw = [tether[(s, max(c[6], 1))] for c in cs]
                Weff, bb, dg = train_map(Z, Zl, ylab, torch.stack([wK[c[6]] for c in cs]), Zs, ys, W0, b0, [x[0] for x in tw], [x[1] for x in tw],
                                         [c[0] for c in cs], [c[3] for c in cs], [c[4] for c in cs], [c[5] for c in cs], lr, iters, s_init)
                sec = time.time() - t0
                for q, (j, c) in enumerate(zip(ch, cs)):
                    meth, lr_, it_, lam, mu, sw, K = c
                    it = dict(method=meth, lr=lr_, iters=it_, mu=mu, K=K, seed_init=s_init, chunk=cid, chunk_pos=q, chunk_size=len(ch))
                    if meth != "mapctl":
                        it["lam"] = lam
                    if meth == "mme":
                        it["sw"] = sw
                    Ws[j], bs[j], items[j] = Weff[q], bb[q], it
                    rws[j] = dict(student=s, Cw=tw[q][0], Cb=tw[q][1], sec_chunk=sec, **it, **dg[q])
                del Weff, bb
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        Zte = save_lin(store, f"map_{s}", torch.stack(Ws), torch.stack(bs), Zt)
        ua = S.uar_batch(u.ytest, Zte)
        for j in range(len(probs_all)):
            rws[j]["uar"] = float(ua[j])
        rows.extend(rws)
        store[f"map_{s}_items"] = np.array([json.dumps(it, sort_keys=True) for it in items])
        store[f"map_{s}_chunks"] = np.array(json.dumps(chunks))
        del Zs, Z, Zl
    store["cut"] = np.array(json.dumps({"mme_sw01": cut_sw, "iter200": cut_it}))


# ------------------------------------------------------------------ xMUDA 型

def xm_problems(cut_it: bool) -> list[tuple]:
    its = tuple(i for i in MAP_IT if not (cut_it and i == 200))
    return [(lr, iters, kap, mu, K) for lr in MAP_LR for iters in its for kap in XM_KAP for mu in MAP_MU for K in KS_ALL]


def train_xmuda(Zm, Zlm, ylab, wl, start, cw, cb, kap, mu, lr, iters, s, n):
    """1 つの束（(学習率, 反復) の 30 問）を Adam で学ぶ（E78 の `run_xmuda` と同じ式）。cw・cb：{m: 問ごとの値}。"""
    mods = "AVT"
    B = len(kap)
    g = torch.Generator(device=DEV).manual_seed(SEED_BASE["xmuda"] + s)
    t = lambda v: torch.tensor(v, dtype=F32, device=DEV)  # noqa: E731
    kap, mu = t(kap), t(mu)
    P = {}
    for m in mods:
        d = Zm[m].shape[1]
        W0, b0 = (x.to(F32) for x in start[m])
        P[m] = {"A": torch.eye(d, device=DEV, dtype=F32).unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True),
                "W": W0.unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True), "b": b0.unsqueeze(0).repeat(B, 1).clone().requires_grad_(True),
                "V": W0.unsqueeze(0).repeat(B, 1, 1).clone().requires_grad_(True), "c": b0.unsqueeze(0).repeat(B, 1).clone().requires_grad_(True),
                "W0": W0, "b0": b0, "I": torch.eye(d, device=DEV, dtype=F32), "cw": t(cw[m]), "cb": t(cb[m])}
    opt = torch.optim.Adam([P[m][k] for m in mods for k in ("A", "W", "b", "V", "c")], lr=lr, **ADAM)
    Sv = 3 * wl.sum(1) + 3 * kap * n                                             # 設計 1 節の S（3 モダリティの K 本の重み ＋ KL の重み × n）
    Sd = Sv.clamp(min=1e-12)
    rec = {}
    for step in range(iters):
        it = torch.randint(0, n, (BATCH,), generator=g, device=DEV)
        opt.zero_grad()
        loss = torch.zeros(B, dtype=F32, device=DEV)
        main_lp, mim_lp = {}, {}
        for m in mods:
            q = P[m]
            H = torch.einsum("nd,bed->bne", Zm[m][it], q["A"])
            main_lp[m] = torch.log_softmax(torch.einsum("bne,bke->bnk", H, q["W"]) + q["b"][:, None], 2)
            mim_lp[m] = torch.log_softmax(torch.einsum("bne,bke->bnk", H, q["V"]) + q["c"][:, None], 2)
            if len(Zlm[m]):
                Ll = torch.einsum("bne,bke->bnk", torch.einsum("nd,bed->bne", Zlm[m], q["A"]), q["W"]) + q["b"][:, None]
                loss = loss + (wl * ce_rows(Ll, ylab)).sum(1)
            loss = loss + ((q["W"] - q["W0"]) ** 2).sum((1, 2)) / (2 * q["cw"]) + ((q["b"] - q["b0"]) ** 2).sum(1) / (2 * q["cb"]) \
                + ((q["A"] - q["I"]) ** 2).sum((1, 2)) * mu
        for m in mods:
            tgt = torch.log_softmax(sum(main_lp[o].detach() for o in mods if o != m), 2)
            kl = (tgt.exp() * (tgt - mim_lp[m])).sum(2).mean(1)
            loss = loss + kap * n * kl
            if step in (0, iters - 1):                                           # 模倣の項が働いたかを測る量（最初と最後）
                with torch.no_grad():
                    rec.setdefault(m, {})[step] = (kl.detach().clone(), -(main_lp[m].exp() * main_lp[m]).sum(2).mean(1).detach().clone())
        (loss / Sd if DIVIDE_BY_S else loss).sum().backward()
        opt.step()
        if not torch.isfinite(loss).all():
            raise FloatingPointError("xmuda の損失が NaN/Inf")
    out = {}
    with torch.no_grad():
        for m in mods:
            q = P[m]
            Weff = torch.einsum("bke,bed->bkd", q["W"], q["A"])
            an = ((q["A"] - q["I"]) ** 2).sum((1, 2)).sqrt()
            wad = ((Weff - q["W0"]) ** 2).sum((1, 2)).sqrt()
            wd = ((q["W"] - q["W0"]) ** 2).sum((1, 2)).sqrt()
            fin = torch.isfinite(Weff).all(2).all(1) & torch.isfinite(q["b"]).all(1)
            (k0, e0), (k1, e1) = rec[m][0], rec[m][iters - 1]
            dg = [dict(S=float(Sv[j]), anorm=float(an[j]), wa_dist=float(wad[j]), w_dist=float(wd[j]), kl_first=float(k0[j]), kl_last=float(k1[j]),
                       ent_first=float(e0[j]), ent_last=float(e1[j]), finite=bool(fin[j])) for j in range(B)]
            out[m] = (Weff.clone(), q["b"].detach().clone(), dg)
    del P, opt
    return out


def pass_xmuda(u, store, rows, tether, s_init, cut_it):
    mods = "AVT"
    probs_all = xm_problems(cut_it)
    pt, ylab, wK = kshot(u)
    Zm = {m: u.Xp[m].to(F32) for m in mods}
    Zlm = {m: Zm[m][pt] for m in mods}
    groups = {}
    for j, c in enumerate(probs_all):
        groups.setdefault((c[0], c[1]), []).append(j)
    res = {m: ([None] * len(probs_all), [None] * len(probs_all)) for m in mods}
    items, chunks, rws = [None] * len(probs_all), [], [[] for _ in probs_all]
    for (lr, iters), ch in groups.items():
        cid = len(chunks)
        chunks.append(ch)
        t0 = time.time()
        cs = [probs_all[j] for j in ch]
        cw = {m: [tether[(m, max(c[4], 1))][0] for c in cs] for m in mods}
        cb = {m: [tether[(m, max(c[4], 1))][1] for c in cs] for m in mods}
        o = train_xmuda(Zm, Zlm, ylab, torch.stack([wK[c[4]] for c in cs]), u.start, cw, cb, [c[2] for c in cs], [c[3] for c in cs],
                        lr, iters, s_init, u.n)
        sec = time.time() - t0
        for q, (j, c) in enumerate(zip(ch, cs)):
            it = dict(method="xmuda", lr=c[0], iters=c[1], kap=c[2], mu=c[3], K=c[4], seed_init=s_init, chunk=cid, chunk_pos=q, chunk_size=len(ch))
            items[j] = it
            for m in mods:
                res[m][0][j], res[m][1][j] = o[m][0][q], o[m][1][q]
                rws[j].append(dict(student=m, Cw=cw[m][q], Cb=cb[m][q], sec_chunk=sec, **it, **o[m][2][q]))
        del o
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for m in mods:
        Zte = save_lin(store, f"xm_{m}", torch.stack(res[m][0]), torch.stack(res[m][1]), u.Xt[m])
        ua = S.uar_batch(u.ytest, Zte)
        for j in range(len(probs_all)):
            r_ = next(x for x in rws[j] if x["student"] == m)
            r_["uar"] = float(ua[j])
    for lst in rws:
        rows.extend(lst)
    store["xm_items"] = np.array([json.dumps(it, sort_keys=True) for it in items])
    store["xm_chunks"] = np.array(json.dumps(chunks))
    store["cut"] = np.array(json.dumps({"iter200": cut_it}))


# ------------------------------------------------------------------ CORAL

CORAL_CACHE: dict = {}                                                  # (T, c, 分割, 種類) → 新しい出発点（種に依らないので種の間で使い回す）
CHECK_DONE: dict = {}                                                   # (ソース, 種類) → そろえない版の検算の相対差


def lw_cov(P):
    Sx, shrink = ledoit_wolf(P.cpu().numpy().astype(np.float64), assume_centered=True)
    return torch.as_tensor(Sx, dtype=S.DT, device=DEV), float(shrink)


def msqrt(Sm, inv=False):
    lam, V = torch.linalg.eigh(Sm)
    lam = lam.clamp(min=1e-10)
    return (V * (lam.rsqrt() if inv else lam.sqrt())) @ V.T


def coral_starts(u, kind):
    """ソースごとに r ∈ CORAL_R でそろえた特徴で段 source の問を学び直し，平均した新しい出発点。戻り：{r: (W, b)}。"""
    key = (u.T, u.c, u.f, kind)
    if key in CORAL_CACHE:
        return CORAL_CACHE[key]
    dai1_dir = Path(dai2.RUN["dai1_dir"])
    Xt_c = u.Xp[kind] - u.Xp[kind].mean(0)
    Wsum = {r: 0 for r in CORAL_R}
    bsum = {r: 0 for r in CORAL_R}
    info = []
    for src in u.srcs:
        cp = dai1.Corpus.get(src)
        mu, sd = dai1.load_source_stats(dai1_dir, src, kind)
        Xs = (cp.X[kind] - mu) / sd
        Cs_src = float(np.load(dai1_dir / "source" / f"{src}_{kind}.npz")["C"])
        G = S.balanced_onehot(cp.y, np.arange(len(cp.y)), len(cp.y))[None]
        d = Xs.shape[1]
        if (src, kind) not in CHECK_DONE:                                      # そろえない版は段 source の識別器と同じになるはず（作り方の検算）
            r0 = S.fit(Xs, G, torch.zeros(4, d), torch.zeros(4), Cw=Cs_src, Cb=math.inf)
            W_ref, _ = dai1.load_source(dai1_dir, src, kind)
            CHECK_DONE[(src, kind)] = float((r0["W"][0] - W_ref).abs().max() / W_ref.abs().max().clamp(min=1e-30))
        info.append(dict(method="_coral_check", student=kind, src=src, rel_diff=CHECK_DONE[(src, kind)]))
        Xs_c = Xs - Xs.mean(0)
        both = torch.cat([Xs_c, Xt_c])
        both = both - both.mean(0)
        ev, V = torch.linalg.eigh(both.T @ both)
        V = V[:, torch.argsort(ev, descending=True)[:max(CORAL_R)]]
        del both
        for r in CORAL_R:
            U = V[:, :r]
            Ps, Pt = Xs_c @ U, Xt_c @ U
            Cs, sh_s = lw_cov(Ps - Ps.mean(0))
            Ct, sh_t = lw_cov(Pt - Pt.mean(0))
            Pn = (Ps - Ps.mean(0)) @ msqrt(Cs, inv=True) @ msqrt(Ct) + Ps.mean(0)
            X2 = Xs + (Pn - Ps) @ U.T
            rs = S.fit(X2, G, torch.zeros(4, d), torch.zeros(4), Cw=Cs_src, Cb=math.inf)
            Wsum[r] = Wsum[r] + rs["W"][0]
            bsum[r] = bsum[r] + rs["b"][0]
            info.append(dict(method="_coral_src", student=kind, src=src, r=r, C_src=Cs_src, grad=float(rs["grad_p"][0]),
                             stalled=bool(rs["stalled_p"][0]), finite=bool(rs["finite_p"][0]), W_norm=float(rs["W"][0].norm()),
                             shift_rel=float((X2 - Xs).norm() / Xs.norm()), shrink_src=sh_s, shrink_tgt=sh_t))
            del X2, Pn, Ps, Pt
        del Xs, Xs_c, V
    out = {r: (Wsum[r] / len(u.srcs), bsum[r] / len(u.srcs)) for r in CORAL_R}
    for k in [k for k in CORAL_CACHE if k[:3] != key[:3]]:                 # 別の (T, c, 分割) の作り置きは捨てる
        del CORAL_CACHE[k]
    CORAL_CACHE[key] = (out, info)
    return CORAL_CACHE[key]


def pass_coral(u, store, rows, tether, fsel=None):
    """fsel：評価の回だけ，{(生徒, K): 選んだ r}．その r の新しい出発点から引き戻しの格子 14 点の曲線を解く。"""
    for s in DD.STUDENTS:
        starts, info = coral_starts(u, s)
        rows.extend(dict(x) for x in info)
        X, Xt = u.Xp[s], u.Xt[s]
        store[f"coral_{s}_start_W"] = dai1.f32(torch.stack([starts[r][0] for r in CORAL_R]))
        store[f"coral_{s}_start_b"] = dai1.f32(torch.stack([starts[r][1] for r in CORAL_R]))
        Z0 = torch.stack([S.logits(Xt, *starts[r]) for r in CORAL_R])
        store[f"coral_{s}_start_pred"] = dai1.pred8(Z0)                     # K = 0（並びは CORAL_R）
        for r, v in zip(CORAL_R, S.uar_batch(u.ytest, Z0)):
            rows.append(dict(method="coral", student=s, K=0, r=r, uar=float(v), S=0.0, grad=0.0, stalled=False, finite=True, n_iter=0))
        specs, W0s, b0s, cws, cbs = [], [], [], [], []
        for r in CORAL_R:
            for K in DD.KS:
                cw, cb = tether[(s, K)]
                specs.append((dict(method="coral", r=r), u.gK(K), K)); W0s.append(starts[r][0]); b0s.append(starts[r][1]); cws.append(cw); cbs.append(cb)
        if fsel is not None:
            for K in DD.KS:
                r = int(fsel[(s, K)])
                for cw, cb in dai1.TETHER:
                    specs.append((dict(method="coral_tether", r=r), u.gK(K), K)); W0s.append(starts[r][0]); b0s.append(starts[r][1])
                    cws.append(cw); cbs.append(cb)
        G = torch.stack([g for _, g, _ in specs])
        res = S.fit(X, G, torch.stack(W0s), torch.stack(b0s), Cw=cws, Cb=cbs)
        Z = S.logits(Xt, res["W"], res["b"])
        ua = S.uar_batch(u.ytest, Z)
        for j, (it, g, K) in enumerate(specs):
            rows.append(dict(student=s, K=K, uar=float(ua[j]), S=float(g.sum()), grad=float(res["grad_p"][j]), stalled=bool(res["stalled_p"][j]),
                             finite=bool(res["finite_p"][j]), n_iter=int(res["n_iter_p"][j]), Cw=cws[j], Cb=cbs[j], **it))
        store[f"coral_{s}_W"], store[f"coral_{s}_b"] = dai1.f32(res["W"]), dai1.f32(res["b"])
        store[f"coral_{s}_pred"] = dai1.pred8(Z)
        store[f"coral_{s}_items"] = np.array([json.dumps(dict(it, K=K, Cw=cws[j], Cb=cbs[j]), sort_keys=True) for j, (it, _, K) in enumerate(specs)])


# ------------------------------------------------------------------ 進め方

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("pseudo", "eval"))
    dai2.common_args(ap)
    ap.add_argument("--parts", nargs="*", default=["map", "xmuda", "coral"], choices=("map", "xmuda", "coral"))
    ap.add_argument("--cut-mme-sw01", action="store_true", help="設計 2.3 節の削り方の 1 段目（MME の sw = 0.1 を外す）")
    ap.add_argument("--cut-iter200", action="store_true", help="同 2 段目（反復 200 を外す）")
    args = ap.parse_args()
    out = dai2.setup_run(args, "dai2_grad.py")
    R = dai2.RUN
    if args.stage == "pseudo":
        cut_sw, cut_it = args.cut_mme_sw01, args.cut_iter200
        for T in args.targets:
            teth, tpath = dai2.tether_pseudo(Path(args.dai1_dir), T, args.allow_partial_tether)
            R["tether_key"] = dai2.tether_key(teth)
            dai2.provenance(out, args, "DAI/exp/dai2_grad.py", dict(stage="pseudo", T=T, tether_file=str(tpath), tether_file_sha256=dai2.sha(tpath),
                                                                  tether_key=R["tether_key"], parts=args.parts, cut_mme_sw01=cut_sw,
                                                                  cut_iter200=cut_it, settings=SETTINGS))
            seeds = args.seeds if args.seeds is not None else [0, 1]
            for c, srcs, f, r in dai2.units_of(args, T, "pseudo", seeds):
                run_parts(out, "pseudo", T, c, srcs, f, r, args.parts, teth, cut_sw, cut_it, None)
        return
    fz2, fz1 = dai2.check_eval_frozen(out, Path(args.dai1_frozen))
    cut = fz2["_meta"]["grad_cut"]
    if (args.cut_mme_sw01, args.cut_iter200) != (False, False) and (args.cut_mme_sw01, args.cut_iter200) != (cut["mme_sw01"], cut["iter200"]):
        sys.exit("評価の回の削り方は固定のファイルの値を使う（引数と食い違う）")
    for T in args.targets:
        teth = dai2.parse_tether(fz1[T]["tether"])
        R["tether_key"] = dai2.tether_key(teth)
        Rs = int(fz1[T]["n_seeds_eval"])
        fsel = {(s, K): fz2[T]["select"][f"{s}|{K}|coral"]["r"] for s in DD.STUDENTS for K in DD.KS}
        dai2.provenance(out, args, "DAI/exp/dai2_grad.py", dict(stage="eval", T=T, frozen_sha256=R["frozen_sha256"], tether_key=R["tether_key"],
                                                              parts=args.parts, grad_cut=cut, settings=SETTINGS, R=Rs))
        grad_parts = [p for p in args.parts if p != "coral"]
        if grad_parts:
            for c, srcs, f, r in dai2.units_of(args, T, "eval", args.seeds if args.seeds is not None else list(GRAD_SEEDS_EVAL)):
                run_parts(out, "eval", T, c, srcs, f, r, grad_parts, teth, cut["mme_sw01"], cut["iter200"], None)
        if "coral" in args.parts:
            for c, srcs, f, r in dai2.units_of(args, T, "eval", args.seeds if args.seeds is not None else list(range(Rs))):
                run_parts(out, "eval", T, c, srcs, f, r, ["coral"], teth, False, False, fsel)


def run_parts(out, tag, T, c, srcs, f, r, parts, teth, cut_sw, cut_it, fsel):
    s_init = r                                                              # 初期化の種 s = r（設計 1 節）
    for part in parts:
        if part == "map":
            dai2.run_unit(out, tag, T, c, srcs, f, r, "map", lambda u, st, rw: pass_map(u, st, rw, teth, s_init, cut_sw, cut_it), teth,
                          extra_tag=json.dumps({"mme_sw01": cut_sw, "iter200": cut_it}))
        elif part == "xmuda":
            dai2.run_unit(out, tag, T, c, srcs, f, r, "xmuda", lambda u, st, rw: pass_xmuda(u, st, rw, teth, s_init, cut_it), teth,
                          extra_tag=json.dumps({"iter200": cut_it}))
        else:
            dai2.run_unit(out, tag, T, c, srcs, f, r, "coral", lambda u, st, rw: pass_coral(u, st, rw, teth, fsel), teth,
                          extra_tag=json.dumps({"curve": fsel is not None}))


if __name__ == "__main__":
    main()
