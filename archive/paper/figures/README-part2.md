# 全体・手法・Part II（5 節）・比較（6 節）の図

8 枚の図の台本，読んだ結果ファイル，キャプションの案，図から読み取れることを図ごとに書く．
データの図は `paper/scripts/figures/fig_<名前>.py` が作り，`style.save()` で `figures/<名前>.pdf`（本文用）と `figures/<名前>.png`（確認用，200 dpi）に書く．
模式図は `figures/src/<名前>.tex`（TikZ の standalone 文書，Helvetica，色は `style.py` の SLOT と同じ 16 進）を `bash figures/src/build.sh` でコンパイルし，`figures/<名前>.pdf` と `.png` に書く（中間生成物は一時フォルダに置く）．

```bash
cd /home/snics/MER-DA/paper
for f in capture ladder compare kcurves entmin finetune; do ../.venv/bin/python scripts/figures/fig_$f.py; done
bash figures/src/build.sh            # overview と method
```

比較の図（compare・kcurves・entmin）と ladder の一部は，台帳 `scripts/ledger/comparison.py` の `_world()` をそのまま呼んで値を作る（台帳と同じ選び方・同じ平均の取り方）．`_world()` は読むだけで `.cache/` には書かない．finetune は同じモジュールの `_ft_runs()` と定数を使う．
図の大きさは段 1 つ 3.41〜3.44 in，2 段ぶち抜き 7.06〜7.12 in で，縮めずに貼れる（字は 6.5 pt 以上．模式図は 7 pt 以上）．

## 色の役割（全図で共通）

| 役割 | 色 |
|---|---|
| ソースのコーパスと出発点（E39，five-value start） | 灰 `#8a8985` |
| ターゲットのラベル，生徒，ours | 青 `#2a78d6`（SLOT[0]） |
| 先生と弱いラベル，ours-ext（oursx） | 紫 `#6250d6`（SLOT[6]） |
| 最も強い相手の調整版 | 橙 `#eb6834`（SLOT[1]） |
| エントロピー最小化の固定版 | 赤 `#e34948`（SLOT[7]） |

---

## 1. overview（Fig. 1，模式図，2 段ぶち抜き 7.11 × 2.39 in）

- ファイル：`figures/overview.pdf`，`figures/overview.png`
- 台本：`figures/src/overview.tex`（`bash figures/src/build.sh overview`）
- 読んだ結果ファイル：なし（数値を描かない．式 1/(2K) と q/K だけを書く）
- キャプションの案：
  > Overview. Left: a linear classifier trained on all labeled source corpora with frozen encoders is the start. Middle (Part I): for single-modality audio and face features with $K \le 5$, most of the gain from $K$ target labels per emotion comes from the thresholds, whose error decreases as $1/(2K)$, whereas the error of the direction decreases as $q/K$. Right (Part II): a teacher trained on the source corpora alone in other modalities (here the fused A+V+T classifier) labels every unlabeled target utterance; after its outputs are rescaled to equal mass per emotion, these weak labels train a linear student anchored to its own start by $\ell_2$ penalties, and the student needs only its own modality at test time.
- 読み取れること：左から右へ，ソースで作った出発点，K 本のラベルが判断基準（切片）を先に直し向きを直しにくいこと，別のモダリティの先生の弱いラベルで向きを補うこと，の 3 段が並ぶ．中央の図では，灰の点線（出発点）を青の実線（K 本のあと）が平行に動かすが，黒の点線（ターゲットの最良）の向きには回り切らない．

## 2. method（Fig. 2，模式図，2 段ぶち抜き 7.12 × 2.81 in）

- ファイル：`figures/method.pdf`，`figures/method.png`
- 台本：`figures/src/method.tex`（`bash figures/src/build.sh method`）
- 読んだ結果ファイル：なし
- キャプションの案：
  > The method. Each corpus is centered with its own mean, without labels. The start $(W_0, b_0)$ is hypothesis transfer from all sources with five values ($\gamma$ and $\rho_s$ form the source draft, $\rho_M$ is the penalty metric, $C'$ and $C_b$ are the weight and intercept penalties). A teacher trained on the sources alone predicts every unlabeled target utterance; its outputs, rescaled to equal mass per emotion with total weight $m$, are weak labels that the student fits together with the $4K$ labels while anchored to the start. The extended version (ours-ext) adds the fused student refit on the $K$ labels as a teacher and unbalanced or hard (above $\tau$) weights. All values are selected on pseudo-targets: the five values once, averaged over $K$, and the teacher, the weighting and $m$ for each $K$, with no weak labels as a candidate.
- 読み取れること：上の段（ソース）で出発点と先生を作り，下の段（ターゲット）で先生の出力を弱いラベルにし，K 本のラベルと一緒に出発点に錨づけた生徒が学ぶ流れである．仮のターゲットで選ぶ値のうち，5 つの値は K によらず 1 度だけ選び，先生・重みの付け方・総重み m は K ごとに選ぶことを下の破線の枠に書いた．

## 3. capture（5 節，段 1 つ 3.44 × 2.21 in）

- ファイル：`figures/capture.pdf`，`figures/capture.png`
- 台本：`scripts/figures/fig_capture.py`
- 読んだ結果ファイル：`results/explore_relation/snapshots/20261021-2245_e58/e58_targets.csv`（1 行 = 種類 × ターゲット，反復 0〜9 の平均．A 7，V 6，T 4 ターゲット）
- 描いたもの：点は「ターゲットの平均の r(S)」÷「ターゲットの平均の r(random_true)」（台帳 `p2-e58-ratio-*` と同じ計算）．横線はターゲットごとの比 r(S)/r(random_true) の最小から最大
- キャプションの案：
  > Label-free selection does not find the direction error. Share of the start's direction error $E_0$ that lies in the span of the $K=5$ labels per emotion (20 utterances), relative to a random draw, for seven label-free selection rules and for teaching with the labels of all candidates (upper reference). Markers: ratio of the means over targets (A: 7, V: 6, T: 4 targets; iterations 0–9); lines: range of the per-target ratios; dashed line: random draw.
- 読み取れること：ラベル無しの 7 通りの選び方は，どれもランダムの 0.5〜1.2 倍の範囲にあり，白色化した座標で遠い発話を選ぶ 2 通りはランダムより下がる．候補のラベルを使う教える選び方でも 1.4〜2.0 倍にとどまる．

## 4. ladder（5 節，段 1 つ 3.41 × 3.31 in）

- ファイル：`figures/ladder.pdf`，`figures/ladder.png`
- 台本：`scripts/figures/fig_ladder.py`
- 読んだ結果ファイル：
  - (a) `results/explore_relation/snapshots/20261017-1654_e31final/e31_finals30.csv`（B1 ターゲットのみ，B4 合併．同じコーパスの組を除き，組の値をソースで平均してターゲットの値にした），`results/explore_relation/snapshots/20261018-0213_ext30/ext30_targets.csv`（span = all の P4 と E39）．音声 WavLM，7 ターゲット，反復 0〜29
  - (b) `results/explore_relation/snapshots/20261022-0055_e59k/e59k_targets.csv`（meta − e39．音声だけの世界，7 ターゲット），`results/explore_relation/snapshots/20261022-0810_e60/e60_targets.csv`（meta − e39．AV の世界 6 ターゲット，AVT の世界 4 ターゲットの音声の生徒），oursx は `results/baselines_fuse-audio-video-text/<T>.csv`（台帳 comparison.py と同じ）．反復 0〜9
- 描き方の判断：(b) の 4 本はターゲットの集合も世界（使う発話）も違い，E39 の水準が違う（K=0 の音声の E39 は，音声だけの世界の 4 ターゲットで 50.2，AVT の世界の同じ 4 ターゲットで 47.4）．そのため (b) は各実験の中の E39 との差にし，線ごとにターゲットの数を書いた．(a) と (b) を同じ縦軸に載せていない
- キャプションの案：
  > The method step by step for the audio (WavLM) student. (a) UAR of target-only training, pooling with one source (averaged over sources), hypothesis transfer from all sources with default values (HT), and the start with five values selected on pseudo-targets; mean over seven targets and 30 iterations. (b) Gain over the start of the same experiment from weak labels: a teacher of the same modality (audio-only configuration), the fused teacher in the AV and AVT configurations, and the extended form in AVT (10 iterations; the number of targets is in parentheses). The curves in (b) use different targets and therefore compare gains, not levels.
- 読み取れること：(a) では，ターゲットのみ → 合併 → 仮説転移 → 5 つの値の出発点の順に UAR が上がり，K が小さいほど段の差が大きい．(b) では，同じモダリティの先生の得（K=0 で +1.4）より別のモダリティの先生の得（+2.9〜+3.4）が大きく，どちらも K とともに減るが，拡張版は K=25 でも +3.1 を保つ．

## 5. compare（6 節，2 段ぶち抜き 7.06 × 3.22 in）

- ファイル：`figures/compare.pdf`，`figures/compare.png`
- 台本：`scripts/figures/fig_compare.py`
- 読んだ結果ファイル：`results/baselines_<世界>/<T>.csv` と `<T>_pseudo.csv`（調整版の選択を作り直す），`results/multimodal_distill_<世界>/<T>.csv` と `<T>_pseudo.csv`（ours の選択）．世界は `fuse-audio-video-text`，`fuse-audio-video`，`fuse-video-text`，`fuse-audio-text`，`fuse2-audio-video-text`．反復 0〜9，仮のターゲットの反復 0〜4．作り直した選択は保存の chosen と全件一致した（AVT で 780/780，E60 80/80 など）
- 描いたもの：棒はターゲットの平均の E39 との差（台帳 `cmp-best-vs-e39-*`，`cmp-ours-vs-e39-*`，`cmp-oursx-vs-e39-*` と同じ値）．**ひげはターゲットの間の範囲（最小から最大）で，95% 区間ではない**．棒の下の橙の略称が，その生徒・K で最も強い相手（略称は本文の表 tab:compare に合わせた．共訓練の 3 つの変種は CoT，共正則化の 2 つは CoR にまとめた）
- キャプションの案：
  > Differences from the five-value start at $K=0$ (a) and $K=5$ (b) for the strongest tuned competitor, ours, and ours-ext, per configuration and student. Bars: mean over targets (four; six for AV) of the per-target mean over ten iterations; whiskers: range over targets. The abbreviation below each group names the strongest competitor (GD, generalized distillation; KD, distillation with a temperature; HPL, hard pseudo-labels; RB, output rebalancing; ST, self-training; CoT, co-training); "none" means that every tuned competitor fell back to the start.
- 読み取れること：22 の（生徒, K）のすべてで，紫（ours-ext）の棒が橙（最も強い相手）の棒より高い．ただしひげは長く，ours-ext が出発点を下回るターゲットが 22 のうち 10 の（生徒, K）で 1 つずつある（最大の損は第 2 エンコーダの顔の K=0 の IEMOCAP 即興で −1.9）．

## 6. kcurves（6 節，段 1 つ 3.41 × 2.75 in）

- ファイル：`figures/kcurves.pdf`，`figures/kcurves.png`
- 台本：`scripts/figures/fig_kcurves.py`
- 読んだ結果ファイル：`results/baselines_fuse-audio-video-text/<T>.csv`，`<T>_pseudo.csv`，`results/multimodal_distill_fuse-audio-video-text/<T>.csv`，`<T>_pseudo.csv`（4 ターゲット，反復 0〜9）
- 描いたもの：最も強い相手は K ごとに選び直し，その略称を各枠の下の段に並べた（compare と同じ略称）
- キャプションの案：
  > Differences from the five-value start against $K$ for the four students of the AVT configuration (first encoders, four targets, ten iterations): ours, ours-ext, and the strongest tuned competitor, which is chosen anew for each $K$ (abbreviations at the bottom of each panel, as in Fig.~\ref{fig:compare}).
- 読み取れること：ours（青）の得は K とともに減り，音声と融合では K=25 で 0 以下になるが，ours-ext（紫）は全部の K で +0.7 以上を保つ．言葉の生徒では K=10・25 で一般化蒸留（GD）が ours-ext をわずかに上回る．

## 7. entmin（6 節，段 1 つ 3.41 × 2.31 in）

- ファイル：`figures/entmin.pdf`，`figures/entmin.png`
- 台本：`scripts/figures/fig_entmin.py`
- 読んだ結果ファイル：`results/baselines_<世界>/<T>.csv`（世界は AVT・VT・AT・AV．融合の生徒，K=0．エントロピー最小化の固定版は `entmin|-|n`，oursx の調整版は `<T>_pseudo.csv` から選び直す）．反復 0〜9
- 描いたもの：点はターゲットごとの差（台帳 `cmp-entmin-fixed-vs-e39-<世界>-f-k0-<T>`），黒い横棒はターゲットの平均（`cmp-entmin-fixed-vs-e39-<世界>-f-k0`，`cmp-oursx-vs-e39-<世界>-f-k0`），細い線は同じターゲットの 2 点を結ぶ．ターゲットは印の形で区別した
- キャプションの案：
  > Differences from the five-value start at $K=0$ for the fused students, per target: entropy minimization with its literature setting (weight one per pool utterance) and ours-ext. Markers are targets, black bars are means over targets, and gray lines join the two values of a target.
- 読み取れること：エントロピー最小化の固定版は，同じ IEMOCAP 即興でも AVT で +4.6，AV で −28.3 と，ターゲットとモダリティの組で大きく得も損もする．ours-ext は 18 点（4 世界 × 4〜6 ターゲット）のうち AV の AFEW（−1.0）を除いて出発点以上にある．

## 8. finetune（6 節，段 1 つ 3.41 × 1.95 in）

- ファイル：`figures/finetune.pdf`，`figures/finetune.png`
- 台本：`scripts/figures/fig_finetune.py`
- 読んだ結果ファイル：`results/finetune_check/<T>.csv`（中心化なし），`<T>_c.csv`（中心化あり），`teacher_<T>.csv`（凍結版）．T は IEMOCAP 即興・台本，MELD，MOSEI，反復 0〜2，λ = 0.01
- 描いたもの：棒は 4 ターゲットの平均の UAR（台帳 `cmp-frozen-uar-*`，`cmp-ft-uar-*`，`cmp-ftc-uar-*` と同じ値）．棒は偶然の水準 25% から立てた（25% 未満の値は無い）．濃い灰は出発点のまま（K=0）か K 本だけ（K=5），薄い灰は K 本＋出発点への L2 罰則（凍結版ではこれが E39），青は弱いラベルを足した形
- キャプションの案：
  > Frozen versus fine-tuned audio encoders at $K=0$ (a) and $K=5$ (b). UAR of the audio student on frozen features (the five-value start, and with the cross-modal weak labels) and with LoRA fine-tuning of WavLM, without and with per-corpus centering: the source-trained start or the $K$ labels alone, the $K$ labels with an $\ell_2$ penalty toward the start ($\lambda = 0.01$), and with the weak labels of the fused teacher added. Mean over four targets and three iterations; bars start at chance (25\%).
- 読み取れること：どの K でも凍結版の 2 本が最も高く，微調整は中心化を足しても届かない．微調整の上でも，青（弱いラベル）が灰の棒より 3〜6 ポイント高く，出発点への罰則（薄い灰）はほとんど上げない．

---

## 本文の草稿との食い違い（2026-10-27 の `sections/5-*.tex`・`6-*.tex` と照らした）

1. compare のキャプションの草稿は「Error bars show the 95% interval over target and iteration pairs」と書くが，図のひげはターゲットの間の範囲である．上の案に替えるか，区間の図に作り直す
2. 表 tab:compare の AVT の顔の K=5 の最も強い相手は「CoR」だが，調整版の相手はどれも全ターゲットで「使わない」を選び，差は 0.00 で同点である（台帳の key は名前の順で coreg を選んでいる）．図では「none」と書いた．表も「none」か「—」にするのがよい
3. ladder のキャプションの草稿は「on a common set of targets」と書くが，(a) は 7 ターゲット，(b) の線は 7・6・4・4 ターゲットで，共通の集合ではない．また 5 節の「Cross-modal weak labels raise every student above the five-value start (Fig. ladder)」は全生徒の話だが，ladder は音声の生徒だけを描く．全生徒の得は compare（K=0・5）にある
4. method のキャプションの草稿は「The teacher, $m$, and the anchors are selected for each $K$」と書くが，錨の強さ C′・C_b は 5 つの値の一部で，K によらず 1 度だけ選ぶ（E39．E60・E61 の生徒は E39 の値のまま）．K ごとに選ぶのは先生・重みの付け方・m である
5. ladder (a) の HT（P4）はターゲットの平均で K=5 に 53.8 だが，台帳 `p2-e31-p4-uar-k5-a` は組の平均の 53.6 である（台帳の note にも書かれている）．本文で P4 の水準を図と並べて書くなら，どちらの平均かをそろえる
6. ladder (a) の出発点は ext30 の E39（3 候補ずつの格子，反復 30 回，A の K=5 で 56.3）で，(b) の差の基準になる E59・E60 の E39（A・V は E40 の random の組の設定，反復 10 回，A の K=5 で 56.2）とは設定が少し違う．本文で両方を「five-value start」と呼ぶなら，その違いを補足資料に書く
