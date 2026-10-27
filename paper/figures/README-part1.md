# Part I（4 節）の図

図はすべて `scripts/figures/fig_<名前>.py` が結果ファイルを読んで作り，`style.save()` で `figures/<名前>.pdf`（本文用）と `.png`（確認用）に保存する．作り直しは paper/ で次を流す（どれも数秒）．

```bash
for f in gainmap twocloud thresholdshare centering lambda sourcevalue; do ../.venv/bin/python scripts/figures/fig_$f.py; done
```

結果ファイルの場所は台帳 `scripts/ledger/part1.py` の定数（`FIXC_RERUN`，`STAGEB`，`E28T9`，`E29`，`E32`）を import して共有し，集計も台帳と同じ式で行う．`fig_gainmap.py` と `fig_thresholdshare.py` の値は台帳の key と一致することを確かめた（`fig_centering.py` は実行時に台帳と同じ値を標準出力に出す）．

| 図 | 大きさ（PDF） | 入れ方 |
|---|---|---|
| gainmap | 7.03 × 2.14 in | `figure*`，`\textwidth` |
| twocloud | 3.41 × 3.13 in | `figure`，`\columnwidth` |
| thresholdshare | 3.41 × 3.42 in | `figure`，`\columnwidth` |
| centering | 3.36 × 1.80 in | `figure`，`\columnwidth` |
| lambda | 3.44 × 2.24 in | `figure`，`\columnwidth`（補足資料に回してもよい） |
| sourcevalue | 3.39 × 2.43 in | `figure`，`\columnwidth` |

---

## 1. gainmap（C を固定した地図の得）

- ファイル：`figures/gainmap.pdf`
- 台本：`scripts/figures/fig_gainmap.py`
- 読んだ結果：`results/fixc_map/<kind>/summary/pairs.csv`（8 種類．列 `diff_mean`・`diff_lo`・`diff_hi`．同じコーパスの組を含む全組，各組 30 回）．台帳 `map-gain-*`・`map-gain-pos-*`・`map-gain-neg-*` と同じ出所
- キャプションの案：Gain of pooling, the pooled UAR minus the target-only UAR in the same iteration, on the gain map in which the pooled classifier takes the $C$ of the source-only classifier. (a) Mean over ordered pairs for each configuration against the label budget $K$. (b) Fraction of ordered pairs whose 95% interval of the paired differences over iterations lies above zero, contains zero, or lies below zero, for A, V, AV, and T; the bars of each group are $K = 1, 5, 10, 25, 50$ and the full pool.
- 読み取れること：組の平均の得は 8 種類とも K=1 で 7.6〜13.7 点あり，K=25 以上ではどの種類も負になる．V だけは K=10 ですでに負である．区間が 0 を上回る組の割合は K とともに減る．

## 2. twocloud（2 つの山の模型と，ソースのみの分類器の損の内訳）

- ファイル：`figures/twocloud.pdf`
- 台本：`scripts/figures/fig_twocloud.py`
- 読んだ結果：(a) は模式図でデータを読まない（d と o は説明用の値で，図に数値を書かない）．(b) は `results/rerun_fixc/explore_relation/snapshots/20261018-0213_fixc/e13_pairs.csv`（7 種類，列 `loss_offset`・`loss_direction`）と同 `check_mh/e13.csv`（A (HuBERT) の要約）．台帳 `p1-e13-loss-offset-*`・`p1-e13-loss-dir-*` と同じ出所
- キャプションの案：(a) Two-Gaussian model of a pairwise decision: $z$ is the score difference between emotions $k$ and $l$, $d$ the separation of the class means, and $o$ the offset of their midpoint from the boundary $z = 0$, both in units of the within-class standard deviation $\sigma$. Each class density is drawn at half height, so the shaded area equals the pairwise error $1 - a(d, o)$ and the hatched area equals the threshold loss that moving the boundary to the midpoint recovers. (b) Threshold loss and direction loss of the source-only classifier for each configuration, averaged over the six emotion pairs and over ordered pairs, in points of pairwise accuracy.
- 読み取れること：音声の 2 エンコーダと AV では向きの損が判断基準の損の 1.9〜2.5 倍で，V だけは判断基準の損（2.4 点）が向きの損（1.9 点）を上回る．言葉を含む 4 種類では判断基準の損が 1 点前後しかなく，損のほとんどが向きの損である．

## 3. thresholdshare（合併の上乗せのうち判断基準の分の割合と，K 本で減る 2 つの誤差）

- ファイル：`figures/thresholdshare.pdf`
- 台本：`scripts/figures/fig_thresholdshare.py`
- 読んだ結果：(a) `results/explore_relation/snapshots/20261011-1524_e29/e29_pairs.csv`（列 `share_pos`，K=1・5・10．同じコーパスの組を除く）．台帳 `p1-e29-critshare-k<K>-<kind>`．(b) `results/explore_relation/snapshots/20261011-0139_e28t9/e28t9_targets.csv`（列 `q_dir` のターゲットの中央値）．台帳 `p1-qdir-*`
- キャプションの案：(a) Share of the pooled classifier's improvement over the source-only classifier that comes from the change of the threshold offset $o$ under the two-Gaussian model; lines show the median over ordered pairs from different corpora, solid for single-modality audio and face and dashed for fused configurations and configurations with text. (b) Error of the estimates from $K$ labels per emotion: the variance of the estimated threshold offset, $1/(2K)$, and the noise-to-signal ratio of the estimated direction, $q/K$, with $q$ the median over targets of the value fitted to the target-only separation. To first order in $1/K$, multiplying both terms by $(\Delta/4)\varphi(\Delta/2)$ gives the expected loss of pairwise accuracy.
- 読み取れること：K=5 では単独の A・A (HuBERT)・V が 0.54〜0.65 で 0.5 の線より上にあり，AV と言葉を含む種類は VT（0.509）を除いて 0.5 より下にある．ただし K=1 では AV・VT も 0.7〜0.8 と高く，K=10 では A (HuBERT) が 0.5 を下回る．(b) では向きの誤差がどの K でも判断基準の誤差の 2q 倍で，音声ではそれが 30 倍を超える．
- 注意：(b) の最後の文（共通の係数で期待の損になる）は 4 節の本文で導く必要がある．導かない場合は最後の文を削る．

## 4. centering（ラベル無しの中心化と K=5 のラベル．E12）

- ファイル：`figures/centering.pdf`
- 台本：`scripts/figures/fig_centering.py`
- 読んだ結果：`results/mech_<kind>/*.csv`（op = `center`・`none`，method = `src_only`・`src_only+D1`・`merged`．学び直しの反復 0〜2）と `results/fixc_map/<kind>/*.csv`（`src_only`，`merged` の K=5）．A，A (HuBERT)，V．台帳 `p1-center-srconly-*`，`p1-d1-gain-*-k5`，`p1-merged-gain-*-k5`，`p1-center-d1-gain-*-k5`，`p1-center-merged-gain-*-k5` と同じ値（後ろの 2 本の棒は「中心化の分 ＋ 中心化の後の上乗せ」）
- キャプションの案：Change of UAR from the source-only classifier for A, A (HuBERT), and V, averaged over ordered pairs. Centering subtracts the mean feature of each corpus without using labels; D1 adds per-emotion offsets learned from $K = 5$ target labels to the source-only scores; pooling trains on the same labels together with the source data. Black ticks mark the level of centering alone, so the distance between a tick and its bar is what the labels add after centering.
- 読み取れること：音声では中心化だけで 4.2〜4.9 点上がり，K=5 の D1（4.1〜4.9 点）と合併（4.6〜5.4 点）の得とほぼ同じである．中心化の後にラベルが足す分は 0.6 点以下である．V では中心化の後に D1 も合併も損をする（合併は 1.6 点下がる）．
- 注意：「Pooled」は C を固定した地図の合併で，「Centering + pooled」は学び直しの合併（C は交差検証）である．C の決め方が 2 本で違うことを本文か補足資料に書く．

## 5. lambda（合併が判断基準のずれを直す割合 λ．E15）

- ファイル：`figures/lambda.pdf`
- 台本：`scripts/figures/fig_lambda.py`
- 読んだ結果：`results/fixc_analysis/explore_relation/snapshots/20261009-0105_fixc/stageB/e15b_pairs.csv`（列 `lam_pred_rep09`・`lam_obs`．組 × K=1・5・10，同じコーパスの組を含む，反復 0〜9）．凡例の順位相関は台帳 `p1-e15-lam-rho-*` と同じ計算
- キャプションの案：Fraction $\lambda$ of the source-only threshold offset that the pooled classifier removes, predicted by $\lambda = \rho/(1 + \rho)$ from a single Newton step (horizontal) and measured on the test utterances of the target (vertical), for each ordered pair and $K \in \{1, 5, 10\}$. The dashed line is $y = x$, and the legend gives the Spearman correlation for each configuration.
- 読み取れること：順位は 4 種類とも 0.84〜0.91 の相関でそろうが，予言の λ が 0.5 を超えるあたりから点は対角線の下に寄り，式は λ を大きめに見積もる．

## 6. sourcevalue（ソースの価値 n．E32）

- ファイル：`figures/sourcevalue.pdf`
- 台本：`scripts/figures/fig_sourcevalue.py`
- 読んだ結果：`results/explore_relation/snapshots/20261016-0954_e32/e32_pairs.csv`（1 ソース：`uar_P1`・`n_pred`．同じコーパスの組を除く）と `e32_allsrc.csv`（全ソース：`P4`・`n_pred`），学習曲線は `results/fixc_map/<kind>/*.csv` の `tgt_only`（反復 0〜9）．測った n は台帳 `part1._curves`・`_k_equiv` で読み直し，ファイルの `n_meas`・`flag_meas` と一致することを確かめた．K=5・10（台帳の順位相関 `p1-e32-rho-*`・`p1-e32-allsrc-rho-*` と同じ範囲）
- キャプションの案：Value of the source in target labels per emotion for $K = 5$ and $10$: the predicted $n = q e^2/(1 - e^2)$ from the direction efficiency $e$ and $q$ (horizontal) against the measured $n$, the number of target labels at which the target-only learning curve reaches the UAR of hypothesis transfer from the source, minus $K$ (vertical). Small markers are single sources from other corpora and large outlined markers all sources together; the top strip holds cases above the target-only UAR at $K = 50$, the bottom strip cases with measured $n \le 0$, and the dashed line is $y = x$.
- 読み取れること：対数の軸で点は対角線に沿い，全ソースの点（音声で 12〜41 本ぶん）は 1 ソースの点より右上にある．測った n が 0 以下の点は V と AV に多く（1 ソースで V 12 点，AV 10 点），予言の n が小さい組に集まる．
