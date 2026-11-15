# 関門 1：実装と出力・差分の第 4 回の検査（DAI-7，2026-11-07，コミット 037a115．設計は第 7 版）

検査したもの：`git diff 7df1bd3 037a115 -- DAI/exp/dai7_probe_v2.py DAI/docs/実験7_下調べ_設計.md` の全部．差分の中で呼ばれる `crossfit`・`lda_fit`・`boot_weights`・`fold_combine`・`boot_D_unit`・`boot_uar_unit`・`ci` と，jackknife の比べのために実験 1 の `dai1_stats.effect_stats`・`g_eff_of`，計画の 8 節（`DAI/docs/計画_2026-11-04.tex` 261 行）と実験 1 の設計 9.3 節も読んだ．`7df1bd3..037a115` で変わったコードは `DAI/exp/dai7_probe_v2.py` だけである（`DAI/exp`，`measure`，`DAI/frozen` の差分の一覧で確かめた）．試走は CPU（`CUDA_VISIBLE_DEVICES=""`）で，コミットした台本（sha256 37982dc3…，037a115 の版と同じ）をそのまま流した．作業木の `dai1.git_commit` が見る範囲は clean で，provenance と集計の出力の commit は 037a115，dirty は False だった．出力は `/home/snics/MER-DA/DAI/checks/DAI-7/code_run_v6/out/`，検算の台本（`check_v6.py`，`check_jk.py`），検算の結果（`check_v6_out.txt`，`check_v6_out2.txt`，`check_jk_out.txt`），起動の台本（`run6.sh`）とログ（`log_*.txt`）も同じフォルダに置いた．`results/dai7_probe/`，`results/dai7_probe_v2/`，`logs/dai7_probe.log` と tmux の画面は開いていない．

本検査が見た値：形，列，NaN と Inf の数，作り直した値と行の値の差，前の版との差，恒等式の差，実効の塊の数，jackknife の標準誤差の比，所要，ファイルの大きさ．D，ΔD，φ，UAR，区間と criteria の判定の値は画面に出していない（criteria の作り直しは一致の有無だけを出した）．

## 判定
通す．第 7 版で足した量（各モダリティだけの判別分析と 4 つの分，H4 の数え方，jackknife と t の区間，共有の比べの添え書き，IEMOCAP の固定の添え書き，POOL_SETS，`n_dropped_scen`，`tether_weakening_units.csv`）は設計どおりに計算され，作り直しの検算と前の版との照合はすべて一致した．残りは記録する 5 件で，どれも結果の数値を変えず，計算の後に保存した出力から直せる．

## A．設計と台本の対応表（差分の範囲）

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 3.5 各モダリティだけの判別分析（3 次元の点数，感情の事前を等しく，同じ 2 つの半分で交差の当てはめ） | `run_unit` の `p_lda = {m: crossfit(zt[m], yt, half, False) for m in MODS}`（447 行） | 合う．`lda_fit` の切片は `0.5·Σ A·mu` だけで事前の項が無いので，事前は等しい．`half` は 9 次元の blk・full と同じものである．試走の 7 単位で，fits の z と half から作り直した判定との不一致は 0 本だった |
| 3.5 恒等式 S = (Ḡ − Ḡ_cal) + (Ḡ_cal − G_full) + ΔL_dep + ΔL_cal | `decompose`（596〜606 行），`summarize` の `bP`（743 行）と `qs`（800 行） | 合う．試走の点推定で \|S − 4 つの分の和\| の最大は 8.9e-16，\|S1 − (Scal + S1b)\| は 4.4e-16 だった．抜き直しの回も同じ `decompose` を通る |
| 3.5 uar.csv に単独・判別分析・prod・blk・full の再現率 | `arows` の `uar_lda_*`，`rec_lda_*_e*`（454〜457 行） | 合う．uar.csv は 65 行 × 66 列（前の版から 15 列増えた）で NaN は 0．UAR と再現率を fits の判定から作り直した差の最大は 7.1e-15 |
| 3.5 G_cal を diffs に出す | `G_cal=dec["Gcal"]`（807 行） | 合う．uar_cells から作り直した差の最大は 5.3e-15 |
| 4 H4：前提 (iv) の区間の条件を満たした K だけを数える | `ivs = [K for K in KS_RULE if gs.loc[K, "hi"] < 0]`（920 行） | 合う．joint_a1 では `additivity_fails` と同じ K の集合になる |
| 4 H4：ΔL_dep の区間の下端 > 0 で割合 0.5 以上・0.25 以上の K を数える | `base_ok`，`H4_n_K_half`，`H4_n_K_part` | 合う．第 6 版の「S > 0」の条件は外れた．前提 (iv) の区間の上端 < 0 の K では S の点推定も正になるので，規則の意味は変わらない |
| 4 H4：4 つの分のうち最も大きい分，同点は全部 | `big`，`cnt`，`top`（929〜936 行） | 合う．K ごとの同点と，K をまたいだ数の同点の両方で全部を `PART_ORDER` の順に書く |
| 4 H4：K ごとの割合と h − h⁰ ≥ 0.5 のセルの数を添える | `H4_share_by_K`，`wd4 += …`（937 行） | 合う．3 つの場合のどれでも添える．前提 (iv) を満たさないコーパスでは添えない（設計どおり）．試走で添え書きの有無と `additivity_fails` は 4 コーパスとも一致した |
| 4 jackknife と t の区間（実効の塊の数 < 20．ΔD の 3 つの差） | `n_eff`（733〜735 行），`boot_D(..., jk=True)`（626 行），`jk_interval`（668 行），`DD_JK`，diffs の `variant="jk"` の行 | 合う（式は下の「jackknife の確かめ」）．本番の 5 分割では実効の数は AFEW 14.99，IEMOCAP 4.91，MELD 8.59，MOSEI 783.22 になり，計画 8 節の値と一致した（`DAI/splits/<c>.csv` の全発話で数えた値と，5 つのテストの発話で数えた値が同じだった．塊は分割をまたがない）．したがって自由度は 13，3，7 になり，MOSEI には jackknife の行が付かない．ただし種の項を足さない点が計画 8 節の規則と違う（[記録 1]） |
| 4 jackknife で合否が変わる K の数 | `jk_{H1,H1p,share}_n_K_flip`，`jk_*_n_lo_pos` | 合う．diffs から作り直した値と一致した |
| 4 共有の比べの添え書き（ρ = 1，a = 1 の点で比べた） | `wd += "…ρ = 1，a = 1 の点で比べた）"` の 2 か所 | 合う |
| 4 m10：joint_a1 の ρ が 1 でない (単位, K, モダリティ) の数 | `n_unit_cells_rho_not1`，`n_unit_cells` | 合う．`tether_weakening_units.csv` から作り直した数と 4 コーパスとも一致した．H1_wording の文には数が入らない（[記録 3]） |
| 4 b3：共有の比べの K にも MOSEI の漏れと塊の水準の確かめを当てる | `chan_share_n_K_robust`，`cent_share_n_K_small`，`shK = ok_K(c, "dD_rho1_ownret", FLOOR_D, True)` | 合う．数える K は共有の比べの条件を満たした K で，設計の「共有の比べを満たした K について」と同じ．diffs から作り直した値と一致した |
| 4 b2：IEMOCAP で H1 と H1′ を満たしたら結果に依らず添える | `if c == "IEMOCAP": wd += …`（H1 と H1′ を満たした枝の中） | 合う．試走で，添え書きの有無と「c が IEMOCAP かつ H1 かつ H1′」は 4 コーパスとも一致した |
| 3.3 `n_dropped_scen` | `run_unit` の行（432 行） | 合う．IEMOCAP の 3 単位で行の値（3，7，3）は fits の `mask_scen` から数えた値と一致した．IEMOCAP 以外と none_tw の行では NaN |
| 5 POOL_SETS に ρ = 1 と tt_ownret | `POOL_SETS`（70 行） | 合う．試走の fits の zpool の組は 9 つ（none，joint，joint_a1，oracle，ρ の 0.0625・1・4，tt_ownret，ret）で，各 K に 9 個あった |
| 5 `tether_weakening_units.csv` | `summarize` の書き出し（`TW`） | 合う．84 行（7 単位 × 4 K × 3 モダリティ）．コーパスごとの平均の表との差の最大は 1.4e-14 |
| 5 fits に各モダリティだけの判別分析の判定 | `K{K}_{set}_pred_lda_{m}` | 合う．新しい鍵は 1 単位あたり 205 個（zpool 10 個と判別分析の判定 195 個） |
| b1・b4・b5・b6・記録 4・記録 6・記録 7 | 設計の文だけ | 台本の変更は要らない |

## A．小さく流した結果

- 自己検算（`--selftest`）：14 項目すべてが通った（2.3 秒）．(13) は塊ごとの値の平均で jackknife の標準誤差が sd/√n に一致することを，(14) は情報が同じで none の点数にだけ感情の偏りがあるとき偏りが Ḡ − Ḡ_cal に入り Ḡ_cal − G_full が 0 になることを確かめる．(14) で Ḡ_cal が 0 になるのは，判別分析が点数に定数を足しても同じ判定を返すためで，式の上でも成り立つ．
- 単位の段（CPU，1 単位ずつ続けて流した）：
  | 単位 | 時間 | 最大メモリ |
  |---|---|---|
  | IEMOCAP_AFEW_f2_r0，IEMOCAP_AFEW_f0_r0（c=AFEW，テスト 158） | 各 7 秒 | 0.9 GB |
  | MELD_IEMOCAP_f0_r0，MOSEI_IEMOCAP_f0_r0，MELD_IEMOCAP_f1_r0（c=IEMOCAP） | 17〜19 秒 | 1.2 GB |
  | AFEW_MELD_f0_r0（c=MELD，テスト 2,105） | 39 秒 | 2.2 GB |
  | AFEW_MOSEI_f1_r0（c=MOSEI，テスト 3,727） | 69 秒 | 3.5 GB |

  前回の 4 単位に，jackknife を確かめるため IEMOCAP の分割 1（MELD_IEMOCAP_f1_r0）と同じ分割 0 の別の T（MOSEI_IEMOCAP_f0_r0），AFEW の分割 0（IEMOCAP_AFEW_f0_r0．1 つの塊がテストの大半を占める分割）を足した．警告と例外は無かった．
- 集計の段（7 単位，`--allow-partial`）：44 秒，0.7 GB．警告は pandas の PerformanceWarning（712 行の列の追加）だけである．
- 形：units.csv は 1 単位 378 行で，列は IEMOCAP 102，ほか 74（前の版から `n_dropped_scen` の 1 列）．cells.csv 2,056 行，uar_cells.csv 260 行 × 59 列，diffs.csv 3,105 行 × 26 列（main 1,860，cent・keep 各 320，scen・chan・sub_*・spk 各 80，jk 45），criteria.csv 4 行 × 110 列，tether_weakening.csv 48 行，tether_weakening_units.csv 84 行．
- NaN：diffs の main で区間が NaN の行は，区間を作らない dDt と G_joint を除いて 0．jk の 45 行（AFEW・IEMOCAP・MELD の 5 K × 3 つの差）の lo・hi に NaN は 0．criteria の NaN は，MOSEI 以外の `chan_*`，jackknife の行が無い MOSEI の `jk_*`，前提 (iv) の区間の条件を満たす K が 0 の MELD と MOSEI の `H4_*largest_by_K`・`H4_*share_by_K`（空の文字列が CSV で NaN になる）だけである．
- 前の版（7df1bd3，`code_run_v4/out/`）と共通の 4 単位の照合：units の 378 行の共通の数の列（IEMOCAP 93 列，ほか 65 列）の差は 0.0 で NaN の形の不一致も 0，uar の 65 行の共通の 43 列の差も 0.0，fits の共通の鍵（915〜918 個）に違う鍵は無く，消えた鍵も無かった．
- criteria の新しい数え方の作り直し：`chan*_n_K_robust`，`cent*_n_K_small`，`jk_*`，`H4*_n_K_iv`・`_n_K_half`・`_n_K_part`・`_largest_by_K`，`n_cells_h_excess_ge_0.5` を diffs と uar_cells から独立に数え直し，4 コーパスで不一致は 0 だった（MELD と MOSEI の `largest_by_K` は台本が空の文字列，CSV では NaN で，作り直しも空だった）．
- 各モダリティだけの判別分析を sklearn の `LinearDiscriminantAnalysis(priors=[0.25]*4, solver="lsqr")`（罰則なし）で作ると，判定の 1.1〜2.5% が違った．台本の判別分析は共分散に 1e-3·tr/3 の罰則を足すので（`EPS_LDA`），違いは罰則の分だと読む（読んだだけ．罰則を 0 にした版との照合はしていない）．

### jackknife の確かめ（確かめてほしいことの 2）

- 重み：`w = 1 − I`（塊の総数 G の正方行列）を `fold_combine` に通すと，行 g では塊 g を含む分割の重みが「残りの塊の数 ÷ 分割の塊の数」= (n_f − 1)/n_f，ほかの分割の重みが 1 になる．塊が 1 つだけの分割（IEMOCAP のすべての分割）では重みが 0 になり，その分割は θ_(−g) から外れる．`boot_D_unit` では塊 g の発話の重みが 0 になるので，分割の値は塊 g を除いた発話で作り直される．感情の本数が 15 本未満になった感情はその回だけ外れる（主の抜き直しと同じ）．
- 検算：塊 g の発話を fits から実際に消して D を作り直し（d⁰ は行の値），上の重みで分割をまとめた θ_(−g) と `boot_D(jk=True)` の値を，4 コーパス・K = 1 と 25・5 つの組で比べた．差の最大は 5.0e-16，NaN の形の不一致は 0 だった．AFEW の分割 0 で大きい塊を抜くと分割 0 が外れることも一致した．
- 分散：`jk_interval` は (G − 1)/G · Σ(θ_(−g) − θ̄)² の平方根に t_{0.975, max(1, ⌊実効の数⌋ − 1)} を掛け，点推定を中心にする．G は有限の θ_(−g) の数（= 塊の総数）である．実験 1 の `effect_stats` の V_jk と同じ式である．
- 実験 1 との違い：実験 1 の `effect_stats` は，塊を抜いても分割が残る限り分割の重みを 1 のままにする．この台本は (n_f − 1)/n_f にする．試走の AFEW（分割 0 と 2）で，同じ θ_(−g) を分割の重み 1 でまとめた版と標準誤差を比べると，台本の版との比は 0.99〜1.08 だった．本番では MELD の分割 4 の塊が 2 つしか無いので，その分割の重みが 1/2 になり，違いが試走より大きくなりうる（[記録 2]）．

## A．本番の規模の見積もり

- 単位の段（CPU）：単位あたりの時間は前の版と変わらない（AFEW 7 秒，IEMOCAP 17〜19 秒，MELD 39〜53 秒，MOSEI 69 秒）．30 × (7 + 19 + 39〜53 + 69) 秒で，約 67〜74 分になる．
- 集計の段：7 単位で 44 秒（前の版の 4 単位で 28 秒）．jackknife は K ごとに G 行（AFEW 51，IEMOCAP 5，MELD 約 270）の重みで `boot_D` を 1 回足すだけで，1,000 回の抜き直しより軽い．本番は約 10〜20 分の見積もりを変えない（読んだだけ）．
- メモリ：単位の段の最大は 3.5 GB（MOSEI），集計の段は 1 GB 以下の見込み．
- `fits/`：1 単位の大きさは AFEW 1.5 MB，IEMOCAP 4.9〜5.3 MB，MELD 14.6 MB（分割 0 の 2,105 本．最大 2,850 本の分割では比例して約 19.7 MB），MOSEI 29.0 MB で，前の版より約 1 割増えた．30 × (1.5 + 5.3 + 14.6〜19.7 + 29.0) MB で，約 1.5〜1.7 GB になる．rows は 1 単位約 0.3 MB．
- GPU は要らない．

## B．予想される指摘と，出力から答えられるか（前回の表に足す行）

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| jackknife の区間 | 計画 8 節どおり種の項 V_seed を足せ | 計画 8 節，実験 1 の 9.3 節 | (a)：units.csv の (T, 分割, 種) ごとの D（pair の 3 行の平均）から，組の差の分割の中の分散を作り，`effect_stats` の式で足す．IEMOCAP は塊 = 分割なので 0 になる | 設計の文を直す（[記録 1]） | — |
| jackknife の区間 | 実験 1 と同じく分割の重みを 1 のままにした jackknife で出せ | 実験 1 の `effect_stats` | (a)：`checks/DAI-7/code_run_v6/check_jk.py` の手順（fits の z から塊を消して作り直す）で出せる．CPU で数分〜数十分 | — | — |
| H3・H4 | UAR から作る量（ΔL_dep，G，S と 4 つの分）にも jackknife の区間を付けよ | m9 | (a)：fits の判定（`pred_prod`・`blk`・`full`・`lda_*`・単独）と `test_cluster` を `boot_uar_unit` に `w = 1 − I` で渡し，`fold_combine` と `jk_interval` に通す | — | — |
| H4 | Ḡ − Ḡ_cal をモダリティごとに出せ．どのモダリティの較正の得か | M4 | (a)：uar_cells の `uar_{m}` と `uar_lda_{m}` の none との差．区間は fits の判定から `boot_uar_unit` で作る | — | — |
| H4 | 較正の得を感情ごとに出せ | a2，b6 | (a)：uar.csv の `rec_{m}_e*` と `rec_lda_{m}_e*` | — | — |
| H4 | 各モダリティだけの判別分析の罰則を変えたら Ḡ_cal は変わるか | R（決め方の感度） | (a)：fits の z と half から `crossfit` を罰則を変えて作り直す | — | — |
| 共有の比べ | 選んだ ρ の点で tt_ownret と比べよ | m10 | (c)：tt_ownret は保存に ρ = 1 だけがあり，ほかの ρ には実験 1 の学び直しが要る．設計 4 節に「ρ = 1，a = 1 の点で比べた」と書き，ρ が 1 でない数を添えると書かれているので，主張はこの範囲に限られている | — | — |
| 共有の比べ・H1 | jackknife の区間を cent・chan の変種にも付けよ | m9，A2 | (a)：`boot_D(fits, Uc, K, var, jk=True)` を変種ごとに呼ぶだけで，fits と rows から出せる | — | — |

学び直し（実験 1 の分類器の学習）が要る (b) は無い．

## 指摘

### [記録 1] jackknife の区間は計画 8 節の規則の種の項を足さないが，設計は「計画 8 節の規則」と書く
- 重さ：記録する
- 該当箇所：設計 3.3 節の「jackknife と t の区間」の段落（「（計画 8 節の規則）」），`DAI/exp/dai7_probe_v2.py:668` の `jk_interval`（分散は `(n - 1) / n * np.sum((th - th.mean()) ** 2)` だけ）
- 問題：計画 8 節（`計画_2026-11-04.tex` 261 行）と実験 1 の設計 9.3 節の規則は「塊を 1 つずつ抜く jackknife の分散に抜き方の種の分散を足し」と書く．台本は塊の項だけで区間を作る．主の抜き直し（4 節）も種の項を含まないので，「jackknife で合否が変わる K の数」は塊の扱いだけの比べとしては筋が通る．ただし設計の文のままでは，読み手が計画 8 節と同じ区間と読む．この台本の 1 つの分割の値は 3 つの T × 2 つの種の 6 単位の平均で，実験 1 の「種」と同じ意味の揺れは 2 つの種の間にしか無い．読んだだけ（試走の単位はすべて種 0 なので，種の項の大きさは確かめていない）．
- 計算の後で直せるか：直せる（units.csv の単位ごとの D から種の項を作れる）．
- 求める対応：設計の文を「計画 8 節の規則のうち塊の jackknife と t の自由度を使い，種の項は足さない（主の抜き直しも種の項を含まないので，2 つの区間を同じ揺れの範囲で比べるため）」に直す．種の項を足した区間も出すなら，計算の後に units.csv から作る．

### [記録 2] jackknife の分割の重みが実験 1 の規則と違う
- 重さ：記録する
- 該当箇所：`DAI/exp/dai7_probe_v2.py:626-627`（`w = 1.0 - np.eye(w.shape[1])`）と `fold_combine`（`wf = cnt / size[None, :]`）
- 問題：台本の θ_(−g) は，塊 g を含む分割の重みを (n_f − 1)/n_f にする．これは主の抜き直しの重みの作り（抜かれた塊の数 ÷ 分割の塊の数）を w = 1 − e_g に当てたもので，主の抜き直しと同じ関数の jackknife である．実験 1 の `effect_stats` は分割の重みを 1 のままにする．試走の AFEW で 2 つの版の標準誤差の比は 0.99〜1.08 だった．本番の MELD の分割 4 は塊が 2 つなので，塊を 1 つ抜くとその分割の重みが 1/2 になる．
- 計算の後で直せるか：直せる（fits から，`check_jk.py` の手順で分割の重み 1 の版を作れる）．
- 求める対応：設計の jackknife の段落に「塊を抜いた分割の重みは，主の抜き直しと同じく残りの塊の割合にする（実験 1 の規則は 1 のまま）」と一文足す．

### [記録 3] H1_wording の文に，joint_a1 の ρ が 1 でないセルの数が入らない
- 重さ：記録する
- 該当箇所：`DAI/exp/dai7_probe_v2.py` の共有の比べの文（`wd += "…ρ = 1，a = 1 の点で比べた）"`）と `r["n_unit_cells_rho_not1"]`
- 問題：設計 4 節は「joint_a1 の ρ が 1 でない (単位, K, モダリティ) の数を添える」と書く．数は criteria の `n_unit_cells_rho_not1` と `n_unit_cells` にあるが，`H1_wording` の文には入らない．
- 計算の後で直せるか：直せる．
- 求める対応：結果の文を書くときに criteria の 2 列から数を添える．台本で文に入れるなら，共有の比べの文の後ろに「（joint_a1 の ρ が 1 でないセル x / y）」を足す．

### [記録 4] provenance に NEFF_JK と PART_ORDER が無い．`ivK` は使われない
- 重さ：記録する
- 該当箇所：`main` の `run = dict(...)`（新しい定数が入っていない），`summarize` の `ivK = [K for K in KS_RULE if g.loc[K, "hi"] < 0]`（917 行）
- 問題：jackknife を付ける実効の数の境（20）と H4 の分の順は provenance.json に残らない．コミット番号から台本を引けば分かるので，害は無い．`ivK` は作るだけで使われない（joint_a1 の `ivs` と同じ集合である）．
- 計算の後で直せるか：直せる．
- 求める対応：次に台本を直すときに provenance に `neff_jk` と `part_order` を足し，`ivK` を消す．本計算の前に直す必要は無い．

### [記録 5] 集計の出力の commit と設計の版
- 重さ：記録する
- 該当箇所：前回の [記録 7] の続き
- 問題：今回のコミット 037a115 は設計の文書と台本を一緒に入れたので，設計の版は commit から引ける．本計算の前に `results/dai7_probe_v2/` に前の版の出力が残っていないことを確かめる（本検査は `results/` を開いていない）．[記録 1]〜[記録 3] の設計の文を直してコミットすると commit が変わるが，`dai1.git_commit` は DAI/docs を見ないので，台本を変えない限り本計算の commit の扱いには影響しない．
- 求める対応：設計の文を直したら，そのコミット番号を台帳に書いてから本計算を流す．

当てて問題の無かった項目（差分の範囲）：I1（第 7 版との対応），I5（同じ単位の同じ添字．判別分析は 9 次元と同じ half），I7（共通の乱数は変わっていない．前の版との差 0），I9・I12（単位の段と集計の段のコミット 037a115，dirty は False），I11（有限の確かめ．uar.csv の NaN 0），I14（前の版の再現：共通の列と鍵の差 0），恒等式（8.9e-16 以下），jackknife の重みと分散の式（消して作り直した値との差 5.0e-16 以下），実効の塊の数（計画 8 節の値と一致）．

## 目録に無い種類の不備
1. I19（案）：同じ名前の区間の規則（例：「計画 8 節の jackknife と t の区間」）を別の実験で使うときは，元の規則の項（塊の項，種の項，抜いた分割の重み，自由度）を 1 つずつ並べ，どれを使い，どれを使わないかを設計に書く．名前だけを引くと，読み手は元の規則と同じ区間と読む（DAI-7 の gate-code の第 4 回の検査．実験 1 の規則は種の項を足し，抜いた分割の重みを 1 のままにするが，DAI-7 の台本はどちらもしない）．
