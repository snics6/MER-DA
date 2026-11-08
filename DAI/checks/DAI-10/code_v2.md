# 関門 1 の再検査：実装と出力（DAI-10 の下調べ，2026-11-08，コミット e408470 と 4a803b2）

検査した差分は次の 2 つである．
- `git diff 5fc3278 e408470 -- DAI/exp/dai10_probe.py DAI/run/dai10_probe.sh DAI/docs/実験10_下調べ_設計.md`：第 1 回（`code.md`）への対応と設計の第 3 版．
- `git diff e408470 4a803b2`：全ラベルの LDA の照合で止める 1 か所と，設計の noneT の定義と 5 節の出力の一覧．

検査の時点で `DAI/exp` などに未コミットの変更や未追跡のファイルは無い．

## 判定

**通す．**第 1 回で「止める」とした 2 つ（オラクルの角 LDA_SO・LDA_OT と，EM_Kmet の候補の上の割り当ての保存）と「直してから進む」とした α = 0.01 は，設計どおりに足されている．試走の 35 単位で照合の不一致は 0，NaN は 0 だった．分け方の足し算は数値の誤差（7.5e-15 以下）で閉じた．残る 5 つは記録の範囲で，計算の後でも直せる．

## A．差分と第 1 回の指摘の対応表

| 第 1 回の指摘 | 直した箇所 | 合っているか |
|---|---|---|
| [R4，R6] LDA_SO（K 本の平均 × 候補の全ラベルの群内の Σ_O） | `dai10_probe.py`（e408470）の `fO = P8.lda_from_R(Xp, onehot_all, Xp.T @ Xp, A_RULE)` と `pred[f"K{K}_{s}_ldaSO_a{A_RULE}"] = P8.delta(Xt, lda_metric(mu, fO["chol"]))` | 合う．Σ_O は DAI-8 の LDA_O と同じ関数・同じ α で作る．試走の 35 単位で，台本の全ラベルの LDA の判定と DAI-8 の保存の不一致は 0 本だった |
| [R4，R6] LDA_OT（全ラベルの平均 × 計量 A） | `pred[f"{s}_ldaOT_a{A_RULE}"] = P8.delta(Xt, lda_metric(fO["mu"], chols[A_RULE]))` | 合う |
| 上の 2 つの比べ | `COMPARE` の `ldaSO_ldaK`，`ldaT_ldaSO`，`ldaO_ldaSO`，`ldaOT_ldaT`．`COMPARE_0` の `ldaO_ldaOT`，`ldaO_oracle`（K = 0 の行） | 合う．設計 3 節の 5 つの差と一致する．試走の IEMOCAP で，LDA_T − LDA_K =（SO − K）+（T − SO）の残差の最大が 3.6e-15，LDA_O − LDA_SO =（O − OT）+（OT − T）+（T − SO）の残差の最大が 7.5e-15 だった |
| 全ラベルの LDA の照合で止める（4a803b2） | `mO > max(1, int(LDAK_TOL * len(pO)))` で `SystemExit` | 合う．K 本だけの LDA の照合と同じ許しである．読んで確かめた．止まる場合は試走で起きていない（不一致 0） |
| [R9] EM_Kmet の候補の上の割り当てと回ごとの値 | `em_kmetric(..., yU=yp[Um])` が `pool_hist` と `R` を返す．診断に `emKmet_hist`，`emKmet_pool_hist`，`P_K{K}_{s}_emKmet_a0.5`（int8）と `pool_y` を保存する | 合う．試走で `hist` は 10 個，`pool_hist` は 11 個（0 回目の初期値を含む）だった．`P_` の割り当てから K 本を除いて作った候補の上の UAR は `pool_hist` の最後と残差 0 で一致した．K 本の位置の `P_` は正解のラベルと一致した．候補のラベルは測定だけに使い，E 段にも M 段にも入らない |
| [R4] α = 0.01 | `ALPHAS = (0.01, 0.1, 0.5, 0.9)`．K 本だけの LDA を各 α で台本の中で解き，`D8_ALPHAS = (0.1, 0.5, 0.9)` では DAI-8 の保存と照合して止める | 合う．試走の 35 単位で照合の不一致は 0．α = 0.01 の LDA_T 系と LDA_K が保存され，`ldaT_a0.01_ldaK_a0.01` の比べがある |
| [記録] 縮みの係数が `diag.csv` から落ちる | 集計の段で 3 個以下のリストを列に展開する | 合う．`diag.csv` に `eigs_1〜3`，`eigs_null_mean_1〜3` が入る．10 個以上のリスト（`emKmet_hist`，`emKmet_pool_hist`）は JSON の文字列として `diag_units.csv` に残る |
| [記録] 起動の台本の `.failed` | `dai10_probe.sh` の `exit 1` の前に `touch logs/dai10_probe.failed` | 合う |
| [I8，I12] 行のコミットと provenance のコミットの照合 | `summarize` の `json.loads(pvf.read_text())["commit"] not in rc` で止める | 合う．単位の間で鍵の組が同じかの確かめは台本に無いが，試走の 30 単位と 5 単位ではどちらも鍵の組が 1 つだった |
| [記録] noneT を写す | `COPY_K` に `K{K}_{s}_noneT`，`COMPARE` に `ldaT_noneT` | 合う |
| [記録] 設計 8 節の範囲の外の理由 3 つ | 設計 8 節の「範囲の外」の行 | 合う |

**設計の第 3 版で足された項目の対応**：
- 縮みの係数の帰無（5 回の並べ替え）と n/d：`emo_eigs_null` と `diag` の `eigs_null_mean`，`eigs_null_summax`，`n_over_d`，`n_pool`，`dim`．合う．並べ替えの種は `SEED + 1 + i` で，DAI-8 の `_top3_null` と同じ作り方である．
- 引き算の版の λ_max：`lam_max_sub`．式は合う．Σ = L(I − (1 − α)/4 · L⁻¹M̂ᵀM̂L⁻ᵀ)Lᵀ なので，正定値であることと 4 × 4 の行列 (1 − α)/4 · M̂A⁻¹M̂ᵀ の最大の固有値が 1 未満であることは同じである．自己検査は尺度 0.3 と 3.0 で正定値の判定と λ_max < 1 の一致を確かめる．私が同じ乱数で流すと，0.3 では λ_max < 1，3.0 では λ_max ≥ 1 になり，両側を通っていた．ただし M̂ の中心の取り方が設計に書かれていない（[記録] の 1 つ目）．
- 仮定の表 (a)・(b)：設計 4 節に基準があるが，`criteria.csv` は数えない（[記録] の 5 つ目）．
- 2 節の「独立な単位」の記述：同じ (c, f, r) の 3 つの T で，台本が解いた判定と `P_` の割り当てが同じかを数えた．IEMOCAP の 10 組の (c, f, r) で 6,480 組の比べのうち違う組は 0 だった．MOSEI の f0 r0 の 3 つの T でも 224 組のうち 0 だった．設計の記述と合う．

## A．小さく流した結果

- 自己検査：`./.venv/bin/python DAI/exp/dai10_probe.py --selftest` は通った．
- 試走（出力は `/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad/dai10v2/` の下．検査の後に消した）：
  - c = IEMOCAP の 30 単位（e408470 で流した．4a803b2 との違いは止める 1 行だけ）：壁時計 1 分 48 秒，単位あたり 3〜4 秒．
  - AFEW_MOSEI_f0_r0，IEMOCAP_MOSEI_f0_r0，MELD_MOSEI_f0_r0，IEMOCAP_MELD_f2_r1，MOSEI_AFEW_f3_r0（4a803b2 で流した）：単位あたり MOSEI 8〜9 秒，MELD 5 秒，AFEW 3 秒．
  - c = IEMOCAP の集計（`--summarize --allow-partial`）：5.5 秒．
- 例外と警告：どの実行も終了コード 0．`-W always` で警告は 0 件だった．
- 照合：K 本だけの LDA の不一致は 35 単位すべてで 0（MOSEI の 1 単位で 178,896 本を比べた），全ラベルの LDA の不一致も 0．
- 出力の形：1 単位に判定の鍵 396，`P_` の鍵 16，`pool_y` は `P_` と同じ長さ．診断に有限でない値は 0．`cells.csv` は 396 行で，各行の単位の数が 30，NaN 0．`diffs.csv` は 300 行（18 の比べ × 4 空間 × 4 K ＋ ldaTblk_ldaT の 4 行 ＋ K = 0 の 2 つの比べ × 4 空間）で 21 種類，effect・lo・hi の NaN 0．`criteria.csv` は 57 列で NaN 0．`checks_units.csv` に `ldaO_mismatch`，`ldaO_compared` の列がある．

## A．本番の規模の見積もり

- 単位の段：単位あたり AFEW 3 秒，IEMOCAP 4 秒，MELD 5 秒，MOSEI 9 秒として，120 単位でおよそ 10〜11 分．
- 集計の段：比べが 196 行から 300 行に増えたので，MOSEI の塊の引き直しでおよそ 1.5 分．
- 合わせて 12〜13 分．
- GPU のメモリの最大は全体で 6.4 GB / 24.5 GB だった（別の表示の処理の 1.6 GB を含む）．6 割の上限と `empty_cache` は第 1 回と同じである．
- ディスク：MOSEI の 1 単位が 0.85 MB，IEMOCAP の 30 単位が 14 MB．120 単位でおよそ 70 MB．

## B．予想される指摘と，出力から答えられるか（第 1 回の表からの変化だけ）

| 表・図 | 予想される指摘 | (a)／(b)／(c) | 出し方 |
|---|---|---|---|
| G1 の表 | 縮みの損を UAR で確かめよ（Krijthe & Loog 2016 の分け方） | (b) から (a) になった | `diffs.csv` の `ldaT_ldaSO`，`ldaSO_ldaK`，`ldaO_ldaSO`，`ldaOT_ldaT`，`ldaO_ldaOT` |
| 2 つの経路の表 | EM_Kmet でも割り当てが感情でない分け方へ移るか | (b) から (a) になった | `diag_units.csv` の `emKmet_pool_hist`（JSON），`P_K{K}_{s}_emKmet_a0.5` と `pool_y`．K 本の位置は保存されていないので，`DD.k_orders(cp.y, pool, c, f, r)` で作り直す（試走で作り直した位置の割り当てが正解と一致した） |
| G1 の表 | α を 0 に近づけたら | (b) から (a) になった | `ldaT_a0.01_ldaK_a0.01` の行 |
| G3 | 選ばない引き戻しの none と比べよ | (a) | `ldaT_noneT` の行 |
| 仮定の表 | 仮定 (a)・(b) の可否を数えよ | (a) | `diffs.csv` の AVT の `ldaSO_ldaK` と `ldaT_ldaSO` の 4 行を，規則と同じ数え方で数える．`criteria.csv` には無い（[記録] の 5 つ目） |
| 引き算の版 | 引き算の版は本当に正定値でなくなるか | (a) | `diag.csv` の `lam_max_sub` を K と空間ごとに見る．中心の取り方の定義は [記録] の 1 つ目 |

第 1 回の表のほかの行（ターゲットごとの内訳，感情ごとの再現率，話者を単位にした区間，性別，MOSEI のチャンネルの部分集合，交互作用の区間）は，出力の形が変わっていないので (a) のままである．

## 指摘

### [記録] λ_max の M̂ の中心の取り方が設計に書かれていない
- 重さ：記録する
- 該当箇所：`dai10_probe.py`（4a803b2）の `lam_max_sub(muc, chols[a], a)`．`muc = mu - mu.mean(0, keepdim=True)`
- 問題：台本は K 本の 4 つの平均から，4 つの平均の平均を引いた M̂ で λ_max を出す．設計 3 節の Σ = A − (1 − α)M̂Π̂M̂ᵀ は，候補の標準化で全体の平均が 0 の座標の式（0 節の導出）である．K 本の平均を中心に寄せない M̂ を使うと，項が 1 つ（平均の平均の外積）増え，λ_max は大きくなる．どちらの定義でも正定値の判定と λ_max < 1 の対応は成り立つが，記述の値が違う．
- 計算の後で直せるか：直せる（判定は変わらず，記述の 1 列の定義の問題）．
- 求める対応：設計 3 節に「M̂ は K 本の 4 つの平均から，その平均を引いたもの（LDA_Tc と同じ）」と書く．

### [記録] 設計 5 節は LDA_O を「台本で解き直したもの」と書くが，保存する判定は DAI-8 の写しである
- 重さ：記録する
- 該当箇所：設計 3 節の「LDA_K と LDA_O は台本で解き直し」，台本の `COPY_0 = ("{s}_ldaO_a0.5", "{s}_oracle")`
- 問題：LDA_K は台本の解を保存するが，LDA_O は台本の解 `pO` を照合だけに使い，DAI-8 の判定を写して保存する．照合で 0.1% を超えれば止めるので，数値は 0.1% 以内で同じである（試走では 0 本違い）．
- 計算の後で直せるか：直せる．
- 求める対応：設計の文を「LDA_O は DAI-8 の判定を写し，台本で解き直した判定と照合する」に直すか，台本の `pO` を保存する．

### [記録] `P_` の割り当てに K 本の位置が入るが，K 本の位置を保存しない
- 重さ：記録する
- 該当箇所：`poolpred[f"K{K}_{s}_emKmet_a{A_RULE}"] = dkm["R"].argmax(1)`，`np.savez_compressed(...)`
- 問題：`P_` は候補の全部の割り当てで，K 本の位置では正解のラベルになる．候補の上の正しさを後から数えるには K 本を除く必要がある．K 本の位置は保存されていないが，`DD.k_orders` で決まった位置を作り直せる（試走で確かめた）．回ごとの値は `emKmet_pool_hist` が K 本を除いて記録している．
- 計算の後で直せるか：直せる．
- 求める対応：`kpos_K{K}`（int32）を保存すると，作り直しが要らなくなる．

### [記録] `diag.csv` は fold と seed の列も平均する
- 重さ：記録する
- 該当箇所：`DG.groupby(["c", "key"]).mean(numeric_only=True)`
- 問題：`diag.csv` の `fold`，`seed` の列は単位の番号の平均で，意味が無い．`n_pool`，`dim` も平均になる．値は `diag_units.csv` に単位ごとに残る．
- 計算の後で直せるか：直せる．
- 求める対応：groupby の前に `fold`，`seed` を落とす．

### [記録] 仮定の表の (a)・(b) を `criteria.csv` で数えない
- 重さ：記録する
- 該当箇所：`summarize` の `RULES = (("G1", ...), ("G2", ...), ("G3n", ...), ("G3j", ...))`．設計 4 節の仮定の表
- 問題：設計 4 節は，仮定 (a)「LDA_SO − LDA_K が規則と同じ数え方で 3 つ以上の K」と仮定 (b)「LDA_T − LDA_SO の区間の上端 < 0 の K が 3 つ以上」の基準を計算の前に固定した．台本はこれを数えないので，結果を見た後で人が数えることになる．`diffs.csv` から同じ数え方で出せるので，学び直しにはならない．
- 計算の後で直せるか：直せる．
- 求める対応：`RULES` に `("Aa", "ldaSO_ldaK")` と `("Ab", "ldaT_ldaSO")` を足すと，今の数え方の列（`_n_K`，`_n_hi_neg`，`_pos`，`_neg`）がそのまま出る．仮定 (b) は `_neg` を読む．

当てた項目で指摘の無いもの：I1，I3（候補のラベルは LDA_SO，LDA_OT，LDA_Tcpi，縮みの係数とその帰無，`pool_hist` の測定だけに入り，鍵の名前でオラクルと分かる），I5，I6，I7，I9，I10，I11，I12，I13，I14，I17，R2，R4，R6，R8，R9，R10．

## 目録に無い種類の不備

なし．

---

試走の出力の消去：検査の後に，作業用のフォルダ `/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad/dai10v2` の下の試走の出力，ログ，メモリの記録，検算の台本を消した．`results/dai10_probe/` は作っていない．値（UAR の差，規則の可否）は画面にも報告にも出していない．
