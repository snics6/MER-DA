# 関門 1 の第 3 回：実装と出力（DAI-10 の下調べ，2026-11-08，差分 4a803b2..764e3da）

検査した差分は `git diff 4a803b2 764e3da -- DAI/exp/dai10_probe.py DAI/run/dai10_probe.sh DAI/docs/実験10_下調べ_設計.md` である．台本が変わったコミットは次の 3 つで，d4c0aff と 346100d は設計の文だけを変えた．
- 035e363：集計の段に部分集合，MOSEI の channel の確かめ，仮定の表の数え上げを足した．
- 735d952：単位の段に NCM と EM_Amet を足し，集計の段，自己検査，起動の台本を直した．
- 764e3da：集計の段の channel の確かめを，全体で条件を満たした K に絞って数えるように直し，G2 を足した．

検査の時点で HEAD は 764e3da で，`DAI/exp` などに未コミットの変更や未追跡のファイルは無い．

## 判定

**通す．**NCM，EM_Amet，比べの 5 つ，規則の Aa・Aa2・Ab・Gn，部分集合の比べ，MOSEI の channel の確かめ，感情ごとの再現率は，設計の第 6 版どおりに計算している．7 単位の試走で次を確かめた．
- 照合の不一致は 0 だった．
- `diffs.csv` の NaN は 0 だった．
- 規則と channel の数を `diffs.csv` から数え直すと，`criteria.csv` の数と一致した．
- K = 1 では LDA_K と NCM の判定が全く同じだった．

残る指摘は記録が 2 つだけで，どちらも計算の後に直せる．

## A．差分と設計の対応表

| 設計の項目（第 6 版） | 台本の箇所（764e3da） | 合っているか |
|---|---|---|
| NCM：Σ = I の LDA | `pred[f"K{K}_{s}_ncm"] = P8.delta(Xt, lda_metric(mu, eye))`．Cholesky の因子に単位行列を渡す | 合う．単位行列の Cholesky の因子は単位行列なので，係数は μᵀ そのものになる．試走の 7 単位 × 4 空間の 28 セルすべてで，K = 1 の LDA_K と NCM の判定が全く同じだった（S_K = 0 なので Σ_K = αI になり，α によらず NCM と同じ判定になる）．自己検査は α = 1 の LDA_T，α = 1 の LDA_K，NCM の一致と，K = 1 の LDA_K と NCM の一致を確かめる |
| EM_Amet：計量 A（α = 0.5）を固定し，K 本を one-hot に固定し，初期値は LDA_T の確率，β = 0，10 回．最後の割り当てと回ごとの候補の上の UAR を保存 | `qT0 = torch.softmax(P8.delta(Xp, fTr), -1)`，`em_kmetric(..., qT0, chols[A_RULE], yU=yp[Um])`，`poolpred[...emAmet...]`，診断の `emAmet_moved`，`emAmet_hist`，`emAmet_pool_hist`，`cos_emAmet_ldaO` | 合う．EM_Kmet と同じ関数を使い，第 1 回に em_fit との一致を確かめた手順をそのまま通る．試走で `hist` は 10 個，`pool_hist` は 11 個だった．`P_` の割り当てから K 本を除いて作った候補の上の UAR（K 本の位置は `DD.k_orders` で作り直した）は，`pool_hist` の最後と残差 0 で一致した |
| 比べ：LDA_T − NCM，LDA_K − NCM，LDA_SO − NCM，EM_Amet − LDA_T | `COMPARE` の `ldaT_ncm`，`ldaK_ncm`，`ldaSO_ncm`，`emAmet_ldaT` | 合う |
| 4 つの角の交互作用（EM_Amet − LDA_T）−（EM_Kmet − LDA_K）は点推定だけで，結果の文を書くときに作る | 台本には無い（`emAmet_ldaT` と `emKmet_ldaK` の行から作る） | 合う．区間が要るときは第 1 回の B の表と同じやり方で出せる．2 つの判定の数を足した塊の数を `effect_stats` に渡す |
| 規則：仮定 (a) LDA_SO − LDA_K，(a′) LDA_SO − NCM，(b) LDA_T − LDA_SO，Gn：LDA_T − NCM | `RULES` の `Aa`，`Aa2`，`Ab`，`Gn` | 合う．試走で，`diffs.csv` から数え直した数が 4 つのコーパスで `criteria.csv` の `_n_K` と `_n_hi_neg` に一致した |
| 部分集合：MOSEI の channel，AFEW の asr_ok，MELD の face_single．AVT の空間だけ | 集計の `subs`（`dai3.subset_masks` から 3 つの名前だけ取る），`SUB_COMPARE`，`diffs.csv` の `subset` の列 | 合う．試走の `diffs.csv` には AFEW の asr_ok，MELD の face_single，MOSEI の channel がそれぞれ 20 行（5 つの比べ × 4 K）あった．IEMOCAP には部分集合の行が無い．規則と分け方の選び出しは `subset == "all"` に絞ってある |
| MOSEI の channel の確かめ：G1・G2・G3 のどれかを全体で満たした K の数 `*_chan_n_K_ok` と，そのうち channel でも「下端 > 0 かつ全体の点推定の半分以上」の K の数 `*_chan_n_K_robust` | `CHAN_RULES`（G1，G2，G3n，G3j）と，764e3da の数え方 | 合う．735d952 の版は全体で満たさなかった K も数えていた．764e3da では DAI-8 の `okK`（`dai8_probe.py` の `summarize`）と同じく，全体で満たした K だけを数える．試走で，DAI-8 と同じ数え方で `diffs.csv` から数え直した 8 つの数が `criteria.csv` と一致した．MOSEI 以外の行ではこの 8 列が NaN になる（作りの上でそうなる） |
| IEMOCAP の添え書き：G1・G2・G3，EM_Amet − LDA_T，EM_Kmet − LDA_K について A と V の空間の同じ差を並べる | `diffs.csv` は全部の空間の比べを出す | 合う（集計の表を作るときに `space` が A と V の行を読む） |
| 感情ごとの再現率と判定の割合（記述）：`recall_units.csv` | `REC_KEYS` と `recrows` | 合う．設計の 9 つに加えて NCM と EM_Amet も出す．試走で 308 行，NaN 0 |
| `diag.csv` から fold と seed を外す（第 2 回の記録 4） | `DG.drop(columns=["fold", "seed"])` | 合う |
| λ_max の M̂ の中心，LDA_O の判定の扱い，`P_` と `k_orders`（第 2 回の記録 1〜3） | 設計 3 節と 5 節 | 合う．設計に書かれ，台本の動作と一致する |
| 起動の台本：`mkdir -p logs` を trap の前に移す | `dai10_probe.sh` | 合う．trap が `logs/` の無い状態で `touch` に失敗することが無くなった |

## A．小さく流した結果

- 自己検査：735d952 で `--selftest` が通った．764e3da の差分は集計の段だけなので，自己検査の部分は変わらない．
- 試走：単位は AFEW_IEMOCAP_f0_r0・f1_r0，IEMOCAP_MOSEI_f0_r0，IEMOCAP_MELD_f0_r0・f1_r0，IEMOCAP_AFEW_f0_r0・f1_r0 の 7 つである．各 c に 2 つの分割（MOSEI は 1 つ）があれば部分集合と区間の形を確かめられるので，この数にした．
  - 単位の段は 346100d で流した．735d952 から台本は変わっていない．
  - 集計の段は 764e3da で流した．
  - 出力は `/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad/dai10v3/` の下に書き，検査の後に消した．
- 例外と警告：単位の段と集計の段はどちらも終了コード 0 で，RuntimeWarning は 0 件だった．
  - 先に，各 c に分割が 1 つしかない 4 単位で集計の段を流したときは，IEMOCAP と AFEW の 780 行が NaN になり，同じ数の警告が出た．テストの塊が 1 つの分割だけでは，塊を 1 つ除く jackknife の値が作れないためで，試走の形による．第 2 回の 30 単位の試走でも，この 7 単位でも NaN は 0 だった．
  - 検査の途中で HEAD が 735d952 から d4c0aff に進み，コミットの違う 2 組の行で集計の段を流すと，集計の段が「行のコミットが 1 つでない」で止まった．第 2 回で足した照合が設計どおりに動いた．
- 照合：7 単位すべてで，K 本だけの LDA と全ラベルの LDA の不一致は 0 だった．
- 出力の形：1 単位に判定の鍵 428，`P_` の鍵 32（EM_Kmet と EM_Amet × 4 空間 × 4 K）．`diffs.csv` は 1,580 行で 26 種類．1 つのコーパスの全体の行は 380（23 の比べ × 4 空間 × 4 K ＋ ldaTblk_ldaT の 4 行 ＋ K = 0 の 2 つの比べ × 4 空間）で，部分集合の行が 20 ずつ足される．effect・lo・hi の NaN は 0．`criteria.csv` は 97 列で，NaN は MOSEI 以外の行の channel の 8 列だけだった．

## A．本番の規模の見積もり

- 単位の段：単位あたり AFEW 4 秒，IEMOCAP 5〜6 秒，MELD 8 秒，MOSEI 14 秒になった．EM_Amet で 10 回の EM が 1 つ増えたためである．120 単位でおよそ 16 分．
- 集計の段：MOSEI の行が約 400 行になり，塊の引き直しでおよそ 2 分．
- 合わせて 18〜20 分．
- GPU のメモリの最大は全体で 6.8 GB / 24.5 GB だった（表示の処理の 1.7 GB を含む）．
- ディスク：7 単位で 4.1 MB．120 単位でおよそ 80 MB．

## B．予想される指摘と，出力から答えられるか（第 2 回からの変化だけ）

| 表・図 | 予想される指摘 | (a)／(b)／(c) | 出し方 |
|---|---|---|---|
| G1 の表 | LDA_T と LDA_K のどちらが NCM からずれたのか | (a) | `diffs.csv` の `ldaT_ncm`，`ldaK_ncm` の行と，`criteria.csv` の `Gn_*` |
| 2 つの経路の表 | 割り当ての分を同じ計量で測れ | (a) | `emAmet_ldaT` と `emKmet_ldaK` の行．EM_Amet の割り当ては `P_K{K}_{s}_emAmet_a0.5` と `diag_units.csv` の `emAmet_pool_hist` にある |
| 4 つの角の交互作用 | 区間を付けよ | (a) | 第 1 回の B の表の交互作用と同じく，2 つの判定の数を足した塊の数を `effect_stats` に渡す |
| MOSEI の G1〜G3 | 漏れを除いた集計 | (a) | `criteria.csv` の `*_chan_n_K_ok`，`*_chan_n_K_robust` と，`diffs.csv` の `subset == "channel"` の行 |
| AFEW，MELD | 書き起こしの疑い，顔の競合を除いた集計 | (a) | `diffs.csv` の `subset == "asr_ok"`，`"face_single"` の行 |
| 感情ごとの表 | 感情ごとの再現率 | (a) | `recall_units.csv`（単位ごと．コーパスごとの平均は単位で平均する） |

## 指摘

### [I1] 比べ `emfromK_emAmet` が設計に書かれていない．この差には計量の違いと初期値の違いが混ざる
- 重さ：記録する
- 該当箇所：`dai10_probe.py`（764e3da）の `COMPARE` の `"emfromK_emAmet": ("K{K}_{s}_emfromK_a0.5_b0.0", "K{K}_{s}_emAmet_a0.5")`．設計の第 6 版には EM_fromK − EM_Amet が出てこない（`grep` で 0 件）
- 問題：EM_fromK と EM_Amet は 2 つの点で違う．計量（EM_fromK は回ごとの柔らかい群内の Σ，EM_Amet は A に固定）と，初期値（EM_fromK は LDA_K の確率，EM_Amet は LDA_T の確率）である．したがってこの差は計量だけの分ではない．`emfromK_emKmet` は初期値が同じなので，計量だけの差である．設計に書かれていない行が `diffs.csv` にあると，結果の文を書くときに「計量の分」と読まれるおそれがある．規則と分け方の数には入らないので，数値は変わらない．
- 計算の後で直せるか：直せる（行を読まないか，読むときに注を付ける）．
- 求める対応：設計 3 節に「EM_fromK − EM_Amet は計量と初期値の両方の違いを含む（記述）」と 1 行書くか，`COMPARE` から外す．

### [記録] `recall_units.csv` は単位ごとの値だけで，コーパスごとの表は作らない
- 重さ：記録する
- 該当箇所：集計の `recrows`
- 問題：設計は感情ごとの再現率を記述として並べると書く．台本は単位ごとの行を出すだけで，コーパス × 方法の平均と区間は作らない．単位ごとの行から作れるので，学び直しにはならない．
- 計算の後で直せるか：直せる．
- 求める対応：要るなら，結果の文を書くときに `recall_units.csv` を (c, key) で平均する．区間が要るときは `rows/*.npz` から感情ごとの塊の数を作って出す．

当てた項目で指摘の無いもの：I3（候補のラベルは LDA_SO・LDA_OT・LDA_Tcpi・縮みの係数・`pool_hist` の測定だけに入る．EM_Amet の E 段と M 段には入らない），I5（NCM，EM_Amet は LDA_T と同じ K 本と同じテストで解く），I6，I7（K = 1 の NCM と LDA_K の一致，P_ と pool_hist の一致），I9，I10，I11，I12（行のコミットの照合が試走で働いた），I13，I17，R1，R4，R6（NCM は凍結した特徴の上の少数ショットの標準の相手），R7（MOSEI の channel，AFEW の asr_ok，MELD の face_single），R8，R9，R10．

## 目録に無い種類の不備

なし．

---

試走の出力の消去：検査の後に，作業用のフォルダ `/tmp/claude-1000/-home-snics-MER-DA/b1f27254-32c7-45f0-ac5a-dee1c5e3ec8a/scratchpad/dai10v3` の下の試走の出力とログを消した．`results/dai10_probe/` は作っていない．値（UAR の差，規則の可否，channel の数）は画面にも報告にも出していない．数え直しは「一致したか」だけを出した．
