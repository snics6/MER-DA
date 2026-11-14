# 関門 1：実装と出力 第 3 回（DAI-18，2026-11-14，コミット 02f356e）

検査した版：台本 `DAI/exp/dai18_probe.py`（コミット 02f356e．作業木の sha256 290bf037bd1976216535fae1d5c20e8b069ebaeb00f2560365b7920b5e008cb0 と，`git show 02f356e:` の sha256 が同じことを確かめた），起動の台本 `DAI/run/dai18_probe.sh`．差分は `git diff 87b1dc7 02f356e -- DAI/exp/dai18_probe.py DAI/run/dai18_probe.sh` の 5 行（台本 4 行，起動の台本 1 行）である．`DAI/exp`・`measure`・`features` のほかのファイルは 87b1dc7 から変わっていない．検査の途中で HEAD が e493c47 に進んだが，その差分は設計の文書だけで，台本と起動の台本は変わっていない（`git diff 02f356e HEAD --stat -- DAI/exp DAI/run measure features` が空）．
試走の置き場は `DAI/checks/DAI-18/code_run_v3/`．`results/dai18_probe/` は作っていない（今も無い）．`results/dai1/eval/` は開いていない．`DAI/exp/` に新しいファイルを置いていない．UAR の値，版の優劣，選ばれた点と λ の値はこの報告に書かない．

## 判定

通す．直した `select_points` は，c ごと・K ごとの 18 組のすべてで，`MS_eq_sel` を `MN_eq_sel` と同じ単位の集まり（同じ `n_units`，同じ `pseudo_targets`，同じ（T, c, 分割, 種, K）の鍵の集合）で選び，ほかの版の選択と集計の出力は 1 行も変わらず，RAVDESS の生徒の `M_F6_mixsel` の 3 行だけが `students_sel.csv` から消え，止める処理は全組がそろった出力では止まらず欠けを作ると止まり，起動の台本は `SHA18` が無いときも空のときも `.failed` を出した．

## A．直した点の対応表

| 依頼・前回の指摘 | 台本の箇所 | 合っているか |
|---|---|---|
| (1) `MS_eq_sel` の λ を `MN_eq_sel` と同じ単位で選ぶ（`probe_v2.md` 指摘 1，設計 第 3 版 5 節「得の帰し方」） | `dai18_probe.py:596-597` | 合う．`MN_eq_sel` の候補（`MN_eq_c{0..7}_l{0..2}`）は，RAVDESS を仮のターゲットにした単位では c4〜c7 が無い（`cfgs_of`，`:82-85`）ので `dropna` で落ちる．`MS_eq_sel` には同じ規則（仮のターゲットのモダリティが 2 つ以上）で同じ単位を落とす行が足された．c＝RAVDESS では選ぶ単位の仮のターゲットがすべて 2 つ以上のモダリティを持つので，この行は何も落とさない |
| (2) RAVDESS の生徒の `M_F6_mixsel` の行を出さない（`code_v2.md` 記録 2） | `:841-842` | 合う．`exp_combos`（`:850`）と同じ条件で飛ばす |
| (3) `SHA18` が無いときに `.failed` を出す（`code_v2.md` 記録 1） | `DAI/run/dai18_probe.sh:9` | 合う |
| 設計 第 3 版 4 節（読み方）「版の格子の点の選択は RAVDESS の単位を除くので 4 つのうち 1 つが演技」，8 節「版の格子の点の選択には CREMA-D の単位だけが入る」 | `:596-597` と `dropna` | 合う．試走の `sel.csv` の `pseudo_targets` で，c が 2 つ以上のモダリティを持つ 15 組は 4 コーパス（RAVDESS を含まない），c＝RAVDESS の 3 組は 5 コーパス |

## A．小さく流した結果

研究室の PC（RTX A5000）．流す前の GPU の使用は 1.4 GB，使用率 0% で，ほかの計算は動いていなかった．検算の台本と出力はすべて `DAI/checks/DAI-18/code_run_v3/` にある．

**(1) `select_points` の単位の集まり（`check_v3.py`，CPU，数秒．ログ `log_check_v3.txt`）**

第 2 回の試走の units の 30 単位（分割 0・種 0，コミット 87b1dc7）の csv に，新しい `select_points` と旧版（`git show 87b1dc7:` を `code_run_v3/dai18_probe_old.py` に写したもの）を当てた．

| c | K | `MS_eq_sel` の n_units 旧 | 新 | `MN_eq_sel` の n_units | pseudo_targets の一致 |
|---|---|---|---|---|---|
| AFEW・IEMOCAP・MOSEI・MELD・CREMA-D（各 K 5・10・25） | | 5 | 4 | 4 | 15 組すべて一致（RAVDESS を含まない 4 コーパス） |
| RAVDESS（K 5・10・25） | | 5 | 5 | 5 | 3 組すべて一致（5 コーパス） |

- 18 組すべてで `n_units` と `pseudo_targets` が一致した．文字列の一致に加えて，選ぶ単位の鍵（T, c, 分割, 種, K）の集合そのものを取り出して比べると，`MS_eq_sel`・`MN_eq_sel`・`MN_F2K_sel`・`MN_F6K_sel` の 4 つで 18 組すべて同じだった．
- 本番の規模では，落とす規則が仮のターゲットのモダリティの数だけで決まり分割と種によらないので，`n_units` は c が 2 つ以上のモダリティを持つ組で 4 × 10 ＝ 40，c＝RAVDESS で 5 × 10 ＝ 50 になる（読んだうえでの計算．本番の規模では流していない）．

**(4) 直しでほかの出力が変わらないか**

- 選択（`check_v3.py`）：選択の鍵は新旧ともに 156 個．違う鍵は 5 個で，すべて `MS_eq_sel`（c は AFEW・CREMA-D・MELD）．`MS_eq_sel` 以外の 138 行（`MS_eq_tied`・`M_F6n_mixsel`・`MN_F6orc_sel` の結びつけた行を含む）は全列が完全に同じ．c＝RAVDESS の `MS_eq_sel` の 3 行も旧版と同じ．旧版の選択は第 2 回の `frozen_sel.json` と同じだった（前提の確かめ）．
- 集計（`check_summary_v3.py`，旧版と新版の `stage_summary` を同じ 30 単位に `allow_partial=False` で当てた．54 秒．ログ `log_check_summary_v3.txt`，`log_compare_v3.txt`）：

  | ファイル | 行（旧 → 新） | `MS_eq_sel` を含まない行 | `MS_eq_sel` を含む行 |
  |---|---|---|---|
  | `sel.csv` | 156 → 156 | 138 行が完全に同じ | 18 |
  | `levels.csv` | 309 → 309 | 291 行が完全に同じ | 18 |
  | `effects.csv` | 846 → 846 | 738 行が完全に同じ | 108 |
  | `criteria.csv` | 39 → 39 | 33 行が完全に同じ | 6 |
  | `per_emotion.csv` | 996 → 996 | 924 行が完全に同じ | 72 |
  | `neighbors.csv`，`convergence.json` | 同じ | — | — |
  | `students_sel.csv` | 420 → 417 | 旧だけの行は（RAVDESS, K, A, `M_F6_mixsel`）の 3 行．ほかは同じ | 60（同じ） |
  | `students_levels.csv`・`students_effects.csv`・`students_criteria.csv`・`students_conditions.csv`・`students_checks.json` | 477・537・108・417 行，同じ | — | — |

  生徒の層の出力が同じなのは，この比べ方では生徒の npz が第 2 回の（旧い選択の）ものだからである．生徒の段の変わり方は次で確かめた．
- 生徒の段（新版で T＝IEMOCAP の分割 0・種 0 の 5 単位を流し直した．4 分 44 秒．置き場 `code_run_v3/stud/`，ログ `log_students_IEMOCAP.txt`，比べた台本 `check_students_v3.py`・`check_students_rows_v3.py`）：
  - 新版が書いた `frozen_sel.json` は `check_v3.py` の新しい選択と一致した．
  - npz の配列 4,102 個のうち 3,895 個がビット単位で一致した．選択が変わった鍵（c と K）の `MS_eq_sel` の生徒の配列は 216 個で，そのうち 205 個が変わった．それ以外で違った配列は 2 個（`IEMOCAP_MELD_f0_r0` の `pred_K25_AVT_MN_F2K_sel_g10` と `pred_K5_V_MN_F6K_sel_g11`）で，どちらも約 2,100 本のうち 1 本だけ違う（一致 0.99952）．
  - csv では，`MS_eq_sel` 以外の行で違ったのは，(a) 単位ごとの勾配の最大を書く `_check` の行の `grad`（差の最大 9.3×10⁻⁹），(b) `IEMOCAP_MELD_f0_r0` の `MN_eq_sel`・`MS_eq_tied`・`MN_F2K_sel`・`MN_F6K_sel` の 5 行の `uar`・`uar_pool`・`mimic`（`mimic` の差の最大 3.3×10⁻⁴，1 本ぶん）だけである．c の選択が変わらなかった 2 単位（c＝MOSEI，RAVDESS）では，`MS_eq_sel` の行も含めて `grad` 以外のすべての数値が同じだった．(b) は第 2 回の記録 3 と同じ現象（同じ問を一度に解く束の中身が `MS_eq_sel` の疑似ラベルの入れ替わりで変わると，止める基準の近くで 1 本ずれる）である．
  - 疑似ラベルの作り直しと保存の一致 `agree_saved` は 102 行（うち新しい `MS_eq_sel` 15 行）すべてで 1.0，勾配の最大 1.0×10⁻⁶，止まり 0，Inf 0．GPU の最大 1.8 GB．警告なし．

**(2) 止める処理（`check_summary_v3.py` と `check_stop_v3.py`．ログ `log_check_stop_v3.txt`）**

- 新版の `stage_summary` を 30 単位（`all_units` を分割 0・種 0 に絞って数えを 30 にした）に `allow_partial=False` で当てると，止まらずに 13 個のファイルを書いた．RAVDESS の `M_F6_mixsel` を `sch` から除いても，`exp_combos` は前から同じ組を除いているので，`missing` は空のままである．
- 欠けを作ると止まった．c＝CREMA-D の格子の点を選ぶ単位（T＝CREMA-D の 5 単位）の生徒の csv から `MN_eq_sel` の行を除いた写しで流すと，「生徒の格子の点が選べない組がある：[('CREMA-D', 5, 'A', 'MN_eq_sel'), …]（計 9）」で止まった．

**(3) 起動の台本（読んだうえで，同じ先頭 10 行の写しを `code_run_v3/shtest/` で流した．`cd` の先だけを一時の置き場に替え，11 行目に `echo GUARD_PASSED` を置いた）**

| `SHA18` | 終了コード | `logs/` にできたもの |
|---|---|---|
| 渡さない（`env -u SHA18`） | 1 | `dai18_probe.failed`（「SHA18 が渡されていない」） |
| 空（`SHA18=`） | 1 | `dai18_probe.failed`（同上） |
| 違う値 | 1 | `dai18_probe.failed`（「sha256 が違う」） |
| 290bf037… | 0 | なし．`GUARD_PASSED` まで進んだ |

作業木の起動の台本の先頭 10 行は 02f356e の版と同じだった．本物の `logs/` には書いていない．

selftest は新版で通った（`log_selftest.txt`）．

## A．本番の規模の見積もり

第 2 回の見積もり（合計約 8〜9 時間，GPU の最大は 9 GB 前後で上限 14.4 GB の内，ディスク約 2 GB）から変わらない．直しは選ぶ単位を減らすだけで，計算の量を増やさない．今回の生徒の 5 単位は 4 分 44 秒で，第 2 回の T ごとの 4 分 10 秒〜5 分 58 秒の範囲に入る．

## B．予想される指摘と，出力から答えられるか

第 1 回・第 2 回の表から変わった行だけを書く．

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| `effects.csv` の `MN_eq_sel − MS_eq_sel` | 2 つの版の λ は同じ仮のターゲットで選ばれたか | `probe_v2.md` 指摘 1，C（比較の公平さ） | (a)：`summary/sel.csv` の `MS_eq_sel` と `MN_eq_sel` の行の `n_units`・`pseudo_targets` を c・K ごとに比べる．試走で 18 組すべて一致した | — | — |
| `students_sel.csv` | RAVDESS の生徒に無い疑似ラベルの選択が載っていないか | 読み方 | (a)：行が無い．試走で 417 行（`students_conditions.csv` と同じ数） | — | — |
| `MS_eq_sel` の選択 | RAVDESS の単位を含めて選んだら（第 2 版の選び方なら）点は変わるか | 感度 | (a)：`units/*.csv` の `MS_eq_l*` の `uar_adj` から，旧版の `select_points` の式（`:595` の後の 2 行を除いたもの）で CPU で選び直せる．生徒の学び直しは範囲の外（設計 8 節の「選び直して点が変わった組の生徒の学び直し」と同じ扱い） | — | — |

計算の後に学び直しを求められる不足は，今回の差分の範囲では見つからなかった．

## 指摘

### [I（再現）] `MS_eq_sel` の疑似ラベルが入れ替わると，ほかの疑似ラベルの生徒の判定が 1 本ずれることがある
- 重さ：記録する
- 該当箇所：`dai18_probe.py:668-669`（生徒を束で解く箇所．読んだだけ．現象は流して確かめた）
- 問題：T＝IEMOCAP の 5 単位を新版で流し直すと，選択が変わっていない `MN_F2K_sel`・`MN_F6K_sel` の生徒の配列 2 個が 1 本だけ違い（一致 0.99952），csv の 5 行の `uar`・`uar_pool`・`mimic` が 1 本ぶん違った．第 2 回の記録 3 と同じ現象である．本番は 1 回の起動ですべての段を同じ選択で流すので，本番の数値の正しさには影響しない．第 2 回の試走の生徒の出力と本番の出力を突き合わせると，`MS_eq_sel` 以外の行でも違いが出る．
- 計算の後で直せるか：直せる（直す必要はない）
- 求める対応：計算の後の点検で生徒の配列を作り直して比べるときは，一致の基準を 0.995（`NONE_AGREE`）にする（第 2 回の記録 3 のとおり）．

### [I（記録）] 本番の結果に書かれるコミットは 02f356e ではなく，その後のコミットになる
- 重さ：記録する
- 該当箇所：`dai18_probe.py:940`（`dai1.git_commit()`）
- 問題：検査の途中で HEAD が e493c47（設計の文書だけの変更）に進んだ．今回の試走の生徒の csv と `frozen_sel.json` にも e493c47 が書かれた．台本の sha256 は 290bf037… のままである．計算の後の点検で「結果ファイルのコミット ＝ 関門 1 で検査したコミット」を比べると一致しない．
- 計算の後で直せるか：直せる（数値に影響しない）
- 求める対応：計算の後の点検では，`provenance.json` の `this_script_sha256` が 290bf037… であることと，02f356e から結果のコミットまでの差分に `DAI/exp`・`measure`・`features`・`DAI/run` が入っていないことを確かめる．

### [I（確かめ）] 集計は `MS_eq_sel` と `MN_eq_sel` の選ぶ単位が同じことを自分では確かめない
- 重さ：記録する
- 該当箇所：`dai18_probe.py:596-597`
- 問題：直しは「仮のターゲットのモダリティが 2 つ以上」という規則で単位を落とし，`MN_eq_sel` の側は候補の欠け（`dropna`）で落ちる．今の格子（`cfgs_of`）ではこの 2 つが同じ単位を落とすことを試走の 18 組で確かめたが，台本は両者の一致を確かめて止める処理を持たない．今の版では数値に影響しない．
- 計算の後で直せるか：直せる
- 求める対応：計算の後の点検で，`summary/sel.csv` の `MS_eq_sel` と `MN_eq_sel` の `n_units`（40 と 40，c＝RAVDESS は 50 と 50）と `pseudo_targets` が 18 組すべてで一致することを確かめる．

## 目録に無い種類の不備

- 候補案（I の項目）：**比べる 2 つの版の選び方を別の規則で合わせたときは，一致を台本の中で確かめる．**片方は候補の欠けで単位が落ち，もう片方は明示の条件で単位を落とす形では，格子が変わると 2 つの単位の集まりが黙ってずれる．選んだ後に 2 つの版の単位の鍵の集合が同じかを確かめて止める行を置くか，計算の後の点検で `n_units` と仮のターゲットの一覧を比べる．

## 第 3 回の検査の範囲

当てた項目：差分の 5 行（`select_points` の単位の絞り込み，`stage_summary` の生徒の選択の飛ばし，起動の台本の `SHA18` の確かめ），直しが届く出力（選択，集計の 13 ファイル，生徒の段の配列と csv），止める処理，selftest．差分の外の台本の部分は読み直していない（第 1 回・第 2 回の検査のとおり）．
