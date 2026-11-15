# 計算の後の点検：E82 の解析（`results/e82_analysis/reselect/`・`compare/`，2026-11-02）

対象：選び直した版の段 delta・select・measure・judge・judge_b（`results/e82_analysis/reselect/`）と比べの段（`results/e82_analysis/compare/`．`--fixed-from results/e79_p4`），台本 `exp/analyze_e82.py`（sha256 先頭 dfa4969c3992cc8d）と `exp/analyze_e79.py`（09f3a345b9455249），ログ `logs/e82a_*.log`，入力 `results/e77_<実行>/`・`results/e78_<実行>/`・`results/e82_<実行>/`．設計は `docs/exploratory-analyses.md` の E82 の 2〜5 節．前の検査は `checks/E82/code.md`（「計算の後の点検で照らす数」の節を使った）．

評価のターゲットの判定の値は点検に使っていない．数え上げの途中で `judgments.csv`・`decision.csv` の先頭の行が 1 回画面に出たが，この報告には書かず，判断にも使っていない．検算の台本は `checks/E79/code_run_post/` にある（`count_e82_inputs.py`，`scan_inputs.py`，`verify_inputs.py`，`diff_units.py`，`count_select.py`，`count_rows.py`，`scan_boot.py`）．固定の版（`results/e79_p4/`）の点検は `checks/E79/code_p4_post.md` にある．

## 判定

通す．設計の範囲（5 つの実行，種類の 3 生徒と 2 つの融合の生徒，K 0・5，3 手法，倍率 1・3・10，none の倍率の対照，版，感度）の行は欠けなく重複なくそろい，provenance の連鎖も切れておらず，入力に 0 バイトや読めないファイルは無い．比べの段の表は設計 3 節の規則を当てるのに足りる列と行を持つ．設計 2 節の記述の列「出発点からの離れ」だけが比べの段の表に無い（前の検査の [I1-dist] のまま．保存した出力から CPU で出せる）．

## 1．入力（E77・E78・E82 の結果フォルダ）

- 0 バイトのファイル：E77 5 フォルダ（3,489 本），E78 5 フォルダ（4,349 本），E82 5 フォルダ（2,755 本）で 0 本．
- 全ファイルを開いた（`scan_inputs.py`）：npy はヘッダの形 × 要素の大きさとファイルの大きさが一致，npz は zip の CRC が一致，csv・json・jsonl は全部読めた．読めないファイルは 0 本．中身が空の csv は E78 の世界の `<T>_noise.csv` 10 本だけで，Co-teaching を回さなかったことによる（`checks/E79/code_p4_post.md` の 5 節）．E82 は読まない．
- E82 のファイルの種類ごとの数：各実行で T ごとに `.csv`・`_pseudo`・`_conv`・`_convp`・`_cp`・`_diag`・`_dist`・`_e82check`・`_samples`・`_settings` が 1 本ずつ，反復 0〜9 の `_W_`・`_b_`・`_wrows_`・`_logp`・`_pred`・`_rowids`・`_te` が 10 本ずつ，仮のターゲットの none の npz が 580 本（A 200，V 140，T 50，AVT 50，AV 140）．
- 前の検査で決めた照らす数（`checks/E82/code.md` の 170〜178 行）：

| 照らす数 | 期待 | 実際 |
|---|---|---|
| 各 `<T>.csv` の行 | 反復 10 × 246 ＝ 2,460．合計 66,420 | 27 本すべて 2,460，合計 66,420．重複 0．反復 10．entmin・IM は種 0・1・2，self は種 0，none の倍率 3・10 がある |
| 各 `<T>_pseudo.csv` | 仮のターゲットの単位あたり 118 行．合計 68,440 | 合計 68,440．重複 0．仮のターゲットごとに反復 5 |
| 仮のターゲットの none の npz | 580 | 580．delta の段で E77 と照らし，食い違い 0，対数の確率の差の最大 0.0 |
| `<T>_e82check.csv` | すべて ok | 1,700 行すべて ok |
| `<T>_convp.csv` の cp | 倍率ごとに 1 つの値 | （ファイル, 生徒, 倍率）81 組すべてで 1 つ |
| `check.csv`（`--check`） | 食い違い 0 | A 740，V 603，T 359，AVT 359，AV 603 行がすべて ok |
| provenance | 522be57 以降，dirty・allow_dirty が偽，GPU が E77 と同じ型 | 5 実行とも cec9e40（522be57 の子孫），`provenance_log.jsonl` の全記録も cec9e40・dirty 偽，`NVIDIA H100 80GB HBM3`（E77 と同じ）．`exp/s43_penalty.py` の sha256 は c5a99690…で，再検査した 4b27fc1 の版と同じ．全 csv の commit 列も cec9e40 だけ |

## 2．選び直した版の段（`reselect/`）の数え上げ

| 段・表 | 設計の範囲 | 出力 | 合っているか |
|---|---|---|---|
| delta | E79 と同じ δ | `delta.csv`・`delta_units.csv`・`delta_reps.csv` が `results/e79_p4/delta/` とバイト単位で同じ（sha256 fbc4d022…，`e82_provenance` の `e79_delta.same` も真） | 合う |
| select `chosen.csv` | E79 と同じ格子．entmin・im・self の種類と融合の生徒の K 0・5 だけ候補を「倍率 1・3・10 の和」に替える | 4,434 行，重複 0，候補なし 142（すべて LP の K=0）．範囲の内の 444 行で候補の数は entmin 12・IM 36・self 36，範囲の外（K=25 と世界の単モダリティ）は 4・12・12 のまま．ほかの 8 族の行は E79 の p4 と方法・設定・点数・候補の数・同点の数が全部一致 | 合う |
| select `fixed.csv` | E79 と同じ | E79 の p4 と内容が同じ（sha256 c8557bb8…） | 合う |
| measure の鍵 | E79 の鍵 ＋ 倍率の none（none_cp）を（T, 生徒, K 0・5, 倍率 3・10）ごと | none_cp の鍵：A 28，V 24，T 16，AVT 16，AV 24（＝ T の数 × 4）．`keys_*.csv` の重複 0 | 合う |
| measure の複製と表 | E79 と同じ | boot 27，bootcl 15，jk 16，jkcl 10．全部で列の数が鍵の数と一致，NaN 0，Inf 0．`rows_*`：反復 10，反復ごとの行の数が同じ，重複 0，Inf 0，UAR の NaN 0，0〜1 の外 0．`D_*`・`model_null_*` は E79 の p4 とバイト単位で同じ | 合う |
| measure の「働かなかった」の材料 | E82 から選んだ設定を同じ C′ の none と比べる（設計 2 節） | `checks.csv` の `e82_inactive_material_added` が 27 の T すべてで 40〜150 行．3 手法の P4・P4′ の 2,088 単位で `inactive_a` の NaN 0 | 合う |
| judge `units.csv`（比べ (a)） | E79 と同じ単位 | 77,838 行，重複 0．E79 の p4 の 79,518 行との鍵の差は E79 の側だけにある E80 の段 c の教師の参照の行 1,680 で，ほかの鍵は全部一致（E82 は段 c を読まない．`checks/E79/code_p4_post.md` の [L-e80c]） | 合う |
| judge_b `units.csv`（比べ (b)） | 同じ単位．倍率 1 の単位と K=0 では (a) と同じ | 77,838 行，重複 0．(a) と点の値か区間が違う行は 1,635 行で，すべて K=5，すべて 3 手法の単位で，すべて選んだ倍率が 1 でない単位（MOSEI・MELD 以外の ns_excl の行は excl の選択を使う）．逆に，倍率が 1 でない K=5 の単位はすべて (a) と違う．K=0 で違う行は 0 | 合う |
| combos・statements | E79 と同じ形 | combos 24,904 行（(a)・(b) とも），重複 0．statements 176 行（22 の tag × 2 族 × 2 subset × 2） | 合う |

## 3．比べの段（`compare/`）の表が設計 3 節の規則に足りるか

| 設計 3 節の項目 | 表と列 | 行の数 | 足りるか |
|---|---|---|---|
| 文の鍵を手法ごとに作り，族ごとの鍵も並べる | `statements.csv` の scope（family，entmin，im，self）× tag × 族 × subset，`rule_fixed`・`rule_a`・`rule_b`，`statement_key_*`，`combos_*` | 242 行，重複 0，rule の NaN 0．unseen_only の行は族の鍵と IM の P4 だけにある（entmin・self は両族で，IM は P4′ で「見た後の予言」の手法なので，1 手法の unseen_only の組み合わせが無い．設計の `SEEN` どおり） | 足りる |
| 判断は subset = all，unseen_only と (2′) の same_combos は記述 | `decision.csv` の `rule_*`，`same_combos_a_desc`・`same_combos_b_desc`，`unseen_only_rules_desc` | 手法 × 族 6 行 ＋ 手法ごと 3 行 ＋ 3 手法のまとめ 1 行 | 足りる |
| same_a・limit・k_label・opposite の分け方（強さ (3)<(4)<(2′)<(2)<(1)） | `decision.csv` の `dir_a`・`dir_b`・`category_keys_only`・`category`・`category_text`，`statements.csv` の `dir_*`・`category` | 同上．鍵の NaN 0 | 足りる |
| 10 倍が過半なら限定を外さない（母数は手法が選ばれた単位，excl） | `multipliers.csv` の `n_chosen`・`n_mult10`・`majority_mult10`（同点を除いた `*_strict` も），`decision.csv` の `edge10`・`edge_note`・`majority_mult10_excl`・`n_groups_majority_mult10_excl` | multipliers 184 行（実行 × 生徒 × K × 手法 × 版 120，族 × K 48，手法 12，全体 4） | 足りる |
| 3 手法 × 族 × K ごとの判定の数（固定・(a)・(b)） | `judgments.csv`（族 × 手法 × K × 感度 × 量 × 版） | 648 行 ＝ 3 版 × 216．単位の数は感度ごとに P4 14，P4′ 8（IEMOCAP 台本を除く行は 3） | 足りる |
| 単位ごとの 3 つの判定と変わったか | `judgment_units.csv` | 2,088 行 ＝ 3 手法 × （P4 438 ＋ P4′ 258），重複 0，判定の NaN 0，相手の無い行 0 | 足りる |
| 倍率ごとに選ばれた単位の数（（実行，生徒，K，手法）ごとにも） | `multiplier_units.csv`，`multipliers.csv` | 444 行（範囲の内の選択の行と同じ数） | 足りる |
| 同点で倍率の小さい方を選んだ単位の数 | `multiplier_units.csv` の `n_tie_recomputed`・`tie_mults`・`tie_multi_mult`・`tie_with_lower_mult`・`mult_string_order`・`tie_chose_lower_mult`・`recomputed_matches_chosen` | 選択の作り直しの食い違い 0 | 足りる |
| 候補の数（entmin 4 → 12，IM・self 12 → 36） | `candidates.csv` | 56 行．3 手法 × 族 × K で固定 4・12・12，選び直し 12・36・36（最小＝最大） | 足りる |
| oracle_dir を 2 つの版で格子の大きさを添えて | `oracle_dir_both.csv`（`grid_size`），`oracle_dir_summary.csv` | 1,790 行 ＝ 895 × 2，46 行 | 足りる |
| 選び直さない手法の文には限定を残す | `decision.csv` の `scope_note` | 1 行 | 足りる |
| 両方の版で (4) のときの書き方 | `decision.csv` の `both4_note` | — | 足りる |
| 検算（ほかの 5 手法は 3 つの版で同じ，K=0 で (a)＝(b)，文の鍵の作り直し，選択の作り直し，条件の一致） | `e82_provenance.json` | すべて 0．`statements_recheck.csv` は 3 版とも 88 行で食い違い 0．measure の条件（boot 1000，seed 0，反復 0〜9，T すべて，切片あり，δ の sha256）は固定と選び直しで同じ | 足りる |
| 解きの収束（前の検査の [I11-conv]） | `convergence_e82.csv` | 90 行．反復の上限に当たった問 0，早く止まった問 0 | 足りる |
| 記述の列：選んだ設定の出発点からの離れを倍率ごとに（設計 2 節） | なし | — | 足りない（指摘 [I1-dist]） |

## 4．provenance の連鎖

| 段 | コミット | 開始 | E79 の台本 | E82 の台本 | dirty | allow_* |
|---|---|---|---|---|---|---|
| delta | 3727cde | 14:38:10 | 09f3a345… | dfa4969c… | 偽・空 | 偽 |
| select | 24197c1 | 14:40:15 | 同 | 同 | 偽・空 | 偽 |
| measure | c4d05b6 | 14:42:31 | 同 | 同 | 偽・空 | 偽 |
| judge | c4d05b6 | 15:07:30 | 同 | 同 | 偽・空 | 偽 |
| judge_b | c4d05b6 | 15:11:25 | 同 | 同 | 偽・空 | 偽 |
| compare | c4d05b6 | 15:15:19 | 同 | 同 | 空 | 偽 |

- 今の 2 つの台本の sha256 は上と同じで，`exp/analyze_e82.py` は再検査（`checks/E82/code.md` の再検査の節）で見た版（dfa4969c3992cc8d），`exp/analyze_e79.py` は E79 の再検査を通った版である．`script_tracked` は真．
- δ の sha256 先頭 fbc4d022… は 3727cde で文書に入り，select の開始より前．`chosen.csv` c6a60dca5e4f91c8 は c4d05b6（14:41:48）で文書に入り，measure の開始（14:42:31）より前．`fixed.csv` c8557bb8… は 24197c1 で文書に入っている．measure・judge・judge_b の provenance の chosen・fixed の sha256 は今のファイルと全桁で一致した．
- compare の段の `fixed_inputs_sha256.chosen` は `results/e79_p4/select/chosen.csv`（3bdc21b8…）と一致し，`reselect_a_inputs_sha256`・`reselect_b_inputs_sha256` の chosen は c6a60dca… である．
- 入力：provenance に書かれた入力（delta 607 件，select 189 件，measure 1,053 件）を今のファイルと照らし，食い違い 0，無いファイル 0 だった．E79 の p4 と共通の入力ファイルの記録はすべて同じ値である．
- E82 の場所は E79 の 3 つ目の波の枠（e80c）に入れて読んだ．本物の `results/e80c_*` は読んでいない（provenance の `e80c_dirs` は E82 の場所だけ）．

## 5．ログ

- `logs/e82a_delta.log`・`_select.log`・`_measure.log`・`_judge.log`・`_judge_b.log`・`_compare.log` がそろい，例外とメモリ不足は無い．`logs/e82a_mj.done` は 0．compare のログの最後の行は検算の食い違いがすべて 0 であることを示し，台本は表を書いてから止まる条件（`analyze_e82.py:784-787`）に当たっていない．
- 警告は E79 の p4 と同じ 2 種類（`Mean of empty slice` と `DtypeWarning`）で，保存した複製に NaN は無く，表の数に影響しない（`checks/E79/code_p4_post.md` の 4 節）．
- judge_b の none の鍵の置き換えは 828 回．置き換えの範囲は 2 節の (a)・(b) の差の照合で確かめた．

## 指摘

### [I1-dist] 比べの段に，選んだ設定の出発点からの離れを倍率ごとに並べる表が無い
- 重さ：記録する（関門 2 の前に出しておくことを勧める）
- 該当箇所：`exp/analyze_e82.py:543-723`（`stage_compare` に離れの集計が無い），設計 2 節の「記述の列」
- 問題：設計 2 節は「選んだ手法の解の出発点からの離れ（‖W − W_出発点‖_F ÷ ‖W_出発点‖_F）を倍率ごとに残す．査読の前提『罰則を弱めると重みが動く』を確かめるため」とする．比べの段の 11 本の表にこの列は無い．前の検査の [I1-dist] で「計算の後に CPU で出す」とした扱いのままである．
- 計算の後で直せるか：直せる．E82 の `<T>_dist.csv`（倍率 3・10 と E39 の none．Z の座標）と，E77・E82 の `<T>_r<r>_W_<生徒>.npy`・`_b_`・`_wrows_`（E77 は none と entmin・IM・self の全設定の W を保存している．AFEW の反復 0 で none 3，entmin 36，IM 114，self 36 行を確かめた）から，評価のターゲットの生の特徴の座標で倍率 1・3・10 をそろえて出せる．学び直しは要らない．仮のターゲットの倍率 1 は W が保存されておらず出せない
- 求める対応：結果を文書に書く前に，`multiplier_units.csv` の選んだ設定ごとに，評価のターゲットの生の特徴の座標の離れ（基準は K=0 の none の行）を倍率ごとに出す CPU の台本を足し，その表を設計 2 節の記述の列として載せる．載せないなら，2 節から記述の列を外し，その理由を書く．

### [記録-sens] `judgments.csv` に MOSEI と AFEW を除いた集計の行が無い
- 重さ：記録する
- 該当箇所：`exp/analyze_e82.py:552-561`（`units.csv` の sensitivity で数える．`excl_mosei_afew` は combos・statements の tag にだけある）
- 問題：文の鍵（`statements.csv`）には `main+excl_mosei_afew`・`delta_half+excl_mosei_afew` の行があるが，判定の数の表には無い．感度の表に「MOSEI と AFEW を除いた判定の数」を載せるときに表が無い．
- 計算の後で直せるか：直せる（`judgment_units.csv` のコーパスの列で MOSEI と AFEW を除いて数え直すだけ）
- 求める対応：感度の表に載せるなら，`judgment_units.csv` から数え直す．

## 目録に無い種類の不備

なし（E79 の p4 の報告の「段の途中のコミット」は，E82 の各段では起動と終了の間にコミットが入っておらず，当たらない）．
