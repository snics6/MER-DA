# 再検査 3 体の指摘の直し（E83，2026-11-03）

対象：`checks/E83/code_recheck2.md`（止める 1，直してから進む 2，記録 3），`stats_recheck.md`（止める 1，直してから進む 1，記録 2），`define_recheck.md`（直してから進む 1，記録 6）．設計は `docs/exploratory-analyses.md` の E83 2〜5 節（6 節と E84 以降は触っていない），台本は `exp/analyze_pairs_e83.py`（1,333 行），台帳は `docs/seen-data.md` 27 行．コミットはしていない．作業木の `exp/analyze_projection_e85.py` の変更は私のものではなく，触っていない．

守ったこと：(A)(B) は流していない．動作の確認は (C) の試走（`--out <scratchpad>/e83_fix3 --stages c --targets MELD --boot 20 --smoke`．28 秒，RSS 0.74 GB，警告なし．出力は消した）と，(B) の経路・直接の差・P3a の関数を results/ を読まない人工データで動かした検算．`python -m py_compile` は通る．`results/` の値は開いていない．

行番号は直した後の版．

## 止める・直してから進む

| 指摘 | どこをどう直したか |
|---|---|
| code 1 [C1・D15]（止める），define 2 [D1・D14]，D19 | 台本 `direct_comparisons`（996〜1038 行）：`both_none`（両方が none）のときだけ `same_by_definition`，片方だけ none は `judge_unit` で判定し，`none_side` 列（a／b／both／空）を残す（1028〜1031 行）．`DIRECT` を 3 つ組（a，b，名前）に（68〜69 行）．`p3a_eval`（892〜980 行）：数の帰無と独立とみなした平均は観測で数えうる行（判定が same_by_definition でない行．`observable`，899 行）だけから作り，`n_tests_observable` 列を足した．人工データで oursx − abl_own（相手が全部 none）の 12 行が `none_side = b` で判定され（pos 6，same 6），`direct_both_chosen` は両方 none 以外の行だけ．設計 10082 行（D19 として明記）と 10104 行 |
| stats 1 [S10・D14・S2]（止める），define 7 [D14]，stats 3 [S2] | `p3a_eval` 905〜918 行：(iii) の対象を 6 対すべてが効果の出うるセルである単位（`full_units`）に限り，24 通りの置換を全部使う無条件の厳密な p（`p_class_perm_full_elig`）を主にした．到達しうる最小の p（`p_min_attainable`．対象の単位の怒りを含む対がすべて減らした，ほかは 0）と `judgeable`（p_min < 0.05），`P3a_verdict`（holds／fails／not_judgeable），`n_units_full_elig`，`full_elig_units_by_corpus`，`n_pos_full_elig`，`n_pos_elig_not_full`．前の条件つきの p は `p_class_perm_cond` の感度の列に残した．人工データで確かめた：u1（6 対が効果の出うる）の平静–怒りが減って p ＝ 0.5 のとき，怒りだけが効果の出うる u2 の減りを足しても p ＝ 0.5 のまま（前の版は 1.0 に上がった）．5 コーパス × 3 単位で怒りの対 1 つずつ → p ＝ 0.03125，p_min ＝ 0.00098．効果の出うるセルが平静–怒りだけの配置 → not_judgeable．設計 10106 行（(iii) の書き直し），10107 行（記述の列），10111 行（最小の型は対象の単位の中の値．(i) の条件：3 コーパスで 3 対ずつ程度．外れの文に並べる列），10118 行（文の規則 (2)：判定できない族はどちらの文も書かない） |
| stats 4 [S10・S13] | 設計 10104 行：「独立とみなしたときの塊の数の平均（期待値）であって上限ではない．独立なら 95% 点は 6」に直した．4 節 10141 行に和集合の区間の単位で帰無の 95% 点が小さい側にずれうることを書いた．台本の注（926 行） |
| define 1 [D12・D1] | `COMPARISONS` に abl_refb（単モダリティの f）・abl_ref0（融合の f）を足し（64〜65 行），無い f は `chosen_missing_expected` として記録して飛ばす（`PARTIAL_SELECTORS` 66 行，523 行）．`DIRECT` に abl_refb − abl_ownK，abl_ref0 − abl_ownK（68〜69 行）．provenance の `comparisons`・`direct`・`candidates` に入る．設計 10081 行，10082 行，文の規則 (3)（10118 行）：「帰せる」の条件に「候補の数が同じ abl_refb − abl_ownK（融合は abl_ref0 − abl_ownK）でも塊の数が帰無の 95% 点を超える」を足し，oursx − abl_ownK だけのときは候補の数の違い（3〜10 倍）を併記して oursx に有利な向きと書く．4 節 10145 行に abl_ownK との候補の数の差 |
| code 2 [L8・I1] | `CEBD64B_CHANCE`（70 行．音声・顔 144 × 0.025 ＝ 3.6，言葉 1.35，融合 3.6）を `p3a_table` から `p3a_eval` に渡し（992 行），`holds_old` はこの固定値で判定（948 行）．`cebd64b_chance` 列．設計 10113 行 |
| code 3 [S3] | `aggregate_unit` 660〜693 行：session_jk は主の区間（百分位 ∪ 話者の jackknife）とセッションの jackknife の和集合（規則 `union_main_session_jk`，689 行）．`lo_session_jk`・`hi_session_jk` 列（712 行）．人工データで IEMOCAP の幅の比が 1.00〜1.24（主より狭くならない）．設計 10084 行 |
| stats 2 [S1・S4] | `summarize` 1138〜1146 行 `with_replaced`：即興だけの判定の数の表と P3a の表は，主の行から AVT の IEMOCAP の行だけを `iemocap_impro_only` の行に差し替えて作る（1171 行）．人工データで感度の表の n_tests が主と同じ．設計 10088 行（AV の IEMOCAP を除かない理由） |

## 記録する

| 指摘 | どこをどう直したか |
|---|---|
| define 6 [D17・D7]：D を学ぶプールの n/d | `stage_b_run` 471〜477 行：（実行，ターゲット，f，反復）ごとにプールの本数（発話の数 − テストの数），`load_any(T, f)` の特徴の次元，n/d を `b_pool_nd.csv`（387 行）と provenance の `pool_nd`（389 行）に．設計 10091 行，4 節 10147 行（n/d が小さいと D が小さく出る向き） |
| define 5 [D8]：仮定の表の 1・3・5 行目 | 1 行目の測る量を「区分の違いは含まない」とし，印の数の行「1+2」を足した（1053〜1057 行）．3 行目の `level_fit_ratio`・`level_fit_ok` を `a_summary_K0.csv`・`a_summary_by_K.csv` に結合（`level_fit_table` 1040〜1046 行，1122 行）．5 行目は f ごとの数（1064〜1067 行）と，族ごと・印の有無ごとの `n_near_boundary`・`n_near_boundary_selected` を `b_p3_checks.csv` に（945 行）．設計 10123・10125・10127 行 |
| define 2 の D19 | 設計 10082 行に片方だけ none の単位の扱いを書いた（上の表） |
| define 3 [D16・D2]：再現率の差は切片を含む | 4 節 10146 行 |
| define 4 [D15]：逆の順序の効果の出うるセル | `pair_diffs_target` 566 行で none の逆の順序の損 100·[A(D, o_none) − A(d_none, o_none)] を出し，単位の行の `none_dirloss_rev`（717 行）．`p3a_eval` は metric が dir_rev の行でこれを効果の出うるセルに使う（896 行）．設計 10106 行と 4 節 10140 行 |
| code 5 [I9]：途中で落ちたとき | `main` 1300〜1301 行：本番は出力先が空でなければ止まる（provenance.json の有無ではなく）．確かめた（ファイルを 1 つ置いた出力先で止まる）．設計 5 節 10154 行 |
| code 6 [I15]：provenance の commit | 設計 5 節 10157 行（流した時の HEAD になる．`deps_sha256` で検査した版と照合し，6 節に検査したコミットと sha256 を書く） |
| code 4 [L8・L11]：台帳 | `docs/seen-data.md` 27 行に (5) として gate-code の 3 回目（1675073．06:50 開始，5 分 51 秒，検算を含めて 06:58 まで，`checks/E83/code_run/full/` と `t_degen`・`t_perturb`・`t_nanmis`・`t_diverge`，07:01 に削除）と，報告の指摘 4 に並ぶ見た値の範囲を書いた．「決めたこと」に再検査 3 体で直したことを足した |

## 入れなかったもの

- gate-stats の再検査の S15・S16，gate-code の再検査の C14・S15，gate-define の再検査の D19・D12 の追記（`docs/methods.md` 9.3 節の目録）は入れていない（範囲は設計と台本．番号は本体が振る）．
- gate-define の D16 の「切片だけを直した感情ごとの再現率」は出さない（4 節に読み方を書いた）．
- gate-stats の再検査の S10・S13 (2)（jackknife の標準誤差で幅を合わせた帰無）は後から出せる感度として 4 節に書くにとどめた．
- gate-code の確認の台本（`check_recheck2.py`・`check_fix2.py`・`check_p3a2*.py`）は，`DIRECT` が 3 つ組になり，`p3a_eval` の引数（`old_chance`）と列，`aggregate_unit` の列（`lo_session_jk` など），`stage_b_run` の引数（`nd_rows`）が変わったので流し直しが要る．

## 確かめたこと（人工データ．`results/` を読まない）

- (iii)：対象の単位を限った p は「減らした」を足しても上がらない（0.5 → 0.5．前の版は 1.0）．5 コーパス × 3 単位で 1 対ずつ p ＝ 0.03125（設計の最小の型と一致），p_min ＝ 0.00098．効果の出うるセルが 1 対だけの配置は not_judgeable．前の条件つきの p は `p_class_perm_cond` に残る．
- 直接の差：相手が全部 none の oursx − abl_own は `none_side = b` で判定され，`direct_both_chosen` から外れる．abl_refb − abl_ownK の行が出る．両方 none の行は `n_tests_observable` と塊から外れる．
- session_jk の幅は主の 1.00〜1.24 倍（主より狭い行は無い）．`none_dirloss_rev` に NaN 0．即興だけの表の母集団は主と同じ．
- (C) の試走：13,950 行，J_dir の NaN 0，provenance に比較 7 つと直接の差 6 つ．空でない出力先で止まる．

## 本番の規模の見込み

比較が 7 つ（abl_refb・abl_ref0 は半分の f だけ），直接の差 6 つ，プールの次元の読み込み（f ごとに `load_any`）が増えたので，gate-code の 3 回目の 5 分 51 秒に対して 7〜9 分，RSS 2 GB 以内と見込む．設計 5 節の見込みを「10 分以内」に直した（10154 行）．GPU は使わない．

## git diff --stat（HEAD c60746e．E83 の 3 本だけ．コミットはしていない）

```
 docs/exploratory-analyses.md | 111 ++++++++++++++---------
 docs/seen-data.md            |   4 +-
 exp/analyze_pairs_e83.py     | 211 ++++++++++++++++++++++++++++++-------------
 3 files changed, 221 insertions(+), 105 deletions(-)
```
