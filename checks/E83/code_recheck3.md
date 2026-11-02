# 関門 1：実装と出力の再検査 3（E83，2026-11-03，コミット 1b775e2）

検査の対象：`exp/analyze_pairs_e83.py`（1b775e2．sha256 56f687d9…），設計 `docs/exploratory-analyses.md` の E83 1〜5 節（1b775e2），`docs/seen-data.md` 27 行，直しの記録 `checks/E83/fix_applied_3.md`，差分 `git diff 1675073 1b775e2 -- exp/analyze_pairs_e83.py docs/exploratory-analyses.md docs/seen-data.md`．全規模の流しは 1b775e2 の台本で行った（出力の先頭の行が `script_sha256=56f687d944d177f0`，`dirty_deps=False`）．

**作業木の変更（検査の途中で見つけた）**：07:30 に，台本（sha256 4551e97e…）と設計 3・4 節に未コミットの変更が入った（`p3a_eval` の `m_ref`，`not_judgeable_by_flag` など．gate-stats の再検査 2 への対応とみられる）．この報告の判定は 1b775e2 についてのものである．作業木の変更は人工の配置で 1 回流しただけで，全規模では確かめていない（指摘 1 と 4）．

## 判定
直してから通す．前回の 3 体の止める 2 件と直してから進む 4 件はすべて台本に入った．全規模の出力から独立に計算し直した P3a の表 208 行は食い違い 0 で，E79 との照合も前回と同じ精度で通った．ただし P3a (iii) の印を除く版（顔の主の版）では，対象の単位が「印を除いた後に 6 行そろう単位」で決まる．このため設計 3 節の「対象は none の損だけで決まり，手法の結果に依らない」に反し，`p_min_attainable` と `judgeable` も印で動く．全規模では対象の単位が顔で 3 のうち 3，音声で 5 のうち 4 が印で外れた．作業木の未コミットの変更がこの点を扱っているので，コミットしてからその差分を再検査し，通れば本番を流す．

## 前回の指摘の入り方

| 出所と番号 | 指摘 | 直した箇所（1b775e2 の行） | 確かめ方 | 結果 |
|---|---|---|---|---|
| code_recheck2 1 [C1・D15]（止める），define_recheck 2 [D1・D14] | 片方だけ none の単位を「定義から同じ水準」にしていた | `direct_comparisons` 996〜1038．`both_none` のときだけ same_by_definition，`none_side` 列 | 全規模の直接の差 2,430 行で，点・区間（ブートストラップの差の百分位と jackknife の差の t 区間の和集合）・判定・none_side を台本の関数を使わずに計算し直した（`check_direct3.py`）．人工の配置でも確かめた（`check_synth3.py` 4 節） | 直った．食い違い 0．片方だけ none の（単位，対）は 276（oursx−abl_own 234，abl_ref0−abl_ownK 18，ほかの 4 つは 6 ずつ）．そのうち same_by_definition 0，missing 0，point ≠ 0 は 276/276．両方 none の行は今のデータでは 0．人工の配置で両方 none は same_by_definition になり，n_tests_observable 0，塊 0，帰無は NaN |
| stats_recheck 1 [S10・D14・S2]（止める），define_recheck 7 [D14] | (iii) の参照の集合が「減らした」に依っていた | `p3a_eval` 892〜980．6 対すべてが効果の出うる単位に限り 24 通りの置換を全部使う．`p_min_attainable`，`judgeable`，`P3a_verdict` | 人工の配置 8 例で，台本の値，コーパスごとの置換の直積を列挙した値，手で計算した値の 3 つを比べた．無作為の配置 300 個で台本と直積を比べ，対象の単位の怒りの対に「減らした」を 1 つ足したときに p が上がらないかを確かめた | 直った．8 例すべて一致．例：1 コーパスで p ＝ 0.5（怒りだけが対象の単位の「減らした」を足しても 0.5 のまま），5 コーパスで 1 対ずつ p ＝ 0.03125・p_min ＝ 0.25^5 ＝ 0.00098，3 コーパスで 2 対ずつ 0.015625，顔の 3 コーパスで (1/6)^3 ＝ 0.00463，2 コーパスなら p_min ＝ 0.0625 で判定できない．無作為の 300 配置で食い違い 0，p が上がった配置 0．p_min が実際に最小であることは次の理由で成り立つ．怒りを固定する 6 通りの置換は怒りの対の数を変えないので，コーパスごとに p ≥ 6/24 になる（顔は 4/24） |
| define_recheck 1 [D12・D1]（直してから進む） | 候補の数が同じ abl_refb・abl_ref0 を比べていなかった | `COMPARISONS` 64〜65，`PARTIAL_SELECTORS` 66，`DIRECT` 68〜69 | 全規模の格子 | 直った．abl_refb は単モダリティの f の 57 単位，abl_ref0 は融合の f の 24 単位．どちらも chosen.csv の n_cand は 20．無い f は `chosen_missing_expected` が 81 行（abl_ref0 57，abl_refb 24）．`chosen_missing` は 0 |
| code_recheck2 2 [L8・I1]（直してから進む） | cebd64b の顔の数の基準 | `CEBD64B_CHANCE` 70，`p3a_eval` 948 | 全規模の表の `cebd64b_chance` と `P3a_holds_cebd64b` を計算し直した | 直った（音声・顔・融合 3.6，言葉 1.35．食い違い 0） |
| code_recheck2 3 [S3]（直してから進む） | session_jk の区間が主の区間を含まなかった | `aggregate_unit` 672〜692 | 全規模の 756 行で，`b_jk_units.npz` の session_jk の鍵（G ＝ 5）から t 区間（自由度 4）を計算し直し，主の区間との和集合と比べた | 直った．セッションの区間の食い違い 0，和集合の食い違い 0，主より狭い行 0．幅の比は 1.00〜1.64（幅 0 の 66 行はすべて none を選んだ単位） |
| stats_recheck 2 [S1・S4]（直してから進む） | 即興だけの表の母集団が主と違った | `summarize` 1138〜1146（`with_replaced`），1171 | 全規模で，主の行から AVT の IEMOCAP の行だけを差し替えた表を自分で作り，感度の表 56 行を計算し直した | 直った．族ごとの n_tests は主と同じ（差 0）．session_jk の表も同じ |
| code_recheck2 4〜6，define_recheck 3〜6・D19 | 台帳，途中で落ちたときの扱い，provenance の commit，n/d，仮定の表，逆の順序の効果の出うるセル | `stage_b_run` 471〜477，1053〜1067，566・717・896，1300〜1301 | 全規模の出力を読んだ | 直った．`b_pool_nd.csv` は 340 行（34 鍵 × 10 反復），重複 0，NaN 0．仮定の表に「1+2」の行がある．`a_summary_*` の `level_fit_ok` に NaN 0．逆の順序の表は `none_dirloss_rev` で効果の出うるセルを決める（全規模の計算し直しで一致） |

## A．設計と台本の対応表

変わった項目だけを書く．前回の表（`code_recheck2.md`）のほかの行は今回も合っている（E79 との照合と全規模の計算し直しで確かめた）．

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 比較相手 7 つ（abl_refb・abl_ref0 を足した） | 64〜66，519〜523 | 合う |
| 直接の差 6 つ．両方 none だけ same_by_definition．none_side | 68〜69，996〜1038 | 合う |
| (i) の帰無と独立の平均は観測で数えうる行から | 925〜936 | 合う |
| (iii) の対象は 6 対すべてが効果の出うる単位．その中で 24 通りの無条件の厳密な p | 906〜918 | 印を含む版では合う．**印を除く版では対象が印で決まり，設計の「手法の結果に依らない」に反する（指摘 1）** |
| `p_min_attainable`・`judgeable`・`P3a_verdict` | 908〜912，957 | 同上 |
| 記述の列（`p_class_perm_cond`，`n_near_boundary*`，`n_pos_elig_not_full`） | 938〜975 | 合う |
| cebd64b の固定の基準 | 70，948 | 合う |
| session_jk は主の区間とセッションの jackknife の和集合 | 684〜692 | 合う |
| 即興だけ・session_jk の表は主の行から差し替えて作る | 1138〜1146，1171〜1173 | 合う |
| `b_pool_nd.csv` | 471〜477，386〜389 | 合う |
| 空でない出力先で止まる | 1300〜1301 | 合う（前回の確かめのまま） |

設計の表に載る組（世界 2 × f × K 3 × コーパス，比較 7，感度 6，直接の差 6）は出力にすべてそろった（下の格子）．

## A．小さく流した結果と全規模の照合

依頼のとおり，評価のターゲットの値を出す (A)(B) を (C) と合わせて全規模で流した．出力は `checks/E83/code_run/full/` と `t_degen`・`t_perturb`・`t_nanmis`・`t_diverge` に置き，照合の後の 07:31:08 にすべて消した．判定の値（減らした／増やしたの数，P3a〜P3c の当否，対ごとの点，厳密な p の値）はこの報告に書かない．

- **コマンド**：`./.venv/bin/python -W default exp/analyze_pairs_e83.py --out checks/E83/code_run/full`（既定の B ＝ 1,000，stages ab c）．07:20:16 開始，**4 分 37 秒**，最大 RSS 1.72 GB，CPU 100%，警告 0，終了 0．別の台本（`analyze_projection_e85.py --smoke`）が同時に CPU を使っていた．
- **照合の台本**（`checks/E83/code_run/`）：`check_recheck2.py`（E79 との照合・格子．出力だけを読むので前回のまま使えた），`check_p3a3.py`（新しい規則の P3a の全列を台本の関数を使わずに計算し直す．厳密な p は直積の列挙），`check_direct3.py`（直接の差，session_jk，checks.csv，b_pool_nd，provenance），`check_synth3.py`（人工の配置．`results/` を読まない），`check_fix2.py`（退化・照合の効き・選択の食い違い．`stage_ab` の引数と戻りが変わっていないので前回のまま使えた）．
- **格子**：b_units_pairs.csv は 13,716 行．比較 5 つ × (main・δ 半分・δ 2 倍・rev_order 各 486 ＋ iemocap_impro_only 216 ＋ session_jk 126) に，abl_refb（342・162・90）と abl_ref0（144・54・36）を足した数になる．鍵の重複 0．point・lo・hi・delta_d_pair・delta_d_pair_corpus・none_dirloss の NaN・Inf 0，missing 0，n_boot_valid の最小 1,000．区間の規則は union_pct_jk 6,912，pct 6,048，union_main_session_jk 756．b_units_direct.csv は 2,430 行（486 × 4 ＋ 342 ＋ 144），重複 0，NaN 0．b_pairs_rep 36,720 行と b_recall_rep 24,480 行は重複 0・NaN 0．複製の鍵 1,314（NaN の行 0），jackknife の鍵 774（G は 5・10・254）．
- **P3a の表の計算し直し**（`check_p3a3.py`）：主 8 行，感度 56 行，比較相手の main 48 行，直接の差の direct_all・direct_both_chosen 96 行の計 208 行で，次の列がすべて一致した（食い違い 0）：n_tests_observable，n_pos_blocks，独立の平均，帰無の 95% 点，count_threshold，cond_count，n_pos_corpora_selected_elig，cond_two_corpora，n_units_full_elig，full_elig_units_by_corpus，n_pos_full_elig，n_pos_elig_not_full，観測の数，p_class_perm_full_elig，p_min_attainable，judgeable，cond_p，cebd64b_chance，P3a_holds_cebd64b，P3a_verdict，n_near_boundary(_selected)，n_eligible．
- **保存済みの値の再現（E79）**：P5-3（6 対の平均）は main 81，rev_order 81，iemocap_impro_only 36 の単位がすべて対応した．点の差の最大は 2.9e−15・2.4e−15・2.9e−15，複製の百分位の差の最大は 5.1e−7・5.8e−7・5.1e−7，AUC 版 3.4e−14，model_err_diff 4.0e−15．none − kd・groupB・abl_ownK・abl_own を E79 の rows の dirloss から作った値と比べると，差の最大は 2.2e−15〜2.8e−15．複製は oursx 9.5e−7，groupB 1.2e−6（float32 で保存した分の誤差）．直接の差 oursx − abl_own・abl_ownK は E79 の ablation_replace_dir と点で 2.7e−15・2.9e−15，百分位で 5.8e−7 以内で一致した．δ の材料は E79 の a_d0 と 2.8e−14・4.3e−14 で一致した．δ 半分・2 倍の行の δ は main のちょうど 0.5・2.0 倍．
- **(A)**：反復の行 9,180，重複 0，NaN 0．平均の行 918 はすべて 10 反復．(B) の none_dirloss と (A) の差の最大 0.0．
- **退化した場合**：全比較の手法の行を none の行に替えて MELD と IEMOCAP で流した（1 分 50 秒）．反復の行 26,460 で diff_dir・diff_dir_rev・diff_dir_auc・diff_model_err・diff_o がすべて 0.0 になった．単位の行 8,946 では point・lo・hi がすべて 0.0，判定はすべて same，印はすべて False（rev_order は NaN）．session_jk の 882 行も同じだった．
- **照合の効き**：pairs の d に 1e−5 × (row_id mod 7) を足すと「最大 2.10e−04」で止まった．手法の行の d を 6 本に 1 本 NaN にすると「NaN の食い違い 1」で止まった．台本の群の audio K5 の選択だけを none に替えると，主の比較では止まった．補助の比較では checks.csv に 9 行を書いて続き，単位の点は (即興の点 ＋ 0)/2 に一致した（比較 5 × 実行 2 で差の最大 4.4e−16）．照合の台本の最後の行が 0.669 を出したのは，照合の台本が 2 つの実行を同じ対の鍵で混ぜて平均したためで，実行ごとに分けて計算し直すと差の最大は 4.4e−16 だった．
- **(C)**：52,560 行，重複 0，J・J_dir の NaN 0．P3c の行 486（teacher_g：ref 300，ref0 162，pF 12，欠け 12）．
- **provenance**：commit 1b775e2（検査したコミットと同じ），dirty_deps False，deps_sha256 5 本（台本 56f687d944d1…），inputs 1,130，comparisons 7，direct 6，pool_nd 34 鍵，コードの写し `code/20261103-072016/`（5 本と空の git_diff_HEAD.patch）．全 CSV の先頭に `# E83 commit=…` がある．
- **作業木の版での参考の流し**（未コミット，sha256 4551e97e…，07:33）：人工の配置（`check_synth3.py`）の 1・2 節は 1b775e2 と同じ値を出した．顔の 3 単位のうち 1 単位の平静–喜びに印を付けた配置で `p3a_table` を通すと，印を除く版は n_units_full_elig 3，n_units_full_elig_used 2，n_full_elig_dropped_by_flag 1 になった．p_min_attainable 0.00463（印を含む行で決まる），p_min_attainable_after_flag 0.0278 で，作業木の設計の文と合う．全規模では流していない．

## A．本番の規模の見積もり

(A)(B)(C) を合わせて 4 分 37 秒，最大 RSS 1.72 GB（CPU だけ．別の台本と並走した）．設計 5 節の「10 分以内，RSS 2 GB 以内」に収まる．GPU は使わない．作業木の変更（印を含む行での対象の決め方）で増える計算は置換の列挙 1 回分で，所要の差は数秒と見込む．

## B．予想される指摘と，出力から答えられるか

前回の表（`code_recheck.md`，`code_recheck2.md`）の (a) の行は引き続き (a) である．変わった行と新しい行だけを書く．学び直し（GPU）を要する行は無い．

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 直接の差 | 候補の数が同じ相手でも教師の寄与は残るか | D12，C2 | (a) `b_units_direct.csv` の abl_refb−abl_ownK・abl_ref0−abl_ownK の行と，`b_p3_checks_by_comparison.csv` の同じ比較の行 | — | — |
| 直接の差 | 片方だけ none の単位を除いたら | C14（案），C1 | (a) `none_side` が空の行に絞る（`direct_both_chosen` の表がこれに当たる） | — | — |
| P3a (iii) | 対象を絞らずに条件つきの置換で検定したら | S15（案） | (a) `p_class_perm_cond` | — | — |
| P3a (iii) | 印を除く版で，印で外れた対象の単位はいくつか．印を含む版ならどうか | D8，S15（案） | (a) `b_units_pairs.csv` の model_dependent・none_dirloss・delta_d_pair・chosen から CPU 数秒で数えられる（作業木の版は `n_full_elig_dropped_by_flag` 列を出す） | — | — |
| P3a (i) | 和集合の区間の単位で，帰無の幅を jackknife の標準誤差に合わせたら | S10・S13（stats_recheck） | (a) `b_boot_units.npz` と `b_jk_units.npz` から CPU で作れる | — | — |
| (B) | セッションの塊で区間を作ったら（IEMOCAP） | S3 | (a) session_jk の行と `lo_session_jk`・`hi_session_jk` | — | — |
| (A)(B) | D を学ぶプールの n/d が小さい f で D が小さく出ていないか | D17 | (a) `b_pool_nd.csv`（n/d の最小 0.41） | — | — |
| 再現率 | 切片だけを直した感情ごとの再現率 | D16 | (b) E83 の出力には無い．保存済みの対数の確率（`results/e77_*/<T>_r<r>_logp.npy`）から CPU で出せる | `pair_diffs_target` に切片を直した argmax の版を足す | CPU 数分，約 2 MB |

## 指摘

### [D14・S15（案）] P3a (iii) の印を除く版では，対象の単位を印を除いた後の行で決めている．このため対象と到達しうる最小の p が手法の結果に依り，顔の主の版の判定できるかどうかを印が決める
- 重さ：直してから進む
- 該当箇所：`exp/analyze_pairs_e83.py:908`（1b775e2）
  ```python
  full_units = {u for u in set(ukey) if int((ukey == u).sum()) == 6 and bool(elig[ukey == u].all())}
  ```
  と `p3a_table`（同 982〜989 行．989 行）の `m = m[m.model_dependent != True]`．設計 `docs/exploratory-analyses.md` 3 節 (iii)（1b775e2）：「(iii) の対象は，6 対すべてが効果の出うるセルである単位（…none の損だけで決まり，手法の結果に依らない）」．同 3 節：「主にする版：音声の f は印を含む版，顔の f は印を除く版」．
- 問題：`p3a_table` は印を除く版で印の付いた行を落としてから `p3a_eval` を呼ぶ．印の付いた対が 1 つでもある単位は 5 行以下になり，`== 6` の条件で対象から外れる．印（`model_dependent`）は手法の点と d_AUC 版の点の差と模型の誤差の差から作る量なので，手法の結果に依る．このため，印を除く版では次の 3 つが設計の文と合わない．(1) 対象の単位．(2) `p_min_attainable` と `judgeable`．設計はこの 2 つを「配置だけで決まる」量として扱う．(3) 文の規則 (2) の「この配置では判定できない」．印で判定できなくなった族にもこの文が付く．
  - 人工の配置（`check_synth3.py` 3 節，読んだうえで流した）：顔の 3 コーパスに 6 対すべてが効果の出うる単位を 1 つずつ置き，平静–怒りが「減らした」とする．印が無ければ n_units_full_elig 3，p ＝ p_min ＝ 0.00463 になる．1 単位の平静–喜び（平静–怒りではない対）に印を 1 つ付けると，印を除く版は n_units_full_elig 2，p ＝ p_min ＝ 0.0278 になる．
  - 全規模（none − oursx の main）：6 対すべてが効果の出うる単位は，印を含む行で数えると音声 24 単位のうち 5，顔 24 単位のうち 3 だった．このうち印の付いた対を含み，印を除く版で対象から外れるのは音声 5 のうち 4，顔 3 のうち 3 だった．顔の主の版は印を除く版なので，顔の (iii) が判定できるかどうかは印で決まる．
- 計算の後で直せるか：直せる（CPU 数秒）．ただし，判定できるかどうかの規則を値を見てから決めたと読まれる．
- 求める対応：計算の前に，対象の単位，`p_min_attainable`，`judgeable` を印を含む行で決め，印を除く版で印によって外れた単位の数と，外れた後の到達しうる最小の p を別の列に出す．そのうえで「印で判定できない」を「配置で判定できない」と別の文にすることを，設計 3 節の (iii) と文の規則 (2) に書く．作業木の未コミットの変更（07:30．`p3a_eval` の `m_ref`，`n_full_elig_dropped_by_flag`，`p_min_attainable_after_flag`，`judgeable_after_flag`，`P3a_verdict` の `not_judgeable_by_flag`，設計 3・4 節の追記）はこの形になっている．人工の配置では，作業木の版がその設計の文どおりに動いた（上の「参考の流し」）．印の付いた単位を丸ごと外す作りは，置換の帰無の 6 つの位置がすべて観測できるという前提を保つ．印の付いた行だけを「減らした」に数えない作りにすると，観測では「減らした」になりえない位置に置換が「減らした」を写すので，作業木の作りのほうがよい．コミットした後，その差分（`p3a_eval`，`p3a_table`，`summarize` の表の列，設計 3・4 節）を gate-code に再検査させる．再検査では `check_p3a3.py` の対象の決め方を作業木の規則に合わせて流し直す．

### [I1] 比較相手の表に，その比較が存在しない族の空の行が 24 行ある
- 重さ：記録する
- 該当箇所：`exp/analyze_pairs_e83.py` `p3a_table`（982 行〜）と `summarize` の `byc`（1180〜1189 行）
- 問題（全規模で確かめた）：`b_p3_checks_by_comparison.csv` で，abl_ref0（融合の f だけ）の音声・顔・言葉の族と，abl_refb（単モダリティの f だけ）の融合の族の行が n_tests 0 で出る．none−abl_ref0 が 6 行，none−abl_refb が 2 行，abl_ref0−abl_ownK が 12 行，abl_refb−abl_ownK が 4 行で，計 24 行．帰無は NaN，塊は 0 になる．落ちずに動き，判定にも使わない．ただし表をそのまま並べると「塊 0」と読まれうる．
- 計算の後で直せるか：直せる
- 求める対応：n_tests 0 の行を書き出さないか，`exists` のような列を足す．どちらもしないなら，6 節に「abl_refb・abl_ref0 の行は存在する族だけを読む」と書く．

### [L8・L11] gate-code の 4 回目の全規模の流しと，見た値の範囲を台帳に書く
- 重さ：記録する
- 該当箇所：`docs/seen-data.md:27`
- 問題：依頼に従い，2026-11-03 07:20:16〜07:24:33 に (A)(B)(C) を全規模で `checks/E83/code_run/full/` に流した．07:25〜07:31 に照合し，検算として MELD と IEMOCAP の (B) を `t_degen`・`t_perturb`・`t_nanmis`・`t_diverge` に流した．07:31:08 にすべて消した．私が見た値は次に限る．
  - E79 との照合の差，行・鍵の数と格子，比較ごとの same_by_definition の行の数（abl_own 234，abl_ref0 18，abl_ownK・groupB・kd 6）
  - 片方だけ none の（単位，対）の数と，その 276 行の判定の 4 区分の内訳．内訳の数は照合の台本の出力で見たが，ここには書かない
  - n_seeds の値，区間の規則の行の数，session_jk と主の幅の比（1.00〜1.64）
  - P3a の表の構造の列：帰無の 95% 点の範囲（主 1〜4，感度 0〜5，比較 0〜4），独立の平均の範囲，n_blocks，n_tests の値，NaN の出る列とその数
  - 指摘 1 の対象の単位の数（音声 5 のうち印で外れる 4，顔 3 のうち 3）．この数から，主の表のどの行が判定できないかが分かる．主の表の `p_min_attainable` の NaN は 2 行だった
  - b_pool_nd の次元と n/d の最小 0.414，checks.csv の行の種類，P3c の teacher_g の内訳，退化した場合の判定（すべて same）
  - 台本の値と私の計算し直しの値が一致したかの真偽
  
  減らした／増やしたの数（主の比較），P3a〜P3c の当否，対ごとの点，厳密な p の値は見ていない．一致の確かめは台本の中で比べ，食い違いの数だけを出した．作業木の版は人工の配置だけで流した．
- 計算の後で直せるか：直せる（記録だけ）
- 求める対応：`docs/seen-data.md` の E83 の行に (6) として上の流しと見た値の範囲を書く．特に，指摘 1 の数（顔の対象の単位が印ですべて外れること）を見たうえで作業木の規則が書かれたのではないことを，作業木の変更の時刻（07:30）と私の照合の時刻（07:25〜07:31）を並べて記録する．私はこの数を本体にも作業木にも伝えていない．

### [I15] 本番の前に作業木の変更をコミットし，provenance の sha256 と検査した版を突き合わせる
- 重さ：記録する
- 該当箇所：作業木の `exp/analyze_pairs_e83.py`（sha256 4551e97e…，07:30:01）と `docs/exploratory-analyses.md`
- 問題：このまま本番を流すと `dirty_deps` が True になり，検査した 1b775e2（56f687d9…）とも違う版で計算される．作業木には E85 の台本の変更も混ざっている（`exp/analyze_projection_e85.py`．E83 の台本は読み込まないので `dirty_deps` には入らない）．
- 計算の後で直せるか：直せない（流した版が検査した版と違えば，検査が効かない）
- 求める対応：E83 の 2 本をコミットし，差分の再検査が通ったコミットで本番を流す．6 節に検査したコミットと台本の sha256 を書く．

当てた項目：I1〜I17（I2〜I5 は新しい学習と選択が無いので，読み込みが `load_any` を通り sample_id の並びを照合すること，選択を chosen.csv から読むことだけを確かめた．I13 は当てはまらない），R1〜R13，S1〜S4・S9〜S14，D1・D8・D12・D14〜D19，C1〜C4・C8・C10〜C12，L8・L10〜L12．

## 目録に無い種類の不備

- S17（案）：置換検定の対象（参照の集合）を，手法の結果から作る印（模型に依る印など）で絞るときは，「配置で判定できない」と「印で判定できない」を別の列と別の文にする．到達しうる最小の p は印を使う前の対象で計算し，印の後の値を並べる．印の付いた行だけを「当たり」から外し，単位を対象に残す作りは使わない．置換は観測で当たりになりえない位置に当たりを写すので，帰無の分布がずれる（E83 の関門 1 の再検査 3）．
- I18（案）：検査役が全規模の流しを始めた後に，作業木の台本が書き換えられることがある．検査役は報告の前に台本の sha256 を取り直し，流した版（出力の先頭の行の `script_sha256`）と比べて，違えば報告の冒頭に書く（E83 の関門 1 の再検査 3 で，照合の途中の 07:30 に作業木が変わった）．
