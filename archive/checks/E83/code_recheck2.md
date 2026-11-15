# 関門 1：実装と出力の再検査 2（E83，2026-11-03，コミット 1675073）

検査の対象：`exp/analyze_pairs_e83.py`（1675073．sha256 b0d612fe…．HEAD の d9e2bc8 でも台本と読み込む 4 本は同じ内容，`git diff HEAD` は空）と，設計 `docs/exploratory-analyses.md` の E83 1〜5 節（10062〜10157 行），`docs/seen-data.md` 27 行．前回の再検査 `checks/E83/code_recheck.md` と，本番の 4 体（stats・define・compare・leak）の指摘が直ったか（`checks/E83/fix_applied_2.md`）を確かめた．

## 判定
止める．P3a の作り（感情の名前の置換の厳密な p，条件つきの並べ替え，複製の帰無の 95% 点，広がりの条件）は設計どおりに実装され，E79 の保存した値も再現したが，手法どうしの直接の差（oursx − 相手）で相手だけが none を選んだ 252 の（単位，対）を，差が 0 でないのに「定義から同じ水準」と判定しており（設計 10082 行の定義の誤り），文の規則 (3) の「両方が none 以外の単位に限っても」が働かないので，設計の 1 文と台本の 1 か所を直し，その差分を再検査してから本番を流す．

## 前回と本番の 4 体の指摘が直ったか

| 出所と番号 | 指摘 | 直した箇所（台本の行） | 確かめ方 | 結果 |
|---|---|---|---|---|
| code_recheck 1 [S4・S11] | (i) が (ii) から自動で成り立つ | `null_block_counts` 843–858，`p3a_eval` 875–880 | 全規模の出力から，台本の関数を使わずに塊の数・独立の上限・帰無の 95% 点・`count_threshold`・`cond_count` を計算し直した（`check_p3a2.py`） | 直った．主 8 行・感度 56 行・比較相手と直接の差 96 行のすべてで一致．塊の数 × 0.025 は判定から外れた |
| code_recheck 2 [D14] | (iii) の分子と分母が別の集合 | 865–866，884，887，889 | 同上．人工の例（喜び–怒りを効果の出うるセルの外に置く）で n_pos 9，n_pos_elig 6，n_pos_not_elig 3 を確かめた | 直った．分子は「減らした かつ 効果の出うるセル」，外の数は `n_pos_not_elig`・`n_pos_not_elig_selected` に出る |
| code_recheck 3 [R13] | 「テストの上の J は保存されていない」 | 設計 10144 | 読んだ | 直った（e77 の teacher の行にあることと，出さない理由を書いた） |
| stats 1・define 1 [S10・D14] | 感情の名前の置換（24 通り）と厳密な p | 72，812–840，884–886 | 下の「厳密な p」 | 直った．gate-stats・gate-define の例の値に一致 |
| stats 2 [D14・S2] | 条件つきの並べ替え | 812–840（`conditional=True`） | 人工の例 3〜5 を直積で数えた | 直った．効果の出うるセルが怒りの対だけなら p ＝ 1（置換を引けたコーパス 0） |
| stats 3 [S10・S11] | 数の基準を複製の帰無の 95% 点に | 843–858，876–880 | 人工の複製で，台本の関数と独立の書き方の差 0.0．全規模でも一致 | 直った |
| stats 4 [S2] | 広がりは怒りの対で数える | 881–883 | 全規模で計算し直して一致 | 直った（`n_pos_corpora_selected_elig`） |
| stats 6 [S3] | セッションの jackknife | 124，489–493，605–607，653–659 | 全規模：IEMOCAP の 630 行，規則 union_pct_session_jk，自由度 4 | 設計どおり．ただし主の区間との和集合ではない（指摘 3） |
| stats 7・code_recheck 8 [I15・S9] | テストの集合が異なる数 | 463–469 | provenance と checks.csv | 直った．IEMOCAP-impro 5，IEMOCAP-script 6，MELD 9，ほか 10（両方の実行） |
| compare 2・3 [C2・C4] | abl_ownK・groupB，種の平均 | 64，68–89，272–273，567–614 | E79 の rows の dirloss・boot の agg の鍵と照合 | 直った（下の照合） |
| stats 9・define 9・compare 1 [S14・C8] | 直接の差 | 935–970 | E79 の ablation_replace_dir と照合 | 点・区間は直った．相手が none の単位の判定に誤り（指摘 1） |
| leak 1 [L8] | cebd64b の規則での当否 | 893–894 | 設計 10113 と照合 | 顔の数の基準が設計と違う（指摘 2） |
| define 8 [D8] | 仮定の表 | 973–1028 | `assumptions.csv` の行 | 直った（1〜8 の行が出る．5 は 2 行） |
| code_recheck 5 [I9] | dirty の範囲，CSV のコミット | 113–118，1216–1220 | 全規模の provenance と CSV の先頭行 | 直った．`dirty_deps` は台本が読む 5 本だけ（False），未追跡の `exp/analyze_pair_routing_e86.py` は `untracked_exp_measure` に分かれた．全 CSV の先頭に `# E83 commit=…` |
| code_recheck 6 [I8・I12] | 試走の出力先，段の分け流し | 1208–1212 | 読んだ．`--smoke` の版を `checks/` に流した | 直った．試走は results/ に書けず，本番は provenance.json があれば止まる |
| code_recheck 7 [R4] | 感度の P3a の表 | 1079–1091 | 計算し直して一致 | 直った（7 つの感度 × 8 行） |
| code_recheck 9 [R1・R7] | 性別・E81・第 2 のエンコーダ | 設計 10146 | 読んだ | 直った（範囲の外と書いた） |
| code_recheck 4 [I8・L8] | 台帳 | seen-data 27 | 読んだ | 直った．今回の流しの分は指摘 5 |

## A．設計と台本の対応表

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| (A) 融合の実行は e79/measure，種類の実行の none は e79_p4/measure | 346–363 | 合う（source 列で分かれ，重複 0） |
| (A) A(D,0)，none の向きの損，d_AUC 版，模型の誤差，分散の比，負の反復の数 | 313–327，370–376 | 合う |
| (B) 81 単位 × 6 対，比較 5 つ | 502–533 | 合う（比較ごとに main 486 行） |
| (B) 種のある手法は種 0〜2 の点と複製を平均 | 567–614 | 合う．E79 の `boot_*.npz` の agg の鍵と同じ（groupB の複製の差の最大 1.2e−6） |
| (B) none を選んだ群は差 0，全群 none は same_by_definition | 555–565，679 | 合う |
| (B) 主の比較で即興と台本の選択が違えば止まる | 509–513 | 合う（差し替えて止まることを確かめた） |
| (B) 1e−6 の照合と NaN の食い違いで止まる | 590–596 | 合う（ずらす・NaN にすると止まる） |
| (B) 2 段のブートストラップ，G_eff < 20 は jackknife との和集合 | 161–167，626–669 | 合う（E79 の百分位に 5.8e−7 以内） |
| (B) session_jk：セッションの jackknife（自由度 4）と百分位の和集合 | 653–659 | 設計どおり．ただし主の区間を含まない（指摘 3） |
| (B) δ_d,対 と `delta_d_pair_corpus` | 391–435 | 合う（E79 の a_d0 に 4.3e−14 以内） |
| (B) 模型に依る印 | 677–678 | 合う |
| (B) 直接の差：複製の差，jackknife の差，相手が none の単位は same_by_definition，`both_chosen` の表 | 935–970，1092–1099 | 点と区間は合う．相手が none の単位の扱いは設計どおりだが設計が誤り．さらに台本は oursx が none の単位も同じ扱い（指摘 1） |
| (B) 再現率の差 | 235–237，620–622，1106–1110 | 合う（NaN 0） |
| (C) 単位，f のアンカー，g，J，J_dir，再現率，φ，`_noscript` | 712–796，1112–1123 | 合う（52,560 行，J・J_dir の NaN 0．units の `none__` は確率で，対数の確率の `none_te__` は読まない） |
| P3a (i)：塊の数 > max(複製の帰無の 95% 点, 独立の上限) | 871–880 | 合う |
| P3a (ii)：効果の出うるセルの怒りの対の減らしたが 2 コーパス以上 | 881–883 | 合う |
| P3a (iii)：条件つきの感情の名前の置換の厳密な p < 0.05 | 884，890 | 合う |
| P3a の記述の列（無条件の p，対の置換の p，固定と条件つきの土台，外の数） | 885–910 | 合う |
| cebd64b の規則：音声「数 > 144 × 0.025」，顔「数 > 3.6」 | 894 | 顔が違う（指摘 2） |
| 感度の P3a（7 つ），比較相手と直接の差の P3a | 1079–1101 | 合う |
| P3c（記述．J，J_dir，選んだ教師の J，層の土台，範囲 5 つ） | 1126–1163 | 合う（teacher_g：ref 300，ref0 162，pF 12，pFK の欠け 12） |
| 仮定の表 1〜8 | 973–1028 | 合う |
| 記録（先頭の # 行，provenance，コードの写し） | 113–118，1183–1192，1214–1238 | 合う（写し 5 本と空の diff） |

設計の表に載る組（世界 2 × f × K 3 × コーパス，比較 5，感度 6，直接の差 4）は，すべて出力にそろった（下の格子）．

## A．小さく流した結果と全規模の照合

依頼のとおり，評価のターゲットの値を出す (A)(B) を (C) と合わせて全規模で流した．出力は `checks/E83/code_run/full/` と `t_*` に置き，照合の後に消した（07:01）．判定の値（減らした／増やしたの数，P3a〜P3c の当否，対ごとの点，厳密な p の値）はこの報告に書かない．

- コマンド：`./.venv/bin/python -W default exp/analyze_pairs_e83.py --out checks/E83/code_run/full`（既定の B ＝ 1,000，stages ab c）．06:50:05 開始，5 分 51 秒，最大 RSS 1.7 GB，警告 0，checks.csv は `distinct_test_sets` の 10 行だけ（chosen_missing・none_rows・ours_rows・seed_rows・chosen_differs_in_corpus は 0 行）．別の台本（E86 の試走）が同時に CPU を使っていた．
- 照合の台本：`checks/E83/code_run/check_recheck2.py`（E79 との照合と格子），`check_fix2.py`（関数を差し替えた検算．前回の `check_fix.py` を今の引数と戻りに合わせた），`check_exactp.py`（厳密な p と数の帰無を人工の例で），`check_p3a2.py`・`check_p3a2_direct.py`（全規模の P3a の表を台本の関数を使わずに計算し直す）．
- **格子**：b_units_pairs.csv 11,430 行 ＝ 比較 5 × (main・δ 半分・δ 2 倍・rev_order 各 486 ＋ iemocap_impro_only 216 ＋ session_jk 126)．鍵の重複 0，point・lo・hi・delta_d_pair・delta_d_pair_corpus・none_dirloss の NaN・Inf 0，missing 0，n_boot_valid の最小 1,000．b_units_direct.csv 1,944 行（4 × 486），重複 0，NaN 0．b_pairs_rep.csv 30,600 行，b_recall_rep.csv 20,400 行，どちらも重複 0・NaN 0．複製の鍵 1,095（NaN の行 0），jackknife の鍵 645（G は 5・10・254）．
- **厳密な p（独立の計算）**：コーパスごとの置換を全コーパスの直積で列挙して数えた（台本はコーパスごとの分布の畳み込み）．gate-stats・gate-define の例：2 コーパスで怒りの 3 対がそろって減ると，感情の名前の置換で p ＝ 0.0625，対の名前の置換で p ＝ 0.0025．1 コーパス 6 単位の怒りの 3 対（18 行）と別コーパスの怒りを含まない 1 行で，感情の置換 p ＝ 0.25，対の置換 p ＝ 0.050．台本と独立の計算がどちらも 4 つの値に一致した．条件つきの例：効果の出うるセルが怒りの対だけなら p ＝ 1.0（置換を引けたコーパス 0），4 対だけなら 3 コーパスで 0.0080（無条件 0.0156）．顔の例（3 コーパスで平静–怒りだけ）は 0.0046 ＝ (1/6)^3．
- **全規模の P3a の表の計算し直し**：主 8 行，感度 56 行，比較相手の main 32 行，直接の差 64 行のすべてで，n_pos_blocks，独立の上限，帰無の 95% 点，count_threshold，cond_count，n_pos_corpora_selected_elig，cond_two_corpora，分子の数，条件つきと無条件の厳密な p，cond_p，条件つきの土台，n_pos_elig，n_pos_not_elig と P3a_holds が一致した（食い違い 0）．
- **保存済みの値の再現（E79 の P5-3，6 対の平均）**：main 81，rev_order 81，iemocap_impro_only 36 の単位がすべて対応し，点の差の最大 2.9e−15・2.4e−15・2.9e−15，複製の百分位と E79 の lo_pct・hi_pct の差の最大 5.1e−7・5.8e−7・5.1e−7，AUC 版 3.4e−14，model_err_diff 4.0e−15（比の中央値 1.000）．δ 半分・2 倍の行の δ は main の 0.5・2.0 倍ちょうど．
- **比較相手の再現**：none − kd，none − groupB，none − abl_ownK，none − abl_own の 6 対の平均を E79 の `rows_*.csv.gz` の dirloss（種は平均）から作った値と比べ，81 単位で差の最大 2.2e−15・2.8e−15・2.4e−15・2.6e−15．複製：none − oursx と none − groupB の複製の 6 対の平均を E79 の `boot_*.npz`（groupB の種のある手法は agg の鍵）と比べ，差の最大 9.5e−7・1.2e−6（float32 の保存の誤差）．
- **直接の差の再現**：oursx − abl_own，oursx − abl_ownK の 6 対の平均が E79 の ablation_replace_dir（abl_*-oursx．向きが逆）に 2.7e−15・2.9e−15 で一致し，複製の百分位は 5.8e−7 以内，jackknife の複製の差から作った和集合の区間は E79 の lo・hi に 1.6e−6 以内で一致した．
- **δ の材料**：AVT 600 行・AV 1,260 行で E79 の delta_reps.csv の a_d0 に 2.8e−14・4.3e−14．
- **(A)**：反復の行 9,180，重複 0，NaN 0．平均の行 918 はすべて 10 反復．(B) の none_dirloss と (A) の差の最大 0.0．
- **退化した場合**：全比較の手法の行を none の行に替えて MELD と IEMOCAP で流すと，反復の行 18,900 で diff_dir・diff_dir_rev・diff_dir_auc・diff_model_err・diff_o がすべて 0.0，単位の行 6,390 で point・lo・hi がすべて 0.0，判定はすべて same，印はすべて False（rev_order は NaN），session_jk の 630 行も同じ．
- **照合の効き**：pairs の d に 1e−5 × (row_id mod 7) を足すと「最大 2.10e−04」で止まり，手法の行の d を 6 本に 1 本 NaN にすると「NaN の食い違い 1」で止まった．台本の群の audio K5 の選択だけを none に替えると主の比較で止まり，補助の比較では checks.csv に 7 行を書いて続き，単位の点が (即興の点 ＋ 0)/2 に一致した（4 比較 × 2 実行で差の最大 4.4e−16）．
- **(C)**：52,560 行，重複 0，J・J_dir の NaN 0．P3c の行 486．
- **provenance**：commit は流した時の HEAD（d9e2bc8．台本と依存の 4 本は 1675073 と同じ），dirty_deps False，deps_sha256 5 本，inputs 1,130 件，候補の数（5 比較），G_eff，セッションの一覧，distinct_test_sets，コードの写し `code/<開始時刻>/`（5 本と空の git_diff_HEAD.patch）．

## A．本番の規模の見積もり

(A)(B)(C) を合わせて 5 分 51 秒，最大 RSS 1.7 GB（CPU だけ．別の台本と並走）．設計 5 節の「5 分以内」は少し超えるが，10 分の上限の内である．GPU は使わない．出力は 63 MB．

## B．予想される指摘と，出力から答えられるか

前回の表（`code_recheck.md`）の (a) の行は引き続き (a)．変わった行と新しい行だけを書く．

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| P3a | 偶然の分布を対の名前の置換にしたら | S10，D14 | (a) `p_pair_perm_uncond` | — | — |
| P3a | 数の基準を独立の上限にしたら | S11 | (a) `chance_pos_blocks_anyof`，`null_count_q95` の列 | — | — |
| P3a | 効果の出うるセルの外の「減らした」を入れたら | D15 | (a) `share_selected_among_pos`，`p_class_perm_uncond`，`n_pos_not_elig*` | — | — |
| P3a・(B) | セッションの塊で区間を作ったら（IEMOCAP） | S3 | (a) ただし今の session_jk の区間は主の区間を含まない（指摘 3） | 主の区間との和集合 | CPU 数秒 |
| P3a・(B) | MELD の対話の塊で区間を作ったら | S3，E79 の cluster_boot | (c) 設計 4 節が範囲の外と書く | （足すなら E79 の cluster_labels） | CPU 数分 |
| 直接の差 | 相手が none の単位を含めても同じか | C1，C3，文の規則 (3) | (b) 今は 252 の（単位，対）を same_by_definition にしている（指摘 1） | 判定を付ける | CPU 数秒 |
| P3a の cebd64b | 最初の規則で当否は変わるか | L8 | (a) ただし顔の数の基準が設計と違う（指摘 2） | 3.6 に固定 | CPU 数秒 |
| (B) | IEMOCAP の即興と台本を別々の区間で | 例「ターゲットごとの内訳」 | 点は (a)，区間は (b)（前回と同じ） | 群ごとの複製 (B,6) | CPU 数秒，約 3 MB |
| (B) | groupB のどの手法が選ばれたか，種の揺れ | C2 | (a) `b_candidates.csv`，単位の行の chosen・n_seeds | — | — |
| P3c | 選んだ教師の J | C8 | (a) `J_teacher_same_corpus`（pFK の 12 行は欠け） | — | — |

学び直し（GPU）を要する行は無い．

## 指摘

### [C1・D15] 直接の差で，片方だけが none を選んだ（単位，対）を「定義から同じ水準」と判定する．差は 0 でない
- 重さ：止める
- 該当箇所：設計 `docs/exploratory-analyses.md:10082`（「相手が none を選んだ単位は『定義から同じ水準』にし，両方が none 以外を選んだ単位に限った表も出す」）；`exp/analyze_pairs_e83.py:964`，`:967`
  ```python
  both = (ra.chosen != "none") and (rb.chosen != "none")
  ... "judgment": judge_unit(x, lo, hi, ra.delta_d_pair) if both else "same_by_definition",
  ```
- 問題：相手（kd・abl_own・abl_ownK・groupB）が none を選んだ単位では，oursx − 相手 ＝ oursx − none であり，点は 0 でない．全規模で流すと，片方だけが none の（単位，対）は oursx − abl_own で 234，ほかの 3 つで 6 ずつ，計 252 あり，252 のすべてで point ≠ 0 なのに judgment は same_by_definition だった．E79 の直接の比較（ablation_replace_dir）はこの単位にも判定を付けている（judgment ＝ judgment_raw）．影響は 3 つある．(1) `b_judgments_by_pair_oursx-*.csv` の same_by_definition の列が「定義から同じ」と読める誤った数を出す．(2) `direct_all` の「減らした」は `direct_both_chosen` と必ず同じになり，文の規則 (3) の「両方が none 以外を選んだ単位に限っても超える」が条件として働かない．(3) 数の帰無（`null_block_counts`）はこの 252 行の複製（0 でない）から帰無の「減らした」を数えるのに，観測ではこの行が「減らした」になり得ないので，`direct_all` の基準が観測と別の集合で作られる．なお台本は，設計が書いていない「oursx が none で相手が none でない」単位も同じ扱いにする（今のデータでは oursx が none を選んだ単位は 0）．
- 計算の後で直せるか：直せる（CPU 数秒）．ただし定義を値を見てから替えると，結果を見てから足した規則と読まれる
- 求める対応：計算の前に，設計 10082 行を「両方が none を選んだ単位だけを『定義から同じ水準』にし，片方だけが none の単位は主と同じ規則で判定する．両方が none 以外を選んだ単位に限った表も出す」に書き直し，台本 967 行の条件を `both_none = (ra.chosen == "none") and (rb.chosen == "none")` に替える（`both_chosen` の列と `direct_both_chosen` の表は今のまま残す）．abl_own の解釈（none を選んだ単位で減らないことを証拠に使わない）は文の規則 (3) のままでよい．

### [L8・I1] cebd64b の規則の顔の数の基準が，設計の 3.6 ではなく「印を除いた判定の数 × 0.025」になっている
- 重さ：直してから進む
- 該当箇所：設計 `:10113`（「顔は『数 > 3.6 かつ平静–怒りの割合 > 1/6（印を除いた数）』」．cebd64b の原文も「数が 3.6 を超える（印を除いた数で読む）」）；`exp/analyze_pairs_e83.py:891`，`:894`
  ```python
  n_tests = len(m)
  holds_old = bool(n_pos > CHANCE * n_tests and ...)
  ```
- 問題：`p3a_table` は印を除く版で `m` を絞ってから `p3a_eval` を呼ぶので，顔の主の版（印を除く）では n_tests が 144 より小さくなり，基準が 3.6 より下がる．全規模で流すと，印を除いた版の n_tests は 91〜112 の値を取り，基準は 2.3〜2.8 になる．`P3a_holds_cebd64b` は「最初の設計の規則での当否」を並べるための列なので，基準が最初の設計と違うと並べる意味が変わる．
- 計算の後で直せるか：直せる（CPU 数秒）．ただし値を見た後で基準を替えたと読まれる
- 求める対応：計算の前に，cebd64b の数の基準を族ごとに固定値（音声・顔とも 144 × 0.025 ＝ 3.6．言葉・融合は判定しないので記述）にする．`chance_pos` の列（記述）は今のままでよい．

### [S3] session_jk の区間は主の区間（話者の jackknife との和集合）を含まず，IEMOCAP の 630 行のうち 273 行で主より狭い
- 重さ：直してから進む
- 該当箇所：設計 `:10084`（「5 つのセッションを 1 つずつ抜く jackknife（自由度 4）と百分位の和集合を感度の行 session_jk として出す」）；`exp/analyze_pairs_e83.py:653–659`
- 問題：主の IEMOCAP の区間は「百分位 ∪ 話者の jackknife（自由度 8）」，session_jk は「百分位 ∪ セッションの jackknife（自由度 4）」で，話者の jackknife を外している．全規模で流すと，session_jk の幅 ÷ 主の幅は 0.60〜1.64 で，273/630 行で主より狭かった（fix_applied_2 の人工データでは 1.0〜1.5 倍だった）．gate-stats の S3 が求めたのは「話者を独立とみなした区間では幅が足りないおそれ」の確かめで，主より狭い区間ではその確かめにならず，主で「判定できない」だった行が session_jk で「減らした」になりうる．E79 の cluster_boot は主の区間との和集合だった．
- 計算の後で直せるか：直せる（CPU 数秒）．ただし同上
- 求める対応：計算の前に，設計 10084 行を「主の区間（百分位 ∪ 話者の jackknife）とセッションの jackknife の和集合」に替え，台本 658 行で session_jk のときも話者の jackknife（`jk`）の区間を合わせる．

### [L8・L11] gate-code の 3 回目の全規模の流しと，見た値の範囲を台帳に書く
- 重さ：記録する
- 該当箇所：`docs/seen-data.md:27`
- 問題：依頼に従い，2026-11-03 06:50〜06:58 に (A)(B)(C) を全規模で `checks/E83/code_run/full/` に流し，検算で MELD と IEMOCAP の (B) を `t_degen`・`t_perturb`・`t_nanmis`・`t_diverge` に流した．07:01 にすべて消した．私が見た値は次に限る．E79 との照合の差，行と鍵の数，格子，比較ごとの same_by_definition の行の数（abl_own 234，ほか 6）と both_chosen の内訳，片方だけ none の 252 行の point ≠ 0 の数，n_seeds の値，区間の規則の行の数，session_jk と主の区間の幅の比（0.60〜1.64，主より狭い 273/630 行），P3a の表の構造の列（n_blocks，n_tests，独立の上限の範囲 0.59〜3.41，帰無の 95% 点の範囲 0〜5），P3a の表で NaN が出る列とその数（分母 0 の行が感度の表に 1 行，比較相手と直接の差の表に 12〜14 行あること），P3c の teacher_g の内訳，選択の設定の文字列（検算の出力），台本の値と私の計算し直しの値が一致したかの真偽．減らした／増やしたの数，P3a〜P3c の当否，対ごとの点，厳密な p の値は見ていない（一致の確かめは台本の中で比べ，真偽だけを出した）．
- 計算の後で直せるか：直せる（記録だけ）
- 求める対応：`docs/seen-data.md` の E83 の行に (5) として上の流しと見た値の範囲を書く．

### [I9] 本番が途中で落ちると provenance.json が無いまま出力が残り，次の本番が止まらずに上書きする
- 重さ：記録する
- 該当箇所：`exp/analyze_pairs_e83.py:1211`，`:1238`
- 問題：前の本番を検出する条件は provenance.json の有無だけで，provenance.json は最後に書く．落ちた回の CSV とコードの写しは残り，次の回はそれを上書きする．1 回で流せば起きない．
- 計算の後で直せるか：直せる
- 求める対応：5 節に「落ちたら `results/e83/` を消してから流し直す」と書くか，開始時に `results/e83/` が空でなければ止める．

### [I15] 本番の provenance の commit は流した時の HEAD になる
- 重さ：記録する
- 該当箇所：`exp/analyze_pairs_e83.py:1215`
- 問題：今は HEAD が d9e2bc8（E84 の結果のコミット）で，検査した 1675073 ではない．台本と読み込む 4 本は同じ内容なので結果は変わらず，provenance の `deps_sha256`（台本 b0d612fe…）で照合できる．
- 計算の後で直せるか：直せる（記録だけ）
- 求める対応：6 節の記録に「検査したコミット 1675073，台本の sha256 b0d612fe…」を書き，provenance の値と突き合わせる．

当てた項目：I1〜I17（I2〜I5 は新しい学習と選択が無いので，読み込みが `load_any` を通り sample_id の並びを照合すること，選択を chosen.csv から読むことだけを確かめた．I13 は当てはまらない），R1〜R13，S1〜S4・S9〜S14，D14〜D18，C1〜C4・C8・C10〜C12，L8・L10〜L12．

## 目録に無い種類の不備

- C14（案）：手法どうしの直接の差では，片方だけが「何もしない」（none）を選んだ単位の差は 0 ではない．「定義から同じ」とするのは両方が none のときだけにする．片方だけ none の単位を外す表を作るときは，外したことを別の判定名（例：partner_none）で書き，帰無の数え上げからも同じ行を外す（E83 の関門 1 の再検査 2）．
- S15（案）：塊（セッション・対話）で作る感度の区間は，主の区間との和集合にする．塊の jackknife で主の話者の jackknife を置き換えると，自由度と抜き方が違うので主より狭くなる行が出て，「主より保守的な確かめ」にならない（E83 で 273/630 行）．人工データでの幅の比は実データでも確かめる．
