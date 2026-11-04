（検査役は Sonnet のサブエージェント．報告の .md を書けなかったので，本体が記録から取り出して保存した．2026-11-05）

判定：直してから進む
理由：欠け・重複・非有限・食い違いは 0 で，独立に作り直した UAR と効果が全て一致した．ただし dai3 の expl4.csv は，ターゲットのコーパスを表す T 列が，生徒 T の値で上書きされている．
（ファイルは書かず，全文を返答する．DAI/checks/DAI-eval/code_post.md への保存は本体に任せる．検算の台本と出力は DAI/checks/DAI-eval/code_run_post/ に置いた．）

# 計算の後の点検：評価の回の出力と集計（DAI-1/2/3，コミット ba98974，2026-11-05）

## 結論

- 評価の回の保存 2440 ファイル（実験 1 が 1200，実験 2 が 1040，実験 3 が 200）は，格子から決まる数と一致する．欠け 0，重複 0，非有限 0，収束しなかった問 0 である．
- 全単位が ba98974・dirty でない・H200 である．gate_sha256 は 1 つ，frozen_sha256 は実験ごとに 1 つで，どちらも eval_gate.json・固定のファイルと一致する．
- 保存した判定（int8）から UAR を作り直し，csv の UAR と照合した．実験 1 は 909,600 行，実験 2 は 421,260 行で，食い違いは 0 である．
- 実験 1 の族 1〜4 の 176 セルの効果 θ を，台本を使わず rows と固定のファイルの選択から作り直した．最大の差は 4.2e-15 である．
- 直す点は 1 つである．expl4.csv の T 列が壊れている．台本は直さず，行の並びから復元する（指摘 1）．

## 1．単位・段・種・分割・コミット・計算機

| 確かめた事項 | 結果 |
|---|---|
| 実験 1 の単位（4 T × 5 分割 × 10 種 × 3 段 × npz/csv） | 1200 ファイル，欠け 0，余り 0 |
| 実験 2 の単位 | lin 200，coral 200，map 60，xmuda 60（各 npz/csv）．期待 4×5×10，4×5×10，4×5×3，4×5×3 と一致 |
| 実験 3 の単位 | pkl 200，欠け 0，余り 0 |
| 1 単位あたりの行数 | 段 1 は 735，段 2 は 385，段 3 は 4396 で，200 単位とも同じ．lin は 2304，coral は 332 である．map（1254〜1258）と xmuda（399〜411）の差は，解き直しの行の役の併合による（`dann_rep` と `mme_rep` は 40 で一定，`mapctl_rep` は 54〜60，`xmuda_rep` は 39〜51） |
| 手法ごとの行数が設計の格子の積と一致するか | 一致する．none 392，oracle 98，tgtonly 192，kbias 32，joint/rolesep/kd/joint_scratch 480，agree/mix 360，cotrain 80，coreg 100，tether 280．lin は none 1，self 12，entmin 4，im 12，t3a 6，t3a_b 6，lp 12，own 12，st 3，saerens 1．DANN と MME は 24，mapctl は 8，xmuda は 24 |
| 引き戻しの曲線（`*_tether`） | 各 (単位，手法，生徒，K) で 14 行．`joint_tether` は 4 T とも 14,000 行．相手の手法は固定のファイルの相手と一致する（kd 75 セル，agree 5 セル．AFEW と IEMOCAP は kd のみ，MOSEI は agree 2，MELD は agree 3）．共訓練と共正則化の相手は無いので曲線も無い |
| 重複した行 | 識別子の列で 0（段 1〜3，lin，map，xmuda）．集計の表も，鍵の重複 0 と全行重複 0 |
| コミット・dirty | npz と csv の全行が ba98974583dfee3504d54f31a5d303ff5394562e，dirty は False．dai3 の pkl 200 も同じで，`dai1_commits` も ba98974 の 1 つ |
| sha256 | gate_sha256 は 001629ec… の 1 つで，eval_gate.json の sha256 と一致．frozen_sha256 は dai1（6926fb3e…）と dai2（98855729…）が各 1 つ，dai3 の pkl の frozen3 は 22efe108… の 1 つで，どれも DAI/frozen/ と results/*/ の写しと一致 |
| 計算機 | 実験 1 と 2 の全行が `NVIDIA H200`．実験 3 の pkl は計算機の列を持たず，provenance の gpu が H200 |
| provenance の台本の sha256 | 4 Pod × (dai1 の 5 本，dai2 の 10 本，dai3 の 1 本) が eval_gate.json の台本・solver・data の sha256 と一致．frozen と gate の sha256 は別行で記録されている．実行は各分割 1 回で，再実行の行は無い |
| Pod の sha256 一覧と研究室の PC | `sha256sum -c` を Pod 4 つ全てで流した．610 ファイルずつ，計 2440 が一致し，重複した名前は 0 |
| 特徴などの入力 | 各 Pod の `.inputs` の sha256 を研究室のファイルと照合した．実験 1 と 2 の分は全て一致する．実験 3 の `.inputs` では results/dai1/provenance_log.jsonl だけが不一致である．これは Pod 側の同ファイルが評価の追記で伸びた後の sha256 を記録したためで，Pod ごとの写し（`provenance_log_eval_dai-e*.jsonl`）の sha256 と一致する．不備ではない |
| 評価の回の経過 | 4 Pod とも `.done` が 3 つ（段 1，2，3）あり，ログに traceback・error・OOM・warning・nan の語は無い．開始 21:07（JST）で，終了は e4 が 00:52，e1 が 02:08，e3 が 03:08，e2 が 03:45 である |

## 2．非有限・収束・発散の印

- **非有限**：csv の数値列に NaN・Inf は 0．UAR の範囲外は 0（最小の UAR は 4.2 で，エントロピー最小化の崩壊した局所解に由来し，範囲内）．npz の float 6.3×10⁹ 値に非有限は 0．int8 の判定は，自己学習の「疑似ラベル無し」の −1（`*_pl` のキー）を除き，全て 0〜3 に収まる．
- **収束**：実験 1 の全問で勾配の最大は 1e-6 以下で，1e-5 超は 0，行き詰まり（stalled）は 0 である．段 1 の `_conv` は 1400 行，段 2 は 3000 行，段 3 の `_conv*` も 1e-5 超 0 である．実験 2 の線形（entmin・im・self・lp・st・none）は最大 1e-6 で，stalled は 0，coral は勾配 1e-5 超が 0 である．実験 3 は 11,831,796 問のうち 1e-5 超が 0，stalled が 0，非有限が 0 で，ログにも「切片の問 0」とある．
- **実験 2 の発散の印**（設計 2.3 節のしきい値で，csv から数え直した．`marks.csv` と一致する．まとめて「発散」と呼ばず，発散した問も格子から外さない設計どおり）：
  - 判別器の印（DANN の `dloss_after` > 10 ln2）は 4500／28,800 問．学習率 1e-2 が 4496／14,400，1e-3 が 4 である．
  - 膨張の印（共分散の差が学ぶ前の 100 倍超）は，DANN が 6494（学習率 1e-2 が 6082，1e-3 が 412），MME が 1335／28,800，mapctl が 3／9600 である．
  - xMUDA 型の ‖A−I‖_F > 100 は 1 行（最大 103.4．MELD，学習率 1e-2，K=1）で，3 生徒が同じ問を共有するので `marks.csv` では 3 問と数える．
  - 選んだ点が印に当たったセルは 0．選択なしの点が当たったセルは 2（IEMOCAP の MME，生徒 AVT，K=25．MOSEI の DANN，生徒 AVT，K=0）である．設計どおり表の注に書く．

## 3．集計の出力の完全性

| 出力 | 結果 |
|---|---|
| `families.csv`（dai1_stats eval） | 176 行 = 4 T × (族 1 が 12 + 族 2 が 8 + 族 3 が 12 + 族 4 が 12)．K の集合は設計どおり（族 2 は 5 と 25，ほかは 0，5，25）．n_units は K=0 が 5，K≥1 が 50 である．NaN は 0．Holm を族ごとに自分で計算し直すと 16 族全てで一致した．標準誤差 0 のセルは 44 で，`counts.csv` の「選んだ設定が選択なしと同じ」44／48 と一致する |
| `desc/`（dai1_desc eval） | 表 27 本と manifest がそろう．`desc_pseudo/` は表 25 本で，引き戻しの曲線 2 本が無いのは設計どおり．T ごとの行数は，ablation 172，confusion 1600，levels 872，recall 720，tether_curve 1400 などで 4 T 等しい．不等なものは，sensitivity（AFEW 22，IEMOCAP 94，MELD 66，MOSEI 44），sensitivity_folds（MOSEI のみ 220），convergence（41〜42）で，部分集合と手法の構成で説明がつく．9.4 節の部分集合（MOSEI のチャンネル，IEMOCAP のマーカーと注記，AFEW の書き起こしの疑い，MELD の顔と対話）が全て出ている．30 本未満の分割は 0（最小のテストの件数は 147）で，`folds_under_30` が全て NaN であることと整合する |
| `dai2/stats/`（dai2_stats） | 表 21 本．`prop_minus_method` は 271 セル × 4 T = 1084 行で，設計 4 節の 271 セル（saerens と t3a_b を除くと 231）と一致する．ほかは `xmuda9` が 36 = 9 × 4，`corpus_sentence` が 4，`fam3_cells` が 48，`marks` が 600，`coral_tether_curve` が 2688 で，`checks.csv` は 4 行（4 T）である．勾配法のセルの n_units は 15，K=0 の線形は 5 である．Welch–Satterthwaite の区間は，種の項の割合が半分を超える 39 セルだけに付き，`ws_applies` と一致する |
| `dai3/eval/` | 表 22 本 + `convergence.csv` + `verdicts.json` がそろう．4 T の行数は等しい（expl1 は 4×8，expl1_tt は 54，expl2 は 12，expl3 は 6，expl3_cells は 12，family5 は 24 など）．一部の T だけを持つ表は設計どおり（expl5 は IEMOCAP のみ，recall_content は IEMOCAP と MOSEI，family5_subsets は AFEW・IEMOCAP・MOSEI）である．`verdicts.json` は expl1・2・3・5 を 4 コーパスで持ち，`descriptive_only` に「説明 4」「説明 6」が入る |
| dai3 の流し直しが保存を読むだけだったか | ログに「eval <T>_f…: … 秒」の行は 0 である．集計は「表 22 本（行 58171），勾配 1e-5 超 0，81 秒」の 1 行だけで，保存を読むだけだった．pkl の sha256 は集計の後も Pod 一覧と一致し，保存は書き換わっていない |
| 集計の台本 | dai1_stats・dai1_desc・dai2_stats・dai3 は ba98974 と HEAD で差分が無い．作業木 /home/snics/MER-DA-eval-ba98974 は未追跡の `.venv` を除き汚れていない．ログの終了コードは 0，例外と警告は 0 である |

## 4．判定からの UAR の作り直しと集計の照合

- **実験 1**（`d10_predcheck1.py`．c9 の範囲に加えて段 3 の本体の格子と引き戻しの曲線も照合）：909,600 行を照合し，食い違い 0 である．
  - 内訳：none 78,400，oracle 19,600，start 1,400，start_srcstd 1,400，kbias 6,400，tgtonly 38,400．
  - joint・kd・agree・mix・tt の格子 424,000．
  - 引き戻しの曲線 112,000（joint 56,000，kd 52,500，agree 3,500）．
  - joint_scratch 96,000，rolesep 96,000，coreg 20,000，cotrain 16,000．
- **実験 2**（`d11_predcheck2.py`）：418,860 行 + 追加で CORAL の K=0 の 2400 行を照合し，食い違い 0 である．複数行に当たる照合も 0．
  - 内訳：lin の全手法（t3a・t3a_b・saerens を含む），coral，coral_tether，map（dann・mme・mapctl）とその `_rep`，xmuda とその `_rep`．
  - 照合の外：自己学習の巡ごと（`_self_round`）の行，ラベル伝播の診断行，`_coral_check` と `_coral_src`．これらは UAR を持たない診断行か，集計の照合の側にある．
- **実験 2 の `none` と実験 1 の比較の基準**：4000 行が UAR まで完全に一致する（K>0 が 3200，K=0 が 800．最大の差 0.0）．実験 2 の `none` 行の `dai1_same_units` は全 4000 行で False（`none_agree_dai1` は NaN）なので，実行の中の照合は走っていない．Pod の連鎖が実験 2 → 実験 1 の順で，実験 2 の実行時には実験 1 の保存がまだ無かったためと推察する（連鎖ログの E2 → E1 → E3 の順による）．集計の側の照合（`d1_rows` 12,000）と私の照合で補われているので，結果への影響は無い．
- **単位の標本**（`d15_ids.py`）：200 単位全てで，テストが分割 f，候補がその他の分割（MELD はテストの対話を除く）と一致する．テストと候補で塊が重ならない．K 本は感情ごとに K 本で入れ子になり，候補の中に入る．実験 2 の 4 種の npz の標本は実験 1 と同一で，K=5 の抜き方は分割ごとに 10 種で全て異なる．
- **族 1〜4 の効果の作り直し**（`d13_theta.py`）：176 セル全てで，固定のファイルの選択（主の版，ρ と a，引き戻し，相手の手法と版）から rows の UAR で θ を作り直した．`families.csv` の effect との最大の差は 4.2e-15 である．
- **集計の台本が自分で出した照合の数**
  - families：4 T とも 0／600．
  - desc eval：19 区分 867,600 行で食い違い 0，欠けた行 0，退化の検算 240／240．
  - desc pseudo：445,560 行で 0，退化の検算 240／240．
  - dai2_stats（`checks.csv` の全列）：unit_sample_mismatch，tether_key_mismatch，frozen_sha_mismatch，rows_duplicate，rows_without_pred，rows_uar_mismatch，dis_rep_mismatch，dis_ctl_mismatch，d1_rows_mismatch，grid_mismatch，split_mismatch，rows_dirty がいずれも 0．checked は 421,260，dis_rep が 10,860，dis_ctl が 79,200，d1_rows が 12,000，d1_tether が 44,800 である．
  - dai3：W と b から作り直した判定が保存の判定と全一致しない問が 8（指摘 3）．行の和の相対誤差は最大 1.1e-7 で，切片の問は 1e-5 超が 0 である．

## 5．評価の回の前に固定した規則が集計の出力に反映されているか

- **記述に回すセル**
  - 固定のファイルの族 1／2／3／4 の記述セルは，AFEW が 12／8／12／2，IEMOCAP が 12／8／12／2，MOSEI が 4／6／9／2，MELD が 10／8／12／2 である．合計は族 1 が 38／48，族 2 が 30／32，族 3 が 45／48，族 4 が 8／48 で，設計 9.2 節の 38・30・45 と一致する．
  - `desc/counts.csv` の `fam{1..3}_descriptive_cell` の数は上と同じである．
  - `dai2/stats/fam3_cells.csv` のセルごとの `descriptive` 列は，固定のファイルと 48 セル全てで一致する（0 不一致）．
  - `corpus_sentence.csv` は `a_ok` が 4 コーパスとも False（設計どおり，族 3 のコーパスの文は書けない）．
  - `families.csv` には `descriptive` の列が無い（指摘 2）．
- **同じ水準の δ**：族 3 で「同じ水準を出しうる」セルは 4／48，族 4 で 32／48，族 1・2 で 0 である（`desc/counts.csv`）．`families.csv` に「同じ水準」の判定列は無く，p・区間・Holm のみである．設計の「出しえないセルでは判定が起こらない」と矛盾しない．
- **dai3 の wording_direction**：`family5.csv` の 96 セルで，固定のファイルの妥当性の確かめ 2 の合否と全て一致する（不一致 0）．不合格の (コーパス，生徒) は AFEW の V・T・AVT と IEMOCAP の V・AVT の 5 個で，30 セルが False である．族 5 の記述セルは 89／96（AFEW 24，IEMOCAP 24，MELD 23，MOSEI 18）で，固定のファイルと一致する．検定に残るのは 7 セル（MOSEI 6，MELD 1）で，設計 1.1 節と同じである．
- **族 5 以外の主張の制限**：`verdicts.json` の `notes` に「方向」の語の制限と，説明 4・6 が記述のみであることが入っている．説明 5 の結論は「言葉についての文は IEMOCAP 即興を除いて書く」である．

## 指摘

### [1] expl4.csv の T 列が生徒 T の値で上書きされ，コーパスが分からない
- 重さ：直してから進む
- 該当箇所：DAI/exp/dai3.py:1721〜1723（`row = dict(T=T, ...)` の後に `for s in "AVT": row[s] = ...` で `row["T"]` を上書きする）．出力は results/dai3/eval/expl4.csv
- 問題：expl4.csv の T 列に，コーパス名ではなく生徒 T の値（例：1.6469，3.7038）が入っている．全 30 行がこうである．
  - expl4 は説明 4（顔の得）の表で，記述として論文に載る見込みがある．
  - 値自体は正しい．A 列と V 列は族 1 の効果と一致する．AFEW の K=5 では A が 4.147787，V が 0.091665 で，`families.csv` と同じである．全 8 組（4 コーパス × K=5・25）で確かめた．
  - 行の並びは AFEW → IEMOCAP → MOSEI → MELD の順で，台本の `for T in args.targets` と `dai_data.CORPORA` に一致する（MELD の末尾 12 行に `face_single` が入る）．
  - 他の表で T 列にコーパス名以外が入るものは，全 CSV を走査して 0 だった（`desc_pseudo/assumptions.csv` の `c` 列にある「pseudo(30 単位)」は記述用の値で問題ない）．
- 計算の後で直せるか：直せる．計算は不要で，次の手順で足りる．
  - dai3.py を直して流し直す方法は使えない．dai3.py の sha256 は eval_gate.json に入っており，保存した pkl のコミットが ba98974 のままだと，新しいコミットでは dai3 eval が「保存済みのコミットが違う」と止まる（dai3.py 1904 行付近）．
  - 台本の外の短い台本で，行の並びからコーパス列を足して別名で保存する．足した後に，各 (コーパス，K) の A・V の値が `families.csv` の族 1 の効果と一致することを確かめる．
- 求める対応：結果を文書に書く前に，expl4.csv の使用を上の復元済みのファイルに替える．dai3.py の 1 行の修正（列名の衝突の解消）は次の版の台本に入れ，今回の評価の回の出力には手を付けない．この経緯を `docs/seen-data.md` と台帳に書く．

### [2] `families.csv` に「記述に回すセル」の列が無い
- 重さ：記録する
- 該当箇所：results/dai1/stats/families.csv（30 列に `descriptive` と `same_level` が無い）．記述のフラグは固定のファイルの `detect.descriptive_cells` と `dai2/stats/fam3_cells.csv` の列にある
- 問題：族 1・2・4 のセル単位のフラグは，固定のファイルを引かないと分からない．`desc/counts.csv` は集計の数しか持たない．族 3 は `fam3_cells.csv` に列がある．族 1・2・4 を表に載せるとき，文を書いてよいセルとの対応を個別に付け直すことになる．件数は固定のファイルと一致しているので，数値の誤りではない．
- 計算の後で直せるか：直せる（CPU，固定のファイルとの結合のみ）．
- 求める対応：表を作る台本で，`families.csv` に固定のファイルの `descriptive_cells` を結合した列を足す．

### [3] dai3 の「W と b から作り直した判定」が保存の判定と全一致しない問が 8 件
- 重さ：記録する
- 該当箇所：results/dai3/eval/assumptions.csv の `作り直した判定が保存と一致する` の 7 行．対象の単位は AFEW_f3_r1，MELD_f0_r0，MELD_f0_r6，MOSEI_f0_r2，MOSEI_f1_r3，MOSEI_f3_r4，MOSEI_f3_r8．
- 問題：float32 で保存した W・b から argmax を作り直すと，テストの 1〜2 本（AFEW の 158 本中 1 本など）の判定が保存の判定と違う．最小の一致率は 0.9937（AFEW 生徒 A，K=1 の 2 問）で，ほかは 0.9995 以上である．これは近い値のとき，丸めで argmax が入れ替わるためと見込む（原因は確かめていない）．
- 影響：UAR は保存の判定（int8）から作るので，実験 1 と 2 の集計には影響が無い．実験 3 は W・b から切片を当て直す計算で，この 8 問の影響は未確認である．
- 計算の後で直せるか：影響の大きさは pkl から CPU で確かめられる．
- 求める対応：結果の文書に，「保存した W・b から作り直した判定は，8 問で 1〜2 本（AFEW は最大 0.6%）保存の判定と違う」と注に書く．

### [4] 実験 2 の `none` 行の実行内の照合が走っていない
- 重さ：記録する
- 該当箇所：DAI/exp/dai2.py:485〜518（`dai1_same_units` が False なら照合しない）．results/dai2/eval/units/*_lin.csv の 4000 行すべてで `dai1_same_units` = False，`none_agree_dai1` = NaN
- 問題：実験 2 の実行時に実験 1 の保存が無かったため，実行の中の照合は走らなかった．
- 補った確認：私が事後に 4000 行すべてで UAR を実験 1 の比較の基準と照合し，差が 0.0 だった．判定の標本（テストと K 本）も実験 1 と同一で，集計の `d1_rows` の 12,000 行も食い違い 0 である．
- 求める対応：論文の方法の節には事後の照合の結果を書く．

### [5] 記録の補足（結果に影響しない）
- `family5.csv` の `verdict` 列は，記述に回した 89 セルにも「上回る」（23）・「下回る」（4）を入れている．文を書いてよいかは `descriptive` が False のセル（7）に限る．表に載せるときは `descriptive` で分けて印を付ける．
- Pod の `verify.txt` は，研究室で流した最初の確認が作業ディレクトリ違いで「No such file」になった記録で，不備ではない．実際の確認は `COLLECTED`（OK 610）と，私の再確認（2440 一致）である．
- エントロピー最小化・IM・ラベル伝播・T3A・DANN で，UAR が 20 を下回る崩壊した解が格子に多い（線形の entmin と im は 1 コーパス・生徒・K の格子点に最大 48 問）．実験 1 では tgtonly の K=1 に 69 問，coreg の少数がある．選択は仮のターゲットの回の出力で済んでいるので，評価の結果の解釈上の注として書く．

## 目録に無い種類の不備（docs/methods.md 9.3 節への追加案）
- **表の列名と鍵の列の衝突**：表の行を `dict(T=<コーパス>, ...)` で作った後に，同じ名前の列（生徒 T など）を代入して鍵が上書きされる．表を書き出す関数で，鍵の列（T，c，K，student）を他の列名が上書きしていないかを自動で確かめる．具体的には，全 CSV で T 列の値がコーパス名の集合に入るかの走査（今回は 1 本だけ違反）．計算の後の点検の標準手順に入れる．

## 本文の照合の記録
- 台本・出力：DAI/checks/DAI-eval/code_run_post/ の d1_units，d2_rowcount，d3_dups，d4_dai2，d7_npzscan，d10_predcheck1，d11_predcheck2，d13_theta，d15_ids．出力は out_d*.txt．
- 評価のターゲットの成績の値は，ここでは要約に書いていない．本文の照合に使った値は，families.csv と expl4.csv の A・V の列の一致の確認に限る．

最終判定：直してから進む（指摘 1 の expl4.csv の T 列の復元のみ．その他は通す）