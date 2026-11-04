# 関門 1：実装と出力 第 2 回（DAI-3，2026-11-04，コミット 9997abd．差分 `git diff 3256892 9997abd -- DAI/exp/dai3.py DAI/docs/実験3_設計.md`）

判定：通す．第 1 回の止める 2 件（R2，R4）と直してから進む 7 件（I1-a，I1-b，I11-a，L4，I1-c，I11-b，I8）は，設計 4 節と 9 節に書き足した規則どおりに直っている．9997abd の台本で 4 単位を流し，独立に作り直した値はすべて一致した．記録の I14 と記録 1・2・3・5・6 も直っている．残るのは記録 3 件で，どれも計算を止める理由にならない．

## 検査の対象と流し方

- 9997abd の DAI/exp/dai3.py（3256892 との差：385 行の変更）と設計 4 節・8 節・9 節．DAI/exp・measure・features・DAI/splits・data/metadata に未コミットの変更は無い．
- 材料は第 1 回と同じ実験 1 の試走 DAI/checks/DAI-1/code_run_v3/out/Y2 である．置き場は DAI/checks/DAI-3/code_run_v2/ に作った（setup.py，run_trial.py は第 1 回の写し）．GPU は A5000，MERDA_GPU_FRACTION=0.2，ほかの GPU の処理は無かった．成績の値は出していない．
  - validate（T = MOSEI，8 単位）：22 秒．確かめ 1 は 896 行（W の種類に W* が加わり 4 種類 × 7 版），確かめ 2 は 96 行．1e-5 を超えた切片の問は 0．
  - detect：38 秒．frozen.json の _meta に encoder，script_sha256，detect_expected_sha256 が入った．効果の見込みは detect_expected.json に分かれ，frozen.json には descriptive の真偽だけが残る．count_rule の行は 4 本（試走の 2 分割）．
  - eval（AFEW と IEMOCAP，種 0，4 単位）：93 秒（AFEW 27〜29 秒，IEMOCAP 18 秒／単位）．表は 22 本（5,655 行）で，oracle_grid（1,344 行）と speaker_disjoint（336 行）に NaN は無い．1e-5 を超えた切片の問は 0．
- 検算の台本：code_run_v2/check_v2.py．

## 直しの確かめ

| 第 1 回の指摘 | 直した箇所 | 確かめ | 結果 |
|---|---|---|---|
| R2（b* と感情ごとの再現率を残さない） | refit_item が `bstar[v]`（(J, 3, k)．組 A・組 B・全体で解いた値．float32）と `rec[orig／refit／fullb][v]`（(J, k)）を返す | 保存した b* から交差の当て直しの UAR を作り直す（2,100 個）．感情ごとの再現率の平均と orig・refit・fullb を比べる（6,300 個）．scipy で解いた b* と比べる（140 個） | 作り直した refit・fullb との差の最大 1.4e-14．再現率の平均との差の最大 8.1e-6（float32 での保存のため）．scipy の b* との差の最大 1.9e-5（共通の定数を除く）．pickle は AFEW で 1.1 → 1.7 MB／単位 |
| R4（W* の引き戻しと 2 組の分け方が 1 通りだけ） | eval_unit で，全体の複製（j = 0）に限って次の 2 つを計算する．(i) W* の格子の全点の refit（`oracle_grid`）．(ii) 話者が重ならない 2 組での refit（`disjoint_halves`，`refit_dis`）．集計は r4_tables．設計 9 節に，区間を付けない記述であることと IEMOCAP は話者 F・M で分けることを書いた | 選んだ点の格子の値を R["refit"]["<種類>|oracle"] と比べる．2 組が塊（IEMOCAP は話者）を共有しないかを数える．差の列が列の差になっているかを確かめる | 選んだ点との差 0（16 個）．格子の点の数は試走の格子どおり 4．共有する塊・話者は 0．差の列の誤差は 1.8e-15．本番のラベルで数えると，話者が重ならない 2 組で計算できる版は AFEW の分割 0 で 0/7，分割 2 で 5/7，分割 4 で 3/7，IEMOCAP の分割 3 で 3/7，ほかは 7/7（表の n_units と n_units_disjoint_not_computed に出る） |
| I1-a（族 5 の判定に確かめ 1 を当てない） | 1437–1439：verdict_raw を残し，check1_pair_ok が True のときだけ verdict にする．Holm の族には残す | family5.csv の 48 行を規則で照合する | 規則どおりでない行は 0．試走では check1_pair_ok がすべて True だったので，False と None の経路は読んで確かめただけである |
| I1-b（確かめ 1 の不合格が支持の側に数えられる） | aggregate で cell_undetermined を (iii)・(iv) の確かめ 1 から決める．rules はそのセルをどの条件にも数えず，分母は 8 のまま | expl1.csv の 8 行と verdicts.json を照合する | 規則どおりでない行は 0．AFEW で判定できないセル 1 つがこの経路を通った |
| I11-a（確かめの欠けを合格として扱う） | 呼ぶ側は `ok is True` のときだけ判定に使う．本番の eval は，check1 の鍵 4 コーパス × 4 生徒 × 6 種類 × 2 版と check2 の鍵 16 がそろい，partial_targets が無いときだけ動く | 読んで確かめた（本番の置き場でしか働かない） | 合う |
| L4（AFEW の 8 セルが件数で決まる） | groups_ok（10 本と 30 本の規則）を分割の値に当て，当たった分割をその量の平均から外す．全部の分割が外れたセルだけを付録に回す．detect が外れる (コーパス，部分集合，版，分割) を frozen.json の count_rule に書く．設計 4 節と 9 節を直した | 本番のラベルで label_tables を流す | 20 個（AFEW の全体で 2：対 1-2 の分割 0 と対 0-2 の分割 4．AFEW の asr_ok で 6，IEMOCAP の nomarker で 12）．設計 9 節の数と一致する．試走の族 5 で appendix_only は 0 |
| I1-c（説明 4 の (b) の書き方） | 設計 9 節に，得と直す余地はテストだけを限り，候補も限るのは再現できる度合いだけと書いた | 文を照合した | 合う |
| I11-b（非有限の b* で止まらない） | solve_b が finite_p と b の有限性を見て，単位・識別器・版を書いて止まる（ctx） | 読んで確かめた | 合う |
| I8（第 2 のエンコーダの置き場で守りが外れる） | production が results/dai3 と results/dai3_enc2 の両方で真になる．写しの名前を置き場ごとに分けた．本番の置き場は決まった --dai1-out と --encoder でだけ動く（置き場を作る前に確かめる）．second_encoder の列を --encoder から書く | `--out-dir results/dai3_enc2 --dai1-out <試走> --encoder 2` で流した | 「本番の置き場 dai3_enc2 は --dai1-out results/dai1_enc2 と --encoder 2 で流す」と表示して止まり，results/ の下に何も作らなかった |
| I14（作り直した判定の照合が一部だけ） | rebuilt_check に 3c（単一の生徒の共訓練），3g の単一の生徒，3r_a，3r_b を足した | assumptions.csv の照合の行 | 130 行，一致しない単位がある行は 0．段は 1，3，3c，3g，3r_a，3r_b |
| 記録 1（空の複製の警告） | nanmean_rows が RuntimeWarning だけを抑える | 4 単位の eval のログ | 警告は出なかった |
| 記録 2（W* が確かめ 1 に無い） | validate の W の種類に oracle を足した（detect と同じ選び方の点） | validate の行の数 | 896 行（種類 4 × 7 版 × 4 生徒 × 8 単位） |
| 記録 3（frozen.json に効果の見込み） | detect_expected.json に分け，frozen.json には sha256 と descriptive の真偽だけを残す | frozen.json の中身 | expected_effect の鍵は無い |
| 記録 5（設計 8 節と 3 節の食い違い） | 8 節の行に「第 3 版で記述だけに替えた」と書き足した | 文を照合した | 合う |
| 記録 6（detect と eval で台本が同じかを照合しない） | frozen.json に dai3.py・dai1_stats.py・dai_solver.py・dai1.py の sha256 とエンコーダを書き，eval で照合して違えば止まる（本番以外でも） | 試走の eval が通った | 合う |

変わっていない部分の再確かめ：恒等式「和 −（位置 + 方向）」の最大は 3.6e-15（336 行）．refit の値は感情ごとの再現率を 2 組で平均してから感情で平均する形に変わったが，どちらの組でも全感情が 10 本以上あるときは前の形と同じ値になり，作り直しとの差は 1.4e-14 だった．

## 記録

### [記録 a] 試走では確かめ 1 の不合格による族 5 の「判定できない」の経路を通っていない
- 重さ：記録する
- 該当箇所：dai3.py:1437–1439
- 内容：試走の check1_pair_ok はすべて True だった．`x["check1_pair_ok"] is True` の分岐は読んで確かめた．validity_ok は True・False・None だけを返す（all の戻り値は bool）．

### [記録 b] MOSEI と MELD の eval の経路は今回は流していない
- 重さ：記録する
- 内容：eval_subsets を subset_masks に書き換えた部分は，label_tables の中で本番の 4 コーパス（MOSEI のチャンネル，MELD の face_single を含む）で動いた．disjoint_halves も本番のラベルの 4 コーパスで動いた．MOSEI と MELD に見立てた eval は書いた役の試走（scratchpad/dai3_trial/emulate2.log．oracle_grid と speaker_disjoint の表が出ている）だけで見た．その試走のコミットは確かめていない．

### [記録 c] 所要が増えた
- 重さ：記録する
- 内容：試走の 1 単位は AFEW で 22〜24 → 27〜29 秒，IEMOCAP で 13 → 18 秒になり，20〜40% 増えた．本番は W* の格子が 14 点 × 4 種類（試走は 4 点）なので，さらに増える．第 1 回の見積もりを 1.3〜1.5 倍に直すと，eval の全部は A5000 で R = 10 のとき 10〜15 時間，R = 20 のとき 21〜30 時間になる．H200 では 1.5〜3 倍速いと置いて，R = 10 で 4〜10 時間，R = 20 で 7〜20 時間である．validate と detect の見積もりは変わらない（validate は W* の分だけ +25%）．pickle は MOSEI で 1 単位 15〜20 MB，全体で 3 GB 以下と見込む．本番の前に，H200 で MOSEI の 1 単位を流して測ることを勧める．

当てた項目で指摘の無いもの：I1，I3（話者が重ならない 2 組も b* を組 A のテストのラベルで解き，組 B で測るだけ），I7，I8，I9，I11，I12，I14，I17，R1，R2，R4，R12．
