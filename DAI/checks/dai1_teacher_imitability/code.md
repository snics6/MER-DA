# 関門 1：実装と出力（計算の後の解析 dai1_teacher_imitability，2026-11-16，コミット 2a0d326）

## 判定
判定：直してから通す．結びつけ（48 点，鍵，値の写し）と「実験 1 の先生 single_m ＝ 実験 14 の判断 J_m」は数値の照合で確かめたが，D が作りの上で 0 に寄る 12 点（frac_changed < 0.40）の印が無く，margin（K 本を除く）と D（K 本を含む）の発話の集合がずれ，要約の残りの順位相関がコーパスの違いを含むので，この 3 点を出力に足してから本番の置き場に書く．

検査した台本：`tools/dai1_teacher_imitability.py`（76 行．コミット 2a0d326 で作業木に差分なし）．入力：`results/dai1/stats/teacher_margin.csv`（コミット dd309bb の台本 `tools/dai1_teacher_margin.py` の出力），`results/dai14/summary/descriptive.csv`（コミット fbd6c07，dirty は偽）．実験 21 の設計は読んでいない．

## A．設計と台本の対応表

設計の文書は無く，台本の docstring と依頼の文が設計にあたる．

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 点：teacher = single_m，生徒 A・V・T，K = 5・25，除外の印が空 | dai1_teacher_imitability.py:44 | 合う．teacher_margin.csv の single_m の行は生徒 A・V・T で 16 行ずつ（計 48），AVT の生徒の 24 行は除かれる．single_m の行に excluded の印は 0 行 |
| 実験 14 の量：readout = R1，subset = all，quantity = R と D_shufcl | :46 | 合う．鍵 (T, K, student, target, quantity) の重複は 0，single_m の行は 96（48 × 2） |
| 鍵：T，K，student，target ↔ teacher | :47〜50 | 合う．target の名前は実験 14 でも `single_<m>` で，teacher_margin.csv の teacher と同じ綴り．生徒 s と判断 m の向きも同じ（実験 14 の target は「生徒 s が再現する判断 J_m」，実験 1 の tt は「生徒 s を J_m で学び直す」） |
| D の区間 lo・hi | :48〜49 | 合う．値は descriptive の lo・hi と差 0.0．ただしこの区間には分け方の揺れが入らない（実験 14 の設計 4 節．下の指摘 6） |
| 48 点がそろうことの確かめ | :51〜52 | 合う（左の結合のあとに欠けと点の数を確かめて止める） |
| margin で説明しきれない gain の残り（K ごとの最小二乗） | :53〜57 | 作りは書いたとおり．24 点（4 コーパス × 3 生徒 × 2 先生）に 1 本の直線を当てる．下の指摘 3 |
| 要約の数え（順位相関 4 つ，margin > 0 の点の 3 区分の数と D の中央値） | :60〜71 | 作りは書いたとおり．3 区分のうち「区間が 0 を含む」点の数の列が無い（指摘 7） |
| 来歴（コミット） | :58〜59，:71 | 一部欠ける．指摘 5 |
| 上書きしない | :40〜42 | 合う |

（1）**実験 1 の先生 single_m と実験 14 の判断 J_m は同じ分類器である．**読んだ箇所と照合の数値：

- 実験 1：`exp/dai1.py:474` `tts = {f"single_{m}": tk[f"none_{m}"][0] for m in "AVT" if m != s}`．`tk` は `exp/dai1.py:700` の `build_tk(u, p1, p2, tether)`（段 1・段 2 の保存した float32 の W・b から作り直す．`exp/dai1.py:310〜319`）の K ごとの辞書で，`[0]` は候補の上の確率．先生の UAR（uar_pool）は `exp/dai1.py:491` で K 本を除いた候補（`nk = np.setdiff1d(np.arange(u.n), u.kpos[K])`，:450）の上で測る．
- 実験 14：`exp/dai14.py:437` `TK = dai1.build_tk(u, sv.p1, sv.p2, sv.teth)`，:490 `dai3.pool_teachers(sv, tk, s, K, names)`，`exp/dai3.py:1051` `Q[name] = tk[f"none_{name[7:]}"][0]`．単位は `dai3.Saved(OUT1, "eval", T, T, f, r, fz1T)`（実験 1 の評価の回の単位．コミット ba98974）．判断は `argmax`（:492）．
- 同じ単位（評価の回の (T, f, r)），同じ K，同じ候補の並び：`tools/dai1_teacher_margin.py:57` が実験 14 の npz の `sid` と実験 1 の `sample_pool` の一致を単位ごとに確かめて止める作りで，teacher_margin.csv はこの確かめを通って書かれている．
- 数値の照合（`checks/dai1_teacher_imitability/code_run/same_judge.py`）：teacher_margin.csv の single_m の teacher_uar_pool と，実験 14 の `conditions.csv` の uar_judge（J_m の K 本を除いた候補の上の UAR．単位の平均）は 48 行すべてで結びつき，差の最大は 1.4 × 10⁻¹⁴．同じ (T, K, m) の先生の UAR は 2 つの生徒で差 0.0．
- 違い（同じ分類器だが，使い方が違う）：実験 1 の学び直しは J_m の**確率**を感情ごとの総量をそろえた柔らかいラベル（`make_G` の "tt"，`S.soft_rows(Q, ρ n)`，ρ = 1）として全部の候補で使い，出発点 `u.start[s]` と引き戻し `tether[(s, K)]` から解く．実験 14 の R1 は J_m の**argmax**を硬いラベルにし，主成分で 512 次元にそろえた特徴の上で，出発点 0，C′ を選び，候補を 2 組に分けた交差の予測で測る．書くときの注意は下の B の表と指摘 2．

（2）**結びつけの鍵と 48 点．**鍵 (T, K, student, teacher) は teacher_margin.csv の側でも descriptive.csv の側でも重複が無く，結合は 48 点すべてで相手を持つ．内訳は (コーパス, K) ごとに 6 点（生徒 3 × 先生 2）で 8 組，(生徒, 先生) ごとに 8 点（コーパス 4 × K 2）で 6 組．先生が生徒自身のモダリティである点は 0．R・D・D_lo・D_hi は descriptive の値と差 3.6 × 10⁻¹⁵ 以下，margin・gain は teacher_margin.csv の値と差 0.0（`code_run/structcheck.py`）．

## A．小さく流した結果

- コマンド：`./.venv/bin/python -W default tools/dai1_teacher_imitability.py --out-dir checks/dai1_teacher_imitability/code_run/out`．所要 2.8 秒（CPU）．警告は出なかった．
- 出力：`code_run/out/teacher_imitability.csv`（48 行，21 列），`code_run/out/teacher_imitability_summary.csv`（3 行：K = 5，25，all）．数値の列に NaN と Inf は無い（NaN は文字の列 excluded の空欄 48 個だけ）．要約の列に NaN は無い．
- 画面に出したのは，行と列の数，鍵ごとの点の数，重複の数，結合の差，コミットの列，要約の n と n_margin_pos（K = 5・25 とも 12．teacher_margin.csv から数えられる数）だけである．順位相関と D の中央値は画面に出していない．
- 退化した場合の検算：この台本は表を結ぶだけで学ぶ部分が無いので，代わりに「結合した値が元の表の値と同じ」ことを上のとおり確かめた．
- 保存済みの値の再現：`code_run/nk_feasible.py` で，実験 14 の単位の npz（`results/dai14/units/`）から `dai14.cell` と `dai14.combo_stats` で R と D_shufcl を作り直すと，AFEW と MOSEI の K = 5・25，(A ← single_V) と (V ← single_A) の 8 組すべてで descriptive.csv の値と 10⁻⁹ 以内で一致した．同じ作りで K 本を除いた候補の上の D と区間も有限の値で出せた（値は画面に出していない）．所要 36 秒（2 コーパス）．
- 試しの出力は `checks/dai1_teacher_imitability/code_run/` に置いた（git に入れない）．`results/` には何も書いていない．

## A．本番の規模の見積もり

本番も CPU で 3 秒．指摘 1・2 の直し（K 本を除いた R・D を実験 14 の単位から作る）を足すと，4 コーパスの単位の読み込みと 48 点 × 2 つの量の数えで 1〜3 分（CPU）．GPU は使わない．

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 48 点の表と要約 | 「D が小さい点は，入れ替えで目標がほとんど変わらなかった点ではないか」 | 実験 14 設計 8 節（frac_changed < 0.40），methods 9.3 D 項 | (a)．`results/dai14/summary/assumptions.csv` の kind = frac_changed，version = shufcl，(T, K, target) を結ぶ．0.40 未満は 24 組のうち 6 組（AFEW の single_V の K = 5・25，MOSEI の single_A・single_V の K = 5・25）で，各組が 2 つの生徒に当たるので **48 点のうち 12 点**．出力の列に足すべき（指摘 1） | — | 数秒 |
| 同 | 「margin は K 本を除いた候補，D は K 本を含む候補．同じ発話の上で比べよ」 | 依頼の (3)，S6 | (a)．実験 14 の npz の `nk__K{K}`，`tgt__K{K}__single_m__{true,shufcl}`，`K{K}__main__R1__{s}__single_m__{true,shufcl}`，`cl` から `dai14.cell(..., K_nk=K)` と `combo_stats` で作れる（`code_run/nk_feasible.py` で作れることを確かめた）．K 本の割合は AFEW の K = 25 で 100 / 632 = 16%，IEMOCAP 4.5%，MELD 1.7%，MOSEI 0.7%，K = 5 は AFEW 3.2% 以下．出力の列に足すべき（指摘 2） | — | 1〜3 分 |
| 同 | 「学び直しは出発点と引き戻しから解く．R1（出発点 0，512 次元，C′ を選ぶ）ではなく，学び直しと同じ作りの R0 で見たらどうか」 | 実験 14 の R0（実験 1 の作り），C 項 | (a)．descriptive.csv の readout = R0，subset = all の R と D_shufcl を同じ鍵で結ぶ（single_m の行はある）．列に足すべき（指摘 2） | — | 数秒 |
| 同 | 「残りと D の順位相関はコーパスの違い（コーパスごとに D も gain も水準が違う）を拾っているだけではないか」 | S 項，A 項 | (a)．出力の T 列で (T, K) の中で順位を取るか中心をそろえる．または (T, K, 生徒) の中の 2 つの先生の比べ（どちらの先生が D が大きく，どちらが gain − margin の当てはめの残りが大きいか）の符号の一致を 24 組で数える．台本に足すべき（指摘 3） | — | 数秒 |
| 同 | 「AFEW の言葉は Whisper の書き起こしで，声と言葉は同じ信号から来る．A ← T，T ← A の点を除いても同じか」 | 実験 14 設計 5 節（AFEW を除く理由），L 項 | (a)．出力の T，student，teacher で AFEW の (A, single_T)，(T, single_A) の 4 点を除いて数え直す | — | 数秒 |
| 同 | 「MELD は 2 つの組が同じ対話を共有する．共有を除いた D で見たら」 | 実験 14 設計 5 節（MELD），S6 | (a)．descriptive.csv の subset = nonshared の行（MELD の 12 点） | — | 数秒 |
| 同 | 「AFEW の D は #shufage でも見よ」「#shuf，#shuffine で見たら」 | 実験 14 設計 3・5 節 | (a)．descriptive.csv の quantity = D_shufage（AFEW），D_shuf，D_shuffine（IEMOCAP・MOSEI） | — | 数秒 |
| 同 | 「J_m の判断の感情の偏り（現れた感情の数）で R の偶然の水準が変わる」 | D 項 | (a)．実験 14 の単位の csv の min_class と npz の `tgt__K{K}__single_m__true` から現れた感情の数を数える．D は差なので偶然の水準の違いはおおむね消えるが，R を使うなら要る | — | 数秒 |
| 同 | 「点どうしは同じ単位を共有する．順位相関に区間を付けよ」 | S 項（独立でない点の数え） | (a)（学び直しは要らないが新しい台本が要る）．実験 1 の単位の p3 csv の tt の行（単位ごとの test UAR）と none の行，`_teacher_stats` の uar_pool，実験 14 の npz の単位ごとの数があるので，塊を除く jackknife で区間を作れる．この台本の範囲では「検定ではない」と書いてある（:13） | — | 新しい台本で数分 |
| 同 | 「確率で測った D（argmax ではなく）で見たら」 | 実験 14 設計 3 節（確率の保存） | (a)（K = 5 だけ）．実験 14 の npz の `prob__K5__main__R1__{s}__single_m__{ver}`．K = 25 は保存に無い | K = 25 は (c) とする（範囲の外と書く） | — |
| 要点 6 の点（margin > 0 で gain の区間の上端 < 0）との対応 | 「要点 6 の 36 点の多くは src_<コーパス>_T や積の先生，AVT の生徒の点で，この 48 点に入らない．残りの点の再現できる度合いは」 | L4（計算しない組） | (b)．実験 14 は判断 single_m と prop_one の R だけを保存している．src_* や pair_* の先生の R は保存に無い | src_*・pair_* の判断について実験 14 の R1 と #shufcl を同じ作りで計算する | 実験 14 の本計算は 4.5〜5 時間（GPU）．先生の種類を増やすとそれに比例して増える．**この解析では範囲の外として，48 点（単独のモダリティの先生と単独のモダリティの生徒）に限ると書けばよい**（指摘 4） |
| 同 | 「AVT の生徒について」 | L4 | (c)．実験 14 は融合を範囲の外にした（設計 7 節） | — | — |
| 同 | 「再現しにくいから下がったのか（原因か）．再現できる発話だけで学び直したら」 | C 項 | (c)．学び直しが要る．この台本は記述で，結果の文で原因の主張をしない（既存の結果の文の 3 節の「1 つの可能な説明」と同じ扱い） | — | — |

「止める」にあたる (b) は無い．(b) の 1 行（src_*・pair_* の先生の R）は，主張を 48 点に限れば学び直しも新しい計算も要らない．

## 指摘

### [S6・D（9.3 節）] D が作りの上で 0 に寄る 12 点の印が出力に無い
- 重さ：直してから進む
- 該当箇所：tools/dai1_teacher_imitability.py:46〜50，:62〜70
- 問題：実験 14 の設計 8 節は「frac_changed の平均が 0.40 未満のときは D が 0 に寄るので，そのコーパスの『判定できない』を『差が無い』の根拠に使わない」と定める．この台本は `assumptions.csv` を読まず，D の小ささをそのまま「再現しにくい」の量として順位相関と中央値に入れる．流して数えると，single_m の (T, K, target) 24 組のうち 6 組（AFEW の single_V の K = 5・25，MOSEI の single_A・single_V の K = 5・25）が 0.40 未満で，48 点のうち 12 点にあたる（`code_run/same_judge.py`）．要約の「下がった点の D の中央値」が小さく出たとき，それが入れ替えの作りから来たのか，再現しにくさから来たのかを区別できない．
- 計算の後で直せるか：直せる（CPU で数秒）．
- 求める対応：(T, K, target, version = shufcl) で `assumptions.csv` の frac_mean を結び，`frac_changed` と `frac_lt_040` の列を足す．要約は全部の点の版と，frac_lt_040 の点を除いた版の両方を出す．あわせて C′ の端の旗（kind = C_edge．single_m では flag 0 組，flag_old 2 組：AFEW の K = 25 の (T ← single_V)，(V ← single_T)）も列に足す．

### [S6（9.3 節）] margin と D が違う発話の集合の上の量である
- 重さ：直してから進む
- 該当箇所：tools/dai1_teacher_imitability.py:46（subset = all），tools/dai1_teacher_margin.py:9・14（margin は K 本を除いた候補）
- 問題：margin は K 本を除いた候補の上の UAR の差，R と D は K 本を含む候補の全体の量（実験 14 設計 2 節「K 本のラベル付きの発話も候補に含まれ，R の計算に入る」）．K 本の上では J_m は自分の学習の発話の判定なので本当の感情に近く，R の目標の性質が残りの候補と違う．K 本の割合は AFEW の K = 25 で 16%（100 / 632），IEMOCAP の K = 25 で 4.5%，ほかは 3.2% 以下．実験 14 の npz から K 本を除いた R・D を学び直さずに作れることを確かめた（`code_run/nk_feasible.py`．候補の全体の値は descriptive.csv と 10⁻⁹ 以内で一致）．あわせて，実験 1 の学び直しは出発点 `u.start[s]` と引き戻しから解くので，学び直しの作りに近いのは R1 ではなく R0 である．R0 の R・D は descriptive.csv にある．
- 計算の後で直せるか：直せる（CPU で 1〜3 分）．
- 求める対応：(i) K 本を除いた候補の上の R1 の R と D_shufcl（と区間）を `dai14.cell(..., K_nk=K)` と `dai14.combo_stats` で作り，`R_nk`・`D_nk`・`D_nk_lo`・`D_nk_hi` の列に足す．どちらを主にするかを docstring に書く（margin と同じ集合にそろえるなら nk を主にするのが筋が通る）．(ii) readout = R0 の R と D_shufcl を `R_R0`・`D_R0` の列に足し，要約の順位相関を R0 でも出す．

### [A・S（9.3 節）] margin の残りの直線を 4 コーパス × 3 生徒にまとめて当てている
- 重さ：直してから進む
- 該当箇所：tools/dai1_teacher_imitability.py:55〜57 `for K, x in m.groupby("K"): b = np.polyfit(x.margin, x.gain, 1)`，:66
- 問題：K ごとの 24 点に 1 本の直線を当て，その残りと D の順位相関を取る．コーパスごとに gain の水準も D の水準も違う（例：frac_changed の 0.40 未満は AFEW と MOSEI に集まる）ので，この順位相関は「同じコーパスの中で再現しにくい先生ほど下がる」とも「D の低いコーパスで gain が下がる」とも読める．48 点で 2 つを分けられない．既存の結果の文の要点 9 も，傾きはまとめた値で内訳が違うと断ったうえで（コーパス，生徒）の中で比べている．
- 計算の後で直せるか：直せる（数秒）．
- 求める対応：(T, K, 生徒) ごとに 2 つの先生がある（24 組）ので，組の中で「D が大きい先生のほうが gain − margin が大きい（または gain の残りが大きい）」組の数を数える列を要約に足す．あわせて，(T, K) の中で中心をそろえた margin・gain・D での順位相関を出す．まとめた直線の版は記述として残してよいが，結果の文には組の中の数えを先に書く．

### [L4（9.3 節）] 要点 6 の点のうち，この 48 点に入らないものがある
- 重さ：記録する
- 該当箇所：tools/dai1_teacher_imitability.py:4〜8（問い），:44
- 問題：docstring の問いは「先生のほうが当たるのに学び直しで下がった点（要点 6）は，先生の判断を生徒の特徴で再現しにくい点か」である．要点 6 の 36 点には src_<コーパス>_T などのソースの先生，積の先生，AVT の生徒の点が含まれる（結果の文 `docs/結果/計算の後の解析_先生と自分の差とK10_2026-11-16.md` の要点 6）．実験 14 の R は single_m と prop_one の判断にしか無いので，この台本が答えられるのは 36 点のうち teacher が single_m で生徒が A・V・T の点だけである（その数は要約の pos_hurt_n の K = all の行）．
- 計算の後で直せるか：書き方で直せる．ほかの先生の R を作るには実験 14 の作りの再計算（GPU で時間の単位）が要る．
- 求める対応：docstring と結果の文に「単独のモダリティの先生と単独のモダリティの生徒の 48 点に限る．要点 6 の 36 点のうち x 点がこれに入る」と書く．

### [I9（9.3 節）] 入力の来歴が出力から消える
- 重さ：記録する
- 該当箇所：tools/dai1_teacher_imitability.py:50 と :59 `m.assign(commit=cm, dirty=dt)`，exp/dai1.py:72〜73
- 問題：teacher_margin.csv の commit 列（dd309bb）は結合で m に入り，:59 で今の HEAD（試しでは 2a0d326）に上書きされる．実験 14 の commit（fbd6c07）は書かれない．`dai1.git_commit()` の dirty は `exp splits measure features data/metadata` だけを見るので，`tools/` の台本を直した作業木でも偽になる．DAI の取り決め（`results/<実験>/provenance.json` に入力の sha256 を残す）の代わりになるものが無い．
- 計算の後で直せるか：直せる．
- 求める対応：上書きの前に `commit` を `margin_commit` に，descriptive の commit を `dai14_commit` に写す．2 つの入力の sha256 と，この台本の sha256 を列か別の json に残す．dirty に `tools/` を足すか，この台本の git の状態を別に書く．

### [D（9.3 節）] D の区間に分け方の揺れが入らない
- 重さ：記録する
- 該当箇所：tools/dai1_teacher_imitability.py:48
- 問題：descriptive.csv の lo・hi は分け方の揺れを足していない（実験 14 設計 4 節「descriptive・conditions・robustness の区間には分け方の揺れが入らない」）．実験 14 の主の判定（揺れを足した区間）より狭い．
- 計算の後で直せるか：書き方で直せる．
- 求める対応：D_lo・D_hi を使う文には「候補を 2 組に分ける分け方の揺れを含まない区間」と書く．D の符号を区間で言う文は書かないか，実験 14 の predictions.csv の主の区間の作りを当てる（P3・P3b の量 D(A ← T)，D(A ← V) には揺れを足した区間がある）．

### [記録] 要約の小さい数と，数えの列の欠け
- 重さ：記録する
- 該当箇所：tools/dai1_teacher_imitability.py:62〜70
- 問題：margin > 0 の点（K ごとに 12 点）を 3 区分に分けて D の中央値を出すので，区分あたり数点の中央値になりうる．pos_hurt_n と pos_gain_ci_pos_n はあるが，3 つ目の区分（区間が 0 を含む）の数の列が無い．`pivot_table` の既定の集計は平均で，鍵が重複すると黙って平均する（今の入力では重複 0 を確かめた）．
- 計算の後で直せるか：直せる．
- 求める対応：`pos_other_n` の列を足す．結果の文では中央値と一緒に区分の点の数を書く．`pivot_table(..., aggfunc="first")` の前に鍵の重複を確かめて止める．

### 書くときの注意（指摘ではない）
- 先生の分類器は同じでも，学び直しは確率の柔らかいラベル（全部の候補，出発点と引き戻し），R は argmax の硬いラベル（候補を 2 組に分けた交差の予測）である．「再現できる度合い」は学び直しの中で起きることそのものではないと書く．
- 48 点は同じ単位・同じテストを共有し，K = 5 と K = 25 の点も同じ単位から来る．K = all の行の順位相関を 48 点の独立な証拠として書かない（台本の :13 と同じ）．
- AFEW の言葉は Whisper の書き起こしなので，AFEW の (A ← single_T)・(T ← single_A) の R・D は同じ信号の上の量である．B の表の (a) の版（4 点を除く）を並べる．

## 目録に無い種類の不備

- **2 つの保存を結ぶ解析で，量を測った発話の集合がそろっているか．**9.3 節の S6 は「確認に使うデータが設計に使ったデータと重ならないか」を見るが，結ぶ 2 つの量が違う部分集合（K 本を含むか除くか，共有の対話を含むか）の上で測られていることは見ない．案：9.3 節の R の項に「計算の後の解析で 2 つの結果ファイルの量を結ぶときは，各量の発話の集合（K 本，テスト，部分集合）と分類器の作り（硬いラベルか確率か，出発点）を表にし，違うなら同じ集合の版を作るか違いを書く」を足す．
- **前の実験の仮定の旗の引き継ぎ．**前の実験が「この旗の立った量を『差が無い』の根拠にしない」と決めた量（frac_changed < 0.40 の D）を，後の解析が旗を持たずに読む．案：「前の実験の結果ファイルを入力にする解析は，その実験の仮定の表（assumptions.csv など）の旗を点ごとに結び，旗の立った点を除いた版を並べる」を足す．
