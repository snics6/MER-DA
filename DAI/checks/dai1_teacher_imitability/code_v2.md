# 関門 1 の再検査：実装と出力（計算の後の解析 dai1_teacher_imitability，2026-11-16，コミット 1dba48f）

## 判定
判定：直してから通す．コミット 1dba48f の台本は 62 行目の `r.T`（pandas の行の転置で，列 T ではない）で落ちて何も書かないが，ここを `r["T"]` に替えた写しでは，K 本を除いた R・D の作り，frac_changed の印，組の中の比べがどれも設計どおりに動き，保存値と 10⁻¹³ 以内で一致した．

再検査した差分：`git diff 2a0d326 1dba48f -- tools/dai1_teacher_imitability.py`（作業木に差分なし）．前の報告は `checks/dai1_teacher_imitability/code.md`．

## A．前の指摘への対応

| 前の指摘 | 直した箇所 | 確かめた結果 |
|---|---|---|
| 1．frac_changed < 0.40 の点の印と，除いた版の要約 | :74〜78（`assumptions.csv` の kind = frac_changed，version = shufcl を (T, K, teacher) で結ぶ），:87〜88（subset = all_points と frac_ok） | 合う．48 点すべてに frac_mean が付き（欠け 0），(T, K, target) あたりの行は 1 つ．frac_low は 12 点で，前の報告で数えた 6 組 × 2 生徒と同じ．要約は 6 行（K = 5，25，all × 2 つの subset），n は 24，24，48，18，18，36 |
| 2．K 本を除いた候補の上の R・D（R1 と R0） | :55〜73 | 作りは合う（下の照合）．区間は付かない（指摘 3） |
| 3．（コーパス，K，生徒）の中の 2 つの先生の比べ | :91〜100 | 作りは合う．組の数は all_points で K ごとに 12，frac_ok で 7（AFEW の 2 組と MOSEI の 3 組は少なくとも一方の先生に印が立つので落ちる）．組の中の D_nk_R1 の差と gain の差に同点は無い（符号の比べで 0 = 0 になる組は無い） |
| 前の記録の項（来歴の上書き，pos_other_n，pivot の集計） | 直していない | 記録の項のまま残る（下の指摘 5） |

## A．小さく流した結果

- コミットの台本そのもの：`./.venv/bin/python -W default tools/dai1_teacher_imitability.py --out-dir checks/dai1_teacher_imitability/code_run_v2/out` は 2.1 秒で `TypeError: unhashable type: 'Series'`（:62 `frozen[r.T]`）で止まり，何も書かなかった．
- 直した写し：`checks/dai1_teacher_imitability/code_run_v2/patched.py`（`r.T` を `r["T"]` に替え，ROOT を DAI の直下に固定しただけ．`diff` は 3 行）を同じ引数で流すと 9.9 秒で終わり，`teacher_imitability.csv`（48 行，27 列）と `teacher_imitability_summary.csv`（6 行）を書いた．警告は `frozen.json` を閉じない ResourceWarning の 1 件だけ．数値の列に NaN と Inf は無く（NaN は文字の列 excluded の空欄だけ），要約に NaN の列は無い．
- 照合（`code_run_v2/verify.py`．所要 2 分 56 秒，CPU．画面に出したのは差・数・真偽だけで，相関・中央値・組の一致の数は出していない）：
  - **候補の全体で作ると descriptive.csv と一致するか**：台本と同じ作り（単位ごとに `dai_solver.uar(目標, 予測)` を取り，50 単位で平均する）で nk の印を外して作ると，48 点 × R1・R0 × R・D_shufcl のすべてで descriptive.csv（subset = all）の値との差の最大は 1.4 × 10⁻¹⁴ だった．`dai_solver.uar` は目標に現れた感情だけで平均するので，実験 14 の `uar_present` と同じ式であり，各分割の種の数が 10 でそろうので，単位の単純な平均は `combo_stats` の theta（分割ごとの平均の平均）と同じになる．
  - **K 本を除いた値の作りが保存値の作りと合うか**：台本の R_nk_R1，D_nk_R1，R_nk_R0，D_nk_R0 を，実験 14 の `dai14.cell(..., K_nk=K)` と `dai14.combo_stats` の effect と比べると，48 点 × 4 列のすべてで差の最大は 2.1 × 10⁻¹⁴ だった．
  - 結びつけ：teacher_margin.csv との margin・gain の差は 0.0（48 点），鍵の重複は 0．
- 試しの出力は `checks/dai1_teacher_imitability/code_run_v2/` に置いた（git に入れない）．`results/` には何も書いていない．

## A．本番の規模の見積もり

直した後の本番も CPU で約 10 秒．GPU は使わない．

## B．予想される指摘と，出力から答えられるか（前の表からの変化だけ）

| 表・図 | 予想される指摘 | (a)／(b)／(c) | 変化 |
|---|---|---|---|
| 48 点の表と要約 | D が作りの上で 0 に寄る点を除いたら | (a) | 出力の列と要約の行になった |
| 同 | margin と同じ発話の上の R・D で | (a) | 出力の列になった（区間は無い．区間が要るなら `dai14.combo_stats` に通す．指摘 3） |
| 同 | 学び直しに近い R0 で | (a) | R_nk_R0，D_nk_R0 の列になった．要約は R0 の順位相関を出さない（出力の列から数秒で出せる） |
| 同 | コーパスの違いを除いた比べ | (a) | 組の中の比べになった．(T, K) の中で中心をそろえた順位相関は無い（出力の列から数秒で出せる） |
| 同 | frac_ok の組の比べはどのコーパスから来るか | (a) | frac_ok の 7 組には MOSEI の組が 1 つも無い（3 組とも印の立つ先生を含む）．AFEW は 1 組（V の生徒）だけ残る．出力の T 列で内訳を出せる |

「止める」にあたる (b) は無い．

## 指摘

### [I（9.3 節）] コミットの台本が 62 行目で落ちる
- 重さ：直してから進む
- 該当箇所：tools/dai1_teacher_imitability.py:62〜63 `for rr in range(int(frozen[r.T]["n_seeds_eval"])):`，`f"{r.T}_f{f}_r{rr}.npz"`
- 問題：`m.iterrows()` の `r` は pandas の Series で，`r.T` は列 T ではなく Series の転置（Series そのもの）を返す．`frozen[r.T]` は `TypeError: unhashable type: 'Series'` で止まる（流して確かめた）．台本は何も書かずに止まるので，誤った数値は出ない．
- 計算の後で直せるか：直せる．
- 求める対応：`T = r["T"]` を先に取り出し，62・63 行目で `T` を使う（f 文字列の中に同じ引用符を入れる書き方を避ける）．この 1 か所だけを替えるなら，替えた写しで上の照合を済ませたので，再検査は要らない．本体は直した後に一時の置き場で流し，48 行と要約 6 行が書かれることを確かめてから本番の置き場に書く．

### [S6（9.3 節）] 要約の中央値は候補の全体の D，順位相関と組の比べは K 本を除いた D で，2 つが混ざる
- 重さ：直してから進む
- 該当箇所：tools/dai1_teacher_imitability.py:97〜102（D_nk_R1）と :104〜109（D）
- 問題：前の報告の指摘 2 は margin と同じ発話の集合にそろえることを求めた．新しい列 spearman_gain_Dnk，spearman_resid_Dnk，pairs_* は D_nk_R1 を使うが，問いに直接答える pos_hurt_D_median，pos_gain_ci_pos_D_median，pos_other_D_median は候補の全体の D のままである．docstring（:7，:11）も「候補の全体」「subset = all」のままで，どちらを主にするか，入力に `assumptions.csv` と実験 14 の単位の npz が増えたことが書かれていない．
- 計算の後で直せるか：直せる（数秒）．
- 求める対応：3 つの中央値を D_nk_R1 でも出す（列を足すか，主を D_nk_R1 に替えて D を副にする）．docstring に主の量，入力の 4 つ，出力の列（R_nk_*，D_nk_*，frac_mean，frac_low）と要約の subset の意味を書く．

### [D（9.3 節）] K 本を除いた R・D に区間が無い
- 重さ：記録する
- 該当箇所：tools/dai1_teacher_imitability.py:71〜72（単位の平均だけ）
- 問題：D_nk_R1 などは点の値だけで，区間が無い．候補の全体の D_lo・D_hi は残るが，集合が違う．
- 計算の後で直せるか：直せる（`dai14.cell(..., K_nk=K)` と `dai14.combo_stats` で作れることを照合で確かめた．48 点 × 2 読み出しで約 3 分，CPU）．
- 求める対応：D_nk の符号を区間で言う文を書くなら区間を足す．書かないなら，結果の文で D_nk は点の値だけだと書く．

### [S（9.3 節）] frac_ok の版の gain_resid は全部の点で当てた直線の残りである
- 重さ：記録する
- 該当箇所：tools/dai1_teacher_imitability.py:80〜83，:88
- 問題：frac_ok の行の spearman_resid_* と pairs_D_resid_agree は，印の立つ点も含めた 24 点の直線の残りを使う．除いた 18 点で当て直した残りではない．
- 計算の後で直せるか：直せる．
- 求める対応：どちらでもよいが，結果の文に「残りは全部の点で当てた直線から取った」と書くか，subset ごとに当て直す．

### [I9 ほか（前の記録の項）] 来歴の上書き，pos_other_n，pivot の集計
- 重さ：記録する
- 該当箇所：:85 `m.assign(commit=cm, dirty=dt)`（teacher_margin.csv の commit dd309bb を上書きする），:109（pos_other_n が無い），:49（`pivot_table` の既定は平均）
- 問題・求める対応：前の報告 `code.md` の指摘 5・7 と同じ．加えて，frac_ok の版は MOSEI の組を含まず，AFEW の組は 1 つだけなので，組の比べの数を書くときは組の数（7）とコーパスの内訳を並べる．K = 5 と K = 25 の組は同じ単位を共有し，同じ先生が 2 つの組に入るので，組の一致の数を独立な数として扱わない．

## 目録に無い種類の不備

- **pandas の行の属性で列を読む．**`row.T` のように，DataFrame や Series の属性と同じ名前の列（T，K の後に足される列など）を属性で読むと，列ではなく属性を返す．コーパスの列の名前が `T` なので，この誤りは DAI の台本のどこでも起こりうる．今回は型の誤りで止まったが，属性が同じ型の値を返す場合は黙って誤った値になる．案：9.3 節の I の項に「pandas の行や表の列は `row["T"]` の形で読む．`T`・`size`・`shape`・`index`・`name` などの属性と同じ名前の列を属性で読まない．台本の検査では `\.T\b` などを検索する」を足す．
