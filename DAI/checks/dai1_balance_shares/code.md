# 関門 1：実装と出力（計算の後の解析 dai1_balance_shares，2026-11-16，コミット 990df58）

## 判定
判定：直してから通す．台本は kd(one, T = 1, ρ = 1) と joint(one, ρ = 1, a = 1) の違いを「感情ごとの重みの合計をそろえるか」だけに限って正しく比べており学び直しも要らないが，答えの割合の作りが要点 17 の出所（emotion_margin_confusion.csv.gz）と違い（IEMOCAP・MELD で最大 0.016 ずれる），MOSEI の区間が保存済みの recall.csv・ablation.csv と別の乱数の列になるため，同じ量に 2 つの値が並ぶ．

検査の対象：`tools/dai1_balance_shares.py`（990df58 に入った版．作業木の差分なし）．台本が呼ぶ `exp/dai1_desc.py`（Unit，Ctx，P3，fam_rows，emo_of，_fmt，t_ablation，t_recall），`exp/dai1_stats.py`（counts，uar_full，effect_stats），`exp/dai1.py`（make_G，pass3，Unit の標準化），`exp/dai_solver.py`（soft_rows，fit）を読んだ．値の大小（joint と kd のどちらが高いか）は報告に書かない．

## A．設計と台本の対応表

| 設計の項目（台本の冒頭の説明） | 台本の箇所 | 合っているか |
|---|---|---|
| 比べる学び直し joint(one, ρ = 1, a = 1) | tools/dai1_balance_shares.py:71 | 合う．t_ablation の d1（exp/dai1_desc.py:658）と同じ指定 |
| 比べる学び直し kd(one, T = 1, ρ = 1) | :72 | 合う．行 d の spec_B（exp/dai1_desc.py:668）と同じ指定 |
| kd(T = 1, ρ = 1) が「同じ疑似ラベルで重みの合計をそろえない」学び直しであること | exp/dai1.py:401-406，exp/dai_solver.py:50-55，:150-166 | 合う（読んだだけ．下の A-2）|
| K 本の基準の分類器 none | :70（fam_rows の "none"．引き戻しの C は fz["tether"]） | 合う．recall.csv の row = none と同じ指定 |
| K = 5，10，25，分類器 A・V・T・AVT，4 コーパス | :39-40，:46 | 合う（DD.CORPORA = AFEW，IEMOCAP，MOSEI，MELD） |
| 単位 = 5 分割 × n_seeds_eval 反復 | :59-62 | 合う．4 コーパスとも R = 10，50 単位（balance_recall.csv の n_units = 50） |
| (i) 答えの割合 = 1 単位の答えの数 ÷ 全体を単位で平均 | :74-79 | 説明どおりに計算する．ただし要点 17 の出所の作り（単位あたりの数の平均を足してから割る）と違う（指摘 [I-定義]） |
| (ii) 感情ごとの再現率の差 − none と区間 | :84-86 | 作りは t_recall（exp/dai1_desc.py:808-812）と同じ．乱数の列だけが違う（指摘 [R-乱数]） |
| (iii) joint − kd の感情ごとの差と区間 | :84-86 | 同上 |
| 乱数の列 = 表の名前 "ablation" の番号 | :82 | 名前は同じだが，t_ablation と消費の順が違うので，ablation.csv の区間を再現しない（指摘 [R-乱数]） |
| 照合 1：4 感情の差の平均 = UAR の差（1e-9） | :87-90 | 実装は合う．θ が単位の差の平均で線形なので，この一致は emo_of と scale = 4 の換算が正しければ必ず成り立つ恒等式である（指摘 [I-照合]） |
| 照合 2：joint − kd の UAR の差 = ablation.csv 行 d の effect（1e-9） | :91-98 | 合う．点の推定だけを比べ，区間は比べていない |
| 照合 3：予測を rows.csv の UAR と照らし，欠けと食い違いが無い | :104-105，exp/dai1_desc.py:390-398 | 合う |
| 上書きしない | :50-52 | 合う．results/dai1/stats/ に balance_* は無い |
| 出力の記録（commit，dirty，units_commit） | :108-111 | units_commit = "ba98974" を書き込むだけで，単位の npz の commit と照らしていない（指摘 [記録]）．DAI の取り決めの provenance.json を書かない |

### A-2．kd(one, T = 1, ρ = 1) と joint(one, ρ = 1, a = 1) の違い（読んだだけ）

- joint：G = a · gk + soft_rows(Q, ρn)（balance = True），G2 = 0，T2 = 1（exp/dai1.py:402）．soft_rows は Q の列（感情）ごとの和で割ってから全体を ρn にそろえる（exp/dai_solver.py:53-55）．
- kd：G = gk，G2 = soft_rows(softmax(log Q / T), ρn, balance = False)，T2 = T（exp/dai1.py:404-406）．T = 1 では softmax(log Q) は Q を行ごとに正規化し直しただけ（Q.clamp(min = 1e-12) の分だけ違う）．fit の目的関数は T2² · Σ G2 · (−log softmax(z / T2)) なので（exp/dai_solver.py:162-164），T2 = 1 では G2 の項は G の項と同じ交差エントロピーになる．
- 重みの和 S = ΣG + ΣG2 は両方とも ΣgK + ρn で同じ．W0，b0，Cw，Cb は同じ solve_items の呼び出しで同じ値（exp/dai1.py:424-429，463-472）．教師 Q は両方とも teach["one"]（exp/dai1.py:448）．
- したがって，a = 1 のとき 2 つの学び直しの違いは「疑似ラベルの感情ごとの重みの合計をそろえるか（joint：各感情 ρn / 4．kd：各発話 ρ）」だけである．台本の説明は正しい．学び直しを流して数値で確かめてはいない．

### A-3．主の提案 prop との関係（流して確かめた）

fam_rows の "prop" は，4 コーパス × 4 分類器 × K = 5，10，25 の 48 条件のすべてで joint(one, ρ = 1, a = 1) と同じ指定だった（frozen.json の main は one，選んだ ρ = 1，a = 1）．したがって，この台本の joint は主の疑似ラベルで学び直した分類器そのもので，joint − none の感情ごとの差は recall.csv の family1（prop − none）と同じ量である（下の照合 1）．

## A．小さく流した結果

- コマンド：`CUDA_VISIBLE_DEVICES="" ./.venv/bin/python tools/dai1_balance_shares.py --out-dir checks/dai1_balance_shares/code_run/afew --targets AFEW`（14 秒）と，全部のコーパス `--out-dir checks/dai1_balance_shares/code_run/all`（2 分 12 秒．CPU だけ．GPU は他の計算が 81% 使っていたので CUDA を見えなくした）．
- 落ちずに最後まで動いた．警告なし．
- 行の数：balance_shares.csv 576 行（4 × 4 × 3 × 3 版 × 4 感情），balance_recall.csv 576 行（4 × 4 × 3 × 3 組 × 4 感情），NaN・Inf なし．answer_share の 4 感情の和と true_share の 4 感情の和はすべて 1．balance_check.csv の NaN は予測の照合の行の student・K・pair の空欄だけ．
- 台本の照合：emotion_mean_vs_uar の差の最大 2.3e-15，uar_vs_ablation_d の差の最大 1e-16．予測の照合は各版 600 件（12 指定 × 50 単位）ずつ調べて欠けと食い違いが 0．
- 保存済みの値との突き合わせ（`checks/dai1_balance_shares/code_run/crosscheck.py`．値の大小は見ず，差の最大だけ）：
  1. joint − none の感情ごとの差 vs recall.csv の family1（prop − none）：192 行すべて対応．effect の差は 4 コーパスとも 0．lo・hi の差は AFEW・IEMOCAP・MELD で 0，MOSEI で最大 0.143 点．
  2. joint − kd の UAR の差 vs ablation.csv の行 d：48 行．effect の差は 0．lo・hi の差は AFEW・IEMOCAP・MELD で 0，MOSEI で最大 0.0030 点．
  3. none と joint の答えの割合 vs 要点 17 の出所 emotion_margin_confusion.csv.gz（who = none_test と who = retrained_test・teacher = main_q の count_mean の列の和 ÷ 全体）：384 行．差の最大は AFEW 2e-16，MOSEI 1.9e-6，IEMOCAP 0.0097，MELD 0.0156．
  4. 同じく recall.csv と同じ単位から作った confusion.csv（none，default）と比べても，差の最大は 0.0156．
- 退化した場合の検算：kd と joint を同じにする版（balance = False の joint）は保存に無く，学び直しが要るので流していない．A-2 の読みで代える．

## A．本番の規模の見積もり

全部のコーパスで 2 分 12 秒（CPU，実時間）．GPU は使わない（import する dai_solver が CUDA を見ると 0.6 の上限を設定して文脈を作るので，本番も `CUDA_VISIBLE_DEVICES=""` を付けるとよい）．出力は 3 つの csv で合わせて 0.3 MB 未満．

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 答えの割合 | 要点 17 の割合と同じ定義か．数が合わない | 9.3 節の定義の一致，照合 claim_emotion_v3 | (a) | emotion_margin_confusion.csv.gz と同じ「単位あたりの数の平均を足して割る」形で計算する（指摘 [I-定義]） | CPU 数秒 |
| 答えの割合 | 答えの割合の差（joint − kd の Neutral など）に区間を付けよ | S 項（話者を単位にした区間），review-ml の区間の指摘 | (a) | 単位の予測・正解・塊（u.pred，u.y，u.cl）から，答えの数を塊ごとに数え，effect_stats と同じ jackknife／重みの引き直しを当てる関数を足す．今の台本は割合の点だけを出す | CPU 数分，台本の追加 1 関数 |
| 答えの割合 | 候補（ラベル無しの発話）の上での疑似ラベルの割合と，学び直した分類器の答えの割合を並べよ | 依頼の (3)，要点 17 の「候補の上の割合」 | (a) | 疑似ラベルの判定は emotion_margin_confusion.csv.gz の who = teacher_pool（K 本を除く）と，各単位の p3.csv の _teacher_stats の colsum_0〜3（確率の和）にある．学び直した分類器の候補の上の答えは，p3.npz の p3_{s}_K{K}_W・_b（float32）と sample_pool から，候補の平均・標準偏差（torch.std の不偏の形．exp/dai1.py:176）で標準化を作り直せば出せる．テストの保存済みの予測を再現できるかを先に確かめる | CPU 数分．標準化の作り直しの照合が要る |
| 取り違え | joint・kd・none の取り違えの表を出せ | 依頼の (3)，review-affect の感情ごとの解析 | (a) | none と joint は confusion.csv の row = none・default（default = joint(one,1,1)）にある．kd は無いが，単位の予測と正解から作れる．台本に 3 版の 4 × 4 の表（単位あたりの平均）を足すとよい | 2304 行，CPU 数秒 |
| 感情ごとの差 | 主の提案 prop で同じ比べをせよ | 依頼の (3) | (a) | 不要．48 条件すべてで prop = joint(one,1,1) だった（A-3）．台本に assert（R["prop"] == specs["joint"]）を入れ，報告にその旨を書けば答えになる | なし |
| 感情ごとの差 | ρ = 1 以外，教師 sel でもそろえる操作の向きは同じか | 9.3 節 R（決め方の感度），review-ml の感度の指摘 | (a) | joint(v, ρ, a = 1) と kd(v, T = 1, ρ) は格子の全部（v ∈ {one, sel}，ρ ∈ {0.0625, 0.25, 1, 4}）で保存されている（exp/dai1.py:467-468）．同じ組を 8 通りに広げられる | 計算は 8 倍で約 18 分（CPU） |
| 感情ごとの差 | 温度 T = 2，4 の蒸留とそろえる操作の関係 | review-ml | (c) | joint には温度が無く，そろえる操作だけを変えた対は T = 1 だけである．範囲の外と書けば足りる | — |
| 感情ごとの差 | 区間の作りと，補正の有無．se と自由度を出せ | S 項（多重比較），C 項 | (b) の軽いもの | balance_recall.csv に se，df，interval（jk+seed か boot_cl+seed）を残す．今は effect，lo，hi だけで，_fmt が p と t を落とし，se・df も書き出していない．576 本の補正していない区間であることを結果の文に書く | 列 3 つ．費用なし |
| 感情ごとの差 | MOSEI の区間が recall.csv・ablation.csv と違う | 9.3 節の I（同じ量の再現） | (a) | 指摘 [R-乱数] | なし |
| 感情ごとの差 | 対話を塊にした区間（MELD），塊の取り方の感度 | S 項 | (a) | Ctx.cell(spec, "dialogue") で同じ部品から出せる | CPU 1 分 |
| 感情ごとの差 | 性別ごとに出せ | review-affect | (a)（AFEW・IEMOCAP）／(c)（MELD・MOSEI） | AFEW は metadata の gender，IEMOCAP は話者名から．MELD・MOSEI の metadata に性別の列は無い | CPU 数分 |
| 感情ごとの差 | IEMOCAP の台本を含めたらどうなるか | review-affect | (c) | 実験 1 は IEMOCAP を即興だけで組んだ（DAI/CLAUDE.md の決定）．範囲の外 | — |
| 答えの割合 | そろえる操作の代わりに，テストの時の切片の補正（事前分布の補正）で同じことが起きるか | review-ml（簡単な代案） | (a) | kd の W・b は p3.npz にある．テストの特徴の標準化を作り直せば，切片だけを動かした予測を CPU で作れる | CPU 数分．標準化の照合が要る |

(b) の行のうち，計算の後に求められて学び直しになるものは無い．どれも単位の保存（ba98974）から CPU で出せる．

## 指摘

### [I-定義] 答えの割合の定義が要点 17 の出所と違い，同じ条件の値が最大 0.016 ずれる
- 重さ：直してから進む
- 該当箇所：tools/dai1_balance_shares.py:74-79
- 問題：台本は `ans.append(np.bincount(p, minlength=4) / len(p))` を単位ごとに作って `np.mean(ans, 0)` で平均する（単位ごとの割合の平均）．要点 17 の出所の emotion_margin_confusion.csv.gz は単位あたりの数の平均 count_mean を足してから全体で割る（全体でまとめた割合）．テストの件数が分割ごとに違うと 2 つは一致しない．流して比べた差の最大は AFEW 2e-16，MOSEI 1.9e-6，IEMOCAP 0.0097，MELD 0.0156 で，要点 17 は割合を小数 3 桁で書いている．balance_shares.csv の none・joint の行を要点 17 の値と並べると，同じ分類器の同じ割合が 2 通りの数になる．
- 計算の後で直せるか：直せる（CPU 数秒）が，照合の段で食い違いとして返されるので，計算の前に直すのが安い．
- 求める対応：答えの数を単位ごとに足し（または単位あたりの平均を足し）全体で割る形に替え，true_share も同じ形にする．照合に「none と joint の答えの割合が emotion_margin_confusion.csv.gz の none_test・retrained_test（main_q）と 1e-9 で合う」を足す．単位で平均する形を残すなら，両方の列を出し，結果の文でどちらを使うかを書く．

### [R-乱数] MOSEI の区間が保存済みの recall.csv・ablation.csv と違う乱数の列になり，同じ量に 2 つの区間が並ぶ
- 重さ：直してから進む
- 該当箇所：tools/dai1_balance_shares.py:82-86
- 問題：`rng = ctx.rng("ablation")` は (T ごとの) 表 "ablation" の種で始まるが，t_ablation は行 a〜h の順に，t_recall は表 "recall" の列で引く．この台本は (分類器，K，組，感情) の順に引くので，重みの引き直しを使う MOSEI（g_eff = 783）だけ区間が保存済みと違う．流して比べると，joint − none の感情ごとの区間の端は recall.csv の family1 と最大 0.143 点，joint − kd の UAR の区間の端は ablation.csv の行 d と最大 0.0030 点違った．AFEW・IEMOCAP・MELD（jackknife）は 0 で一致した．台本の説明「乱数の列は表の名前 "ablation" の番号」は ablation.csv を再現するように読める．
- 計算の後で直せるか：直せる．
- 求める対応：(1) joint − none の感情ごとの行は recall.csv の family1 と同じ量なので，recall.csv の値をそのまま写すか，照合で effect と（MOSEI 以外の）lo・hi が合うことを確かめて，MOSEI は recall.csv の区間を使う．(2) joint − kd の UAR の区間は ablation.csv の行 d の lo・hi を使い，台本の照合に MOSEI 以外の lo・hi の一致（1e-9）を足す．(3) joint − kd の感情ごとの区間は新しい量なので，説明を「MOSEI の区間は保存済みの表と別の乱数の列で引く」に直す（結果の文 emotion_margin と同じ断り方）．

### [I-照合] 照合 1 は作りから必ず成り立つ恒等式で，台本の誤りの多くを見つけない
- 重さ：記録する
- 該当箇所：tools/dai1_balance_shares.py:87-90
- 問題：effect_stats の点の推定 θ は単位の差の平均で，uar_full(emo_of(cell, e)) × 4 は感情 e の再現率である．4 感情の平均が UAR の差に一致するのは，emo_of と scale = 4 が正しければ必ず成り立つ．指定の取り違え（例：kd の代わりに別の行を読む）は照合 2 と照合 3 が見つける．
- 計算の後で直せるか：直せる．
- 求める対応：照合 2 を区間まで広げ（[R-乱数]），上の突き合わせ（recall.csv の family1，emotion_margin_confusion.csv.gz の答えの割合）を照合に足す．`checks/dai1_balance_shares/code_run/crosscheck.py` の 1・2・4 をそのまま写してよい．あわせて `assert R["prop"] == specs["joint"]`（A-3）を入れる．

### [記録] 単位のコミットを書き込むだけで照らしておらず，provenance.json を書かない
- 重さ：記録する
- 該当箇所：tools/dai1_balance_shares.py:109-111
- 問題：`units_commit="ba98974"` は固定の文字列である．流して調べると 200 個の p3.npz の commit はすべて ba98974…，dirty = False だったので今は正しい．DAI の取り決め（DAI/CLAUDE.md の provenance.json）では台本と入力の sha256 を残す．
- 計算の後で直せるか：直せる．
- 求める対応：u.commit と u.frozen_sha が全単位で同じ値かを確かめて書き込む．emotion_margin と同じ形の provenance（台本・dai1_desc・dai1_stats・frozen.json・ablation.csv の sha256）を残す．

### [B-足すもの] 後から求められやすい表を同じ実行で書き出しておくとよい
- 重さ：記録する（どれも単位の保存から CPU で出せるので，学び直しにはならない）
- 該当箇所：tools/dai1_balance_shares.py:73-102
- 問題と求める対応：B の表の (a) と (b) のうち，(1) 3 版の取り違えの表（単位あたりの平均），(2) balance_recall.csv の se・df・interval の列，(3) ρ ∈ {0.0625, 0.25, 1, 4} と教師 sel に広げたそろえる操作の対（joint(v, ρ, 1) − kd(v, 1, ρ)）を同じ台本で出す．(3) は約 18 分の CPU．

## 目録に無い種類の不備
- 「同じ量を別の台本で作り直すとき，乱数の列と割合の集め方（単位で平均するか，数を足してから割るか）を前の表に合わせる」の項目を 9.3 節の I に足す案．照合は点の推定だけでなく区間の端と，前の結果の文が使った出所の値まで照らす．

当てた項目：I（指定の対応，部品の引数と戻り値，乱数の列，照合の意味），R（後から求められる表の (a)／(b)／(c)），S（区間と補正），L（範囲の外：IEMOCAP 台本，温度 T ≠ 1）．
