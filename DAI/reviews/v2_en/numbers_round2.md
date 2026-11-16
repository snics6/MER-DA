# 英語の原稿 ver2 の数値の再点検（第 2 回，2026-11-17）

- 対象：`paper/main.tex`（要旨，付録 A の表 A1）と `paper/sections/v2_01`〜`v2_09`（作業木）．`sections/v2_*.tex` は git に入っていない（`??`）ため `git diff` では差分が出ない．そこで，依頼の 7 点を結果ファイルで照らしたうえで，英語の 9 節に出る数をすべて抜き出し，日本語版 `paper/ja_v2/*.md` に同じ数があるかを機械で照らした．日本語版に無い数は表 3 の列幅（1.75cm）だけだった．前回の指摘 1〜14 が直ったかも確かめた．
- 結果ファイルは `.venv/bin/python` で読んだ．学習も計算のやり直しもしていない．`results/dai23/asr_check.csv` は開いていない．
- 照合の状態：付録 A の 4 つの比べの値は，実験 20 の結果の文（`docs/結果/実験20_結果の文の下書き_2026-11-15.md` 104〜111 行．照合 `checks/DAI-20/claim_v2.md`）に同じ値がある．最も強い 1 つの比べの値は結果の文 第 2.1 版（照合 `checks/dai20_select_strongest/claim_v2.md`，判定 通す）の要点 1・2・5 にある．6.1.1 節の 12 個の範囲は `checks/paper-v2/claim_round5.md` 18 行で結果ファイルと照らされている．照合を通っていない結果の文から取った値は見つからなかった．

## 依頼の 7 点の照合

| 番号 | 原稿の文（抜き書き） | 原稿の値 | 出所の値 | 判定 | 直し方 |
|---|---|---|---|---|---|
| 1a | 表 A1（main.tex の appendices）の 48 欄 | 点推定 48，区間 96 端 | `results/dai20_eval/summary/effects.csv`（a = prop，seeds = all，b = psel，conf，cf_D_l05，temp）．48 欄すべて点推定（小数 2 桁）と区間（小数 1 桁，0 に近い端は 2 桁）が一致した．例：psel の MOSEI の K = 5 は 0.4513 [−0.0242, 0.9267] → +0.45 [−0.02, +0.9]，K = 10 は下端 0.0376 → +0.04．* の付け方（下端 > 0）も 25 欄すべて合う．† は 0 欄で，上端 < 0 の欄も 0 | 合う | — |
| 1b | 6.2.1 節 "the proposal was higher in 39 of 48 conditions in point estimates, with intervals above zero in 25 and below zero in none" | 39，25，0 | 同じファイルで数え直すと，点推定が正 39（psel 12，conf 12，cf_D_l05 8，temp 7），下端 > 0 は 25（11，5，3，6），上端 < 0 は 0 | 合う | — |
| 1c | 付録の 4 つの比べの説明 | — | 設計 `docs/設計/実験20_設計.md` 77〜84 行と照らした．指数（実験 1 の格子から別のコーパスで選んだ指数を平均 1 に割る．AFEW は (1, 1, 1) で積と同じ判定），確信度（発話ごとに最大の確率を重みにした平均．確信度を最大の確率で測るのはこの研究の選択），表の読み替え（対角だけの表，非対角は (1 − 対角)/3，対角の平均の表へ半分縮める，対角の平均 0.25 以下の分類器は使わない），モダリティの重み（感情に共通の 3 つの重みを，まとめた答えの K 本の上の対数尤度で同時に合わせる，Heskes 1997 の型）はどれも設計と合う．AFEW の psel の値が 提案 − 積（表 8）と同じ値になっていることも確かめた | 合う | 任意で次の 2 点を足す．(i) モダリティの重みは「1 へ引く罰則 0.01，範囲 0〜5，合わせた大きさをそのまま使う」（設計 80 行）．(ii) 表で読み替える版で使わなかった分類器の数は 200 単位 × 3 つの K で合わせて 19（結果の文 109 行） |
| 2 | 6.2.1 節 "Against strength-only weights the differences were −1.16 to +0.73 points, with intervals above zero on MOSEI at all K (+0.51 to +0.73 points) and on IEMOCAP at K = 5 (+0.24 points)" | −1.16〜+0.73，MOSEI 3 つ，IEMOCAP の K = 5 の +0.24 | `effects.csv` の b = M_F2_K：最小 −1.156（AFEW，K = 5），最大 +0.729（MOSEI，K = 5）．下端 > 0 は MOSEI の 3 つ（0.452，0.361，0.256）と IEMOCAP の K = 5（+0.242 [+0.032, +0.451]）の 4 つだけ．IEMOCAP の K = 10 の下端は −0.047 | 合う | — |
| 3a | 6.2.1 節 "The intercept shift lowered this comparison's UAR on AFEW by 0.44--0.65 points" | 0.44〜0.65 | `select_strongest_levels.csv` の sel_K − sel_K@adj：AFEW 0.597，0.652，0.442．要点 5 の「0.44〜0.65 点下げた」と合う | 合う | — |
| 3b | 同 "without the shift for either method, proposal − strongest single was +1.86 to +6.11 points in all 12 conditions, with intervals above zero in eight" | +1.86〜+6.11，12，8 | `select_strongest_effects.csv` の a = prop@raw，b = sel_K：最小 +1.857（AFEW，K = 5），最大 +6.109（IEMOCAP，K = 5），12 条件すべて正，下端 > 0 は 8（AFEW の 3 つと IEMOCAP の K = 25 が 0 を含む）．要点 5 と合う | 合う | — |
| 3c | 同 "the product was also higher than this comparison in point estimates (+2.04 to +5.02 points; intervals above zero in 7 of 9)" | +2.04〜+5.02，7/9 | a = prod，b = sel_K@adj の IEMOCAP・MOSEI・MELD：+2.04〜+5.02，下端 > 0 は 7（IEMOCAP の K = 10 [−0.98, +10.87] と K = 25 [−3.92, +9.97] が 0 を含む）．要点 2 と合う | 合う | — |
| 4 | 表 8 の題 "Strength-only weights gave the same values as the proposal without profiles (Table 9) in 8 of the 12 conditions on the four corpora and in all three on CREMA-D" | 8/12，3/3 | `dai20_eval/summary/levels.csv` で M_F2_K − prop_noprof の UAR の差が 0 なのは AFEW の K = 10，IEMOCAP の 3 つ，MOSEI の 3 つ，MELD の K = 25 の 8 条件．ほかの 4 条件の差は 1.54，0.32，0.57，0.34．CREMA-D は `dai21_eval5/summary/levels.csv` で 3 つとも同じ値（85.26，85.89，87.46）．6.1.3 節の「n0 が 0 なのは 12 条件のうち 8」とも数が合う | 合う（下の指摘 A） | 下の指摘 A |
| 5 | 表 2 の CREMA-D の実効のクラスタ数 90.98（\PROV を外した） | 90.98 | `results/dai18_probe/splits_new.csv` の CREMA-D：4,899 発話，cluster の列で 91 クラスタ，(Σn)²/Σn² = 90.9826．感情ごとの数 1,087 / 1,271 / 1,270 / 1,271 も合う．分割のファイルは speaker と同じ値を cluster の列に持つので，前回の残りの点（分割が話者をそのままクラスタにしているか）も確かめられた | 合う | — |
| 6 | 要旨（240 語），序論，結論，7 節の数と条件 | — | 要旨は単語 240（記号の "=" を除くと 239）で 250 未満．1.4〜4.1（声の gain の点推定 1.446〜4.148．`results/dai1/stats/families.csv`），11/12，2.1〜3.5（2.13，3.51，3.14），0.80・0.83，173・184，38/81，7/210，4.69，5.26 は前回と同じく合う．除外の書き方には直す点が残る（下の指摘 B〜E） | 直す | 下の指摘 B〜E |
| 7 | 6.1.1 節 "the classifier-by-emotion component left after fitting classifier and emotion effects and Tukey's one-degree-of-freedom non-additivity term ... correlated 0.463--0.780" | 書き方，0.463〜0.780 | 日本語版 `paper/ja/07_問い2.md` 23 行の「水準などを除いた分類器 × 感情の成分（Tukey の非加法性の 1 自由度の当てはめの残り）」と同じ意味である．計算の中身（`exp/dai16_probe.py` 185〜203 行の `tukey_fit` と `dcenter`）は「全体の平均＋行の差 a＋列の差 b＋λab」を当てはめて残りを取るので，英語の "after fitting classifier and emotion effects and Tukey's ... term" はこの式と合う．値は `results/dai16_probe/summary/r0.csv` と `results/dai16_probe_k10/summary/r0.csv`（fam = M，12 個）から計算し直して，水準の相関 −0.954〜+0.403，一致の割合 0〜0.467（K 本の表 0.433〜1.000），成分の相関 0.463〜0.780 になった | 合う | — |

## 新しく見つけた指摘

| 番号 | 重さ | ファイルと行 | 原稿の文（抜き書き） | 原稿の値 | 出所の値 | 判定 | 直し方 |
|---|---|---|---|---|---|---|---|
| A | 軽 | v2_06_rq2.tex:97（表 8 の題） | "Strength-only weights gave the same values as the proposal without profiles (Table 9)" | same values | UAR は 8 条件で同じだが，区間は比べごとに引き直しているので端が少しずれる．MOSEI の K = 25 は表 8 で +0.51 [+0.3, +0.8]（上端 0.756），表 9 で +0.51 [+0.3, +0.7]（上端 0.750）と，表示の桁でも違う | 直す | "gave the same UAR as ..." にし，「intervals differ slightly because the resampling was drawn separately」を足すか，"same decisions" と書く |
| B | 中 | main.tex:26（要旨） | "distillation ... raised audio unweighted average recall (UAR) over retraining on the K labels alone in all 12 conditions (point estimates; 1.4--4.1 points)" | raised，12 条件 | 3.5 節の決まり（v2_03:69 "We write that UAR rose or fell only when the 95% interval ... lies above or below zero, and write 'higher in point estimates' otherwise"）では，点推定だけの比べに "raised" は使えない．12 条件のうち IEMOCAP の 3 つは区間が 0 を含む（+2.41 [−0.45, +5.27]，+2.01 [−0.57, +4.58]，+1.45 [−0.48, +3.37]）．また日本語版の要旨は「12 条件（K = 5，10，25）」と K を書くが，英語版は K を書いていない．LOCO 設定には K = 0 の条件もあるので，12 条件の中身が読み取れない | 直す | "distillation ... gave higher audio UAR than retraining on the K labels alone in all 12 conditions (K = 5, 10, 25; point estimates, 1.4--4.1 points)" にする |
| C | 軽 | v2_09_conclusion.tex:37，v2_05_rq1b.tex:4 | "Across 360 post hoc pseudo-label variants per K (K = 5, 25; pseudo-labels made only of the retrained classifier's own predictions excluded)" | 360，除外は自身の予測だけ | `results/dai1/stats/teacher_margin.csv` は K ごとに 368 行で，excluded は self_teacher 4 と same_as_poe_none（ほかの出所と同じ疑似ラベル）4．自身の予測だけを除くと 364 で，360 にはならない．序論（v2_01:26）は "and duplicates excluded"，5.1.3 節（v2_05:48）は "and duplicates of other providers" と書いている | 直す | 結論と 5 節の冒頭にも "and duplicates of other providers" を足す（要旨は 360 を書いていないのでこのままでよい） |
| D | 軽 | v2_07_discussion.tex:14（7.2 節） | "we found, in cross-corpus MER on four corpora with a few target labels, that in post hoc variants UAR rose only when ..." | K の記載なし | 当たる幅を測ったのは K = 5，25 だけ（5.1.3 節，限界の節 v2_08:31）．7.1 節（v2_07:5）は "in post hoc variants at K = 5 and 25" に直っている | 直す | "in post hoc variants at K = 5 and 25" にそろえる（前回の指摘 3 の残り） |
| E | 軽 | v2_07_discussion.tex:17（7.3 節） | "the audio classifier retrained on unlabelled utterances and pseudo-labels alone (K = 0) had higher UAR than the K-label baselines with 5 and 10 labels per emotion on all four corpora" | 4 コーパスすべて | `results/dai1/stats/desc/levels.csv`（student = A）の K = 0 の joint\|one\|main 53.37，69.20，36.47，45.17 と，none の K = 5（50.09，67.05，34.86，41.49），K = 10（51.12，68.33，35.13，41.80）で値は合う．ただし差の区間は作っていない（5.2.1 節 "no intervals computed"）．3.5 節の決まりでは "higher in point estimates" と書く | 直す | "had higher UAR in point estimates (no intervals computed) than ..." にする |
| F | 軽 | v2_03_setup.tex:69（3.5 節） | "the variance of a cluster jackknife (cluster resampling for MOSEI) is added ... jackknife t intervals are used for corpora with fewer than 20 effective test clusters" | MOSEI だけがクラスタの引き直し | `results/dai21_eval5/summary/effects.csv` の method は CREMA-D も boot_cl+seed（実効のクラスタ数 90.98 ≥ 20）．AFEW，IEMOCAP，MELD は jk+seed，MOSEI は boot_cl+seed | 直す | "(cluster resampling for MOSEI and CREMA-D)" にする．同じ文の前の "clusters for MOSEI and the LOCO setting" はこのままでよい |
| G | 軽 | v2_06_rq2.tex:94 | "than the visual classifier alone in Sect.~\ref{sec:rq2} (70.70)" | 参照先 | 70.70 は 6.1.1 節（v2_06:9）の AFEW の K = 5 の ret_V．日本語版 06:99 は「6.1.1 節の顔の分類器 1 つ」．英語版は 6 節全体を指していて，自分の節を指す参照になっている．値は合う | 直す | 6.1.1 節の小見出しに \label を付けてそこを参照するか，"(Sect. 6.1.1, 70.70)" にする |
| H | 軽 | main.tex の appendices の説明 | "compares the estimation-time proposal with four further combinations ..." | — | 表 A1 の値は相手にも提案と同じラベル無しの切片のずらしを当てた判定（adj 層）どうしの差である（b = psel 等．b = psel@raw 等ではない）．本文で切片のずらしを書いているのは最も強い 1 つの比べ（v2_06:65）だけで，A の組のほかの相手がずらした答えかは書いていない．ずらさない相手の判定（raw）との差で数えても，点推定が正 41，下端 > 0 は 26，上端 < 0 は 0 で，6.2.1 節の読み（48 のうち 39，25，0）は変わらない | 直す（任意） | 付録の最初の文か表 A1 の題に "Both sides use the label-free intercept shift of Sect. 6.1.2" を足す．表 8 の A の組にも同じことが言える |

## 前回の指摘 1〜14 の状態

| 前回の番号 | 状態 |
|---|---|
| 1（除外の括弧） | 要旨と序論は直った．結論と 5 節の冒頭は自身の予測だけを書き，ほかの出所と同じものの除外が抜けている（指摘 C） |
| 2（labels-only baseline） | 直った（"over retraining on the K labels alone"） |
| 3（7 節の限定） | 7.1 節は直った．7.2 節に K が無い（指摘 D） |
| 4，5，6，7，8，10，12，13，14 | 直った |
| 9（区間の和集合） | 直った（v2_04:28 "the union of this interval and a speaker-jackknife t interval"） |
| 11（K = 25 の d_T/D） | 値は変えていない（0.77，0.58，1.10）．照合に回すかどうかは前回と同じく未決 |
| 仮の値の印（CREMA-D の 90.98） | 外してよいことを確かめた（上の 5） |

## 仮の値と【TBA】

- main.tex で \PROV を使う箇所は無くなった（定義だけ残る）．\TBA を使う箇所も本文と付録には無い．前回の表 7 の脚注の "Appendix \TBA" は付録 A への参照に替わっている．
- 出所の注が無い数値：英語の各節の冒頭の注は日本語版の注を出所として指している．英語の 9 節の数は表 3 の列幅を除いてすべて日本語版に同じ数があった．付録 A の値は日本語版 `paper/ja_v2/10_付録.md` に同じ表があり，その注が `effects.csv` と設計の行を指している．

## 判定

直してから通す．依頼の 7 点の値（表 A1 の 48 欄，39/48・25・0，強さだけの重みの区間，切片のずらしの 0.44〜0.65 と +1.86〜+6.11・8，積の 7/9，表 8 の題の 8/12・3/3，CREMA-D の 90.98，Tukey の書き方と 12 個の範囲）は，すべて結果ファイルと表示の桁で一致した．直す点は文の書き方だけで，中が 1（指摘 B：要旨で点推定だけの比べに "raised" を使い，K を書いていない），軽が 7（指摘 A，C〜H）である．計算のやり直しは要らない．
