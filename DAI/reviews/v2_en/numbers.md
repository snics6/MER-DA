# 英語の原稿 ver2 の数値の点検（2026-11-17）

- 対象：`paper/main.tex`（要旨）と `paper/sections/v2_01_introduction.tex`〜`v2_09_conclusion.tex`．行の番号は点検した時点の作業木の行である．組んだ PDF（`main_en_v2.pdf`，36 頁，作業木の tex と同じ時刻に組んだもの）で図の中の数も見た．
- 照らした元：日本語の下書き ver2（`paper/ja_v2/` の 00〜09．コミット a4af2e4 の後の作業木）と，その注に書かれた結果ファイル．結果ファイルは `.venv/bin/python` で読んだ（学習も計算のやり直しもしていない．`results/dai23/asr_check.csv` は開かず，`asr_check_summary.csv` だけを読んだ）．
- 照合の状態：計算の後の比べ「K 本で最も強い 1 つ」の結果の文（第 2.1 版）は `checks/dai20_select_strongest/claim_v2.md` で「判定：通す」．ほかの値は日本語版の点検 `reviews/v2_ja/numbers.md` が第 2 版（照合済み）と照らしている．4.2.3 節の K = 25 の d_T/D だけは，照合を通った結果の文に無い値である（指摘 11）．

## 重さの区分

- 重大：読み手が結果を誤って読む．値の範囲や対象が文と合わない．
- 中：必ず直す．比べる相手の名前の誤り，K や除外の書き落とし．
- 軽：直すとよい．範囲の言い足し，区間の作り方の語．

## 指摘

| 番号 | 重さ | ファイルと行 | 原稿の文（抜き書き） | 原稿の値 | 出所の値 | 直し方 |
|---|---|---|---|---|---|---|
| 1 | 重大 | main.tex:26（要旨），v2_01_introduction.tex:26（貢献 2），v2_09_conclusion.tex:4 | 要旨 "but across 360 post hoc pseudo-label variants per K, UAR rose (95% interval above zero) only when the pseudo-labels were more accurate than the classifier before retraining"．序論と結論 "Across 360 post hoc pseudo-label variants per K (K = 5, 25), ..." | 360 per K．除外の記述なし．要旨は K の値もなし | 日本語版の要旨（00:9），序論（01:33），結論（09:3）は「K = 5，25 の 360 ずつの組み合わせ（学び直す分類器自身の予測だけの疑似ラベルなどを除く）」と書く．除いた点のうち MOSEI の AVT の K = 25（`results/dai1/stats/teacher_margin.csv` の excluded = self_teacher）は当たる幅が作りの上で 0 で，gain が +0.30 [+0.07, +0.54]．除外を書かないと「当たる場合だけ」が成り立たない．これは日本語版の点検の指摘 3（重大）で直した箇所で，英訳で抜けた．要旨は K = 5，25 も書いていないので，K = 5，10，25 のどれでも 360 と読める | 要旨：「Across 360 post hoc pseudo-label variants for each of K = 5 and 25 (excluding pseudo-labels that are only the retrained classifier's own predictions or duplicate other sources), UAR rose ... only when ...」．序論と結論にも括弧の除外を足す |
| 2 | 中 | main.tex:26（要旨） | "gave the audio classifier higher UAR than the labels-only baseline in all 12 conditions" | labels-only baseline | 比べた相手は K 本の基準の分類器（3 つのソースの分類器の平均をアンカーにし，K 本で学び直した分類器．3.4 節）で，ソースの情報を含む．日本語版の要旨は「学び直す前より」．4.1.2 節には別の分類器「target-only classifier (K labels only)」があり，"labels-only" はこれと読める | 「than the same classifier before retraining (the K-label baseline)」か「than the baseline retrained on the few labels without pseudo-labels」にする |
| 3 | 中 | v2_07_discussion.tex:5，:14 | 7.1 "RQ1(b) showed that ... UAR rose only with pseudo-labels more accurate than the retrained classifier"．7.2 "we found, on four corpora with a few target labels, that UAR rose only when the pseudo-labels were more accurate" | — | 日本語版 07:5 は「疑似ラベルの出所を替えた組み合わせを計算の後に並べると」と限っている．英語版の 7.1 はこの限定を落とした．どちらも K = 5，25 と除外を書いていない（日本語版の点検の指摘 3 は 07:5・11 も対象にしていた） | 7.1 と 7.2 に「in post hoc variants at K = 5 and 25 (Sect. 5.1.3)」を足す |
| 4 | 軽 | v2_05_rq1b.tex:4（5 節の冒頭の要約） | "Across 360 post hoc pseudo-label variants per K, the UAR of any classifier rose only when ..." | per K | 本文 48 行に K = 5，25 と除外がある．日本語版 05:5 は「K = 5，25 の 360 ずつ」 | 「at K = 5 and 25」を足す |
| 5 | 軽 | main.tex:26（要旨） | "On AFEW, IEMOCAP, MOSEI and MELD it outperformed the product" | IEMOCAP | LOCO 設定の IEMOCAP は即興の部だけ．日本語版の要旨は「IEMOCAP（即興の部）」，英語版の結論（v2_09:4）は "IEMOCAP (improvised)" | 要旨も "IEMOCAP (improvised)" にする |
| 6 | 軽 | v2_05_rq1b.tex:57 | "Text gained +5.01 to +7.34 points on IEMOCAP with intervals above zero" | +5.01〜+7.34 | 範囲は K = 5，10，25 の値（表 5）．直前の文は K = 0 の話で，K = 0 の言葉の IEMOCAP は +7.52 [+4.75, +10.29] なので，K = 0 を含めると範囲の上端が合わない．日本語版 05:71 は「3 つの K とも」と書く | 「at K = 5, 10 and 25」を足す |
| 7 | 軽 | v2_05_rq1b.tex:57 | "(form (b) in all 48 conditions with K = 5, 10, 25) changed audio UAR by −0.79 to +0.87 points over the 16 conditions with K = 0, 5, 10, 25" | 48（K = 5，10，25），16（K = 0〜25） | `results/dai1/stats/desc/levels.csv` の row = partner：K = 5，10，25 の 48 はすべて kd（形 (b)）．声の K = 0 の 4 つも kd．K = 0 では 16 のうち 3（MOSEI の顔，MELD の顔，MELD の AVT）が agree．値は正しいが，2 つの数の K の範囲がそろわない（日本語版の点検の指摘 14 と同じで，日本語版も同じ文） | 「(form (b) in all 16 audio conditions with K = 0, 5, 10, 25)」にそろえる（任意） |
| 8 | 軽 | v2_06_rq2.tex:93 | "Against strength-only weights the differences were −1.16 to +0.73 points, with intervals above zero on MOSEI at all K (+0.51 to +0.73)" | MOSEI の 3 つだけと読める | `results/dai20_eval/summary/effects.csv` の a = prop，b = M_F2_K：IEMOCAP の K = 5 も +0.24 [+0.03, +0.45] で下端が 0 を超える（表 8 にも * がある）．範囲 −1.16〜+0.73 は合う | 「on MOSEI at all K (+0.51 to +0.73) and IEMOCAP at K = 5」にする |
| 9 | 軽 | v2_04_rq1a.tex:28 | "percentile intervals of a two-stage bootstrap ..., combined with speaker-jackknife t intervals for targets with fewer than 20 effective speakers" | combined | 日本語版 04:37 は「和集合」．"combined" は分散を合わせたとも読める | "the union of ... and speaker-jackknife t intervals" にする |
| 10 | 軽 | v2_05_rq1b.tex:24 | "the exponents of each modality in the product ($\{0,0.5,1,2\}^3$)" | 4³ = 64 通りと読める | 日本語版 05:23 は「{0, 0.5, 1, 2}³ の 63 通り」（全部 0 を除く） | 「(63 combinations of $\{0,0.5,1,2\}^3$ excluding all zeros)」にする |
| 11 | 軽 | v2_04_rq1a.tex:62 | "0.23, 0.47, 0.62 and 0.77 for audio ..., 0.11, 0.32, 0.42 and 0.58 for text. ... (1.10 at K = 25 ...)" | K = 25 の 0.77，0.58，1.10 | `paper/figures/data/fig5_bias_and_direction.csv`：A 0.7714，V 1.1044（used_in_fit = False）で値は合う．照合を通った結果の文は K = 1，5，10 だけを書く（日本語版の点検の指摘 12．日本語版は注でスライドの数値の点検と照らしたと書いて残した） | 値は直さない．照合に回すか，「K = 25, not used in the fit」と書くかを決める |
| 12 | 軽 | v2_08_limitations.tex:23 | "an intercept-only fix on the K labels was below the K-label baseline in 18 of 32 conditions" | 18/32 | 出所（`docs/結果/資料_2026-11-10_追加の結果の文.md` の L5，第 2 版 05:31）は点推定の数え（中央値 −1.6 点）．3.5 節の決まりでは区間で下回ったときだけ "fell" と書くので，"was below" は区間でも下回ったと読める | 「was below ... in point estimates in 18 of 32 conditions」にする |
| 13 | 軽 | v2_01_introduction.tex:26 | "and 11 methods that do not use pseudo-labels including other modalities were all lower in point estimates" | — | 日本語版 01:33 は「どれも点推定で蒸留より低かった」．比べる相手が書いていない | "were all lower than distillation in point estimates" にする |
| 14 | 軽 | v2_06_rq2.tex:95 | "On IEMOCAP, MOSEI and MELD, the product was also higher than this comparison in point estimates (+2.04 to +5.02 points)" | +2.04〜+5.02 | `results/dai20_eval/posthoc/select_strongest_effects.csv`（a = prod，b = sel_K@adj）：+2.04〜+5.02 で合う．9 条件のうち区間の下端が 0 を超えたのは 7（IEMOCAP の K = 10 [−0.98, +10.87]，K = 25 [−3.92, +9.97] は 0 を含む） | 任意．「(intervals above zero in 7 of 9)」を足す（日本語版の点検の指摘 17 と同じ） |

## 仮の値の印

- 表 2 の CREMA-D の実効のクラスタ数 90.98 には \PROV が付いている．`measure/data.py` の `load_group('CREMA-D', 'video')` で読んだ 4,899 発話（91 の speaker_group）から $(\sum_g n_g)^2/\sum_g n_g^2$ を計算し直すと 90.98（90.9826）だった．声の 4,900 発話でも 90.98（90.9835）．分割のファイルの fold ごとの集計ではないので，LOCO 設定の分割が speaker_group をそのままクラスタにしていることを確かめれば \PROV を外せる．
- ほかに仮の値で印の無いものは見つからなかった．英語版で日本語版に無い数は v2_05:130 の「0.24 points」だけで，37.80 − 37.56 = 0.24 と一致する（`teacher_margin.csv` の self_uar_pool：MOSEI の K = 25 で顔 37.80，声 37.56）．
- 表 7 の脚注の "Appendix \TBA" は日本語版の【TBA】と同じで，印は付いている．

## 日本語版から外した図と短くした節

- 外した 3 つの図の数値は失われていない．当たる幅の実例の 2 例（42.2 対 37.6，+2.05 [+0.74, +3.35]，54.4 対 71.1，−4.40 [−7.70, −1.09]）は v2_05:119 の本文にある．提案 − 積の棒グラフの値は表 8 にある．D の測り方の例の図は説明のための例で，実測の値を含まない．
- 限界の節（v2_08）と 7.2 節を短くしたことで誤って読める数値は，指摘 3 と 12 のほかには見つからなかった．日本語版 08:29 の「4.2 節の顔の d_T/D と表 4 の顔の分類器の行は，この特徴の値である」（日本語版の点検の指摘 20）は英語版では落ちていて，残った文（v2_08:23）は正しい．
- 表 6，表 8，表 9 の行の名前を短くしたことによる値の取り違えは無い（下の照らした値を参照）．表 8 の題は略した名前を説明している．

## 照らして合った値

- 表 1，表 2：日本語版の表 1，表 2 と全欄一致した．感情ごとの数の和も発話の数と一致した（例：MOSEI 8,847 + 6,790 + 1,830 + 1,452 = 18,919，MELD 6,383 + 2,283 + 999 + 1,602 = 11,267）．
- 表 3：`paper/figures/data/fig5_bias_and_direction.csv` の panel a（WavLM 5.26，4.69，+0.38 [−0.12, +0.88]，HuBERT 4.73，4.18，+0.18 [−0.42, +0.77]，顔 2.47，2.40，−0.39 [−0.78, +0.02]）．平均を引く方法の区間 [3.81, 5.57] も同じファイル．
- 4.2.1 節：`fig4_mixing_gain.csv` の K = 1（顔 6.91 [4.82, 8.68]，AV 13.39 [11.36, 15.43]）と K = all（顔 −2.05，言葉 −4.80．8 種類とも hi < 0）．
- 表 4：日本語版の表 4 と全欄一致した．列の和（38/7/1/35，32/5/0/44，7 + 60 + 79 + 48 + 16 = 210）も合う．2.02/81 と 5.25/210 はどちらも 0.025．
- 表 5：日本語版の表 5 と 64 欄一致した．K = 0，5，25 の 48 欄は `results/dai1/stats/families.csv`（A = prop，B = none）の effect，lo，hi と小数 2 桁で一致した．K = 10 の 16 欄は `docs/結果/計算の後の解析_先生と自分の差とK10_2026-11-16.md` の要点 1・2 と一致した．
- 表 6：日本語版の表 6 と全欄と印（* と †）が一致した．AVT の 2 つの参照行は `results/dai21_eval5/summary/levels.csv` の none_AVT（69.74，72.55，41.46，47.59）と stud_prod（70.77，72.71，41.10，50.13）とも合う．
- 5.2.3 節：`teacher_margin.csv` から計算し直した．除外を除いた 360 ずつ，順位相関 0.801（K = 5），0.825（K = 25），32 組の中の相関はすべて正（中央値 0.922，最小 0.394），当たる幅が負 173・184（上がった 0，下がった 125・144），正 187・176（上がった 58・42，下がった 16・20．MOSEI の言葉 9 + 8 = 17，MELD の顔 4 + 4 = 8，自分のモダリティを含まない出所 14 + 14 = 28）．最も強いモダリティ（AFEW の顔 71.08・74.73，IEMOCAP の声 68.13・71.84，MOSEI の顔 36.43・37.80，MELD の言葉 50.30・51.96）とその 8 点の gain（−4.40，−3.24，−1.34，−1.02，−3.75，−4.21，−3.31，−1.48），当たる幅が負の 6 点も一致した．図 6 の帯の数（48，113，40，114）も 360 から引いた値と一致した．
- 5.2.4 節：`results/dai23/summary/main.csv` の K = 5：DAT 19.30，10.51，17.71，DBT 10.46，3.06，10.50，M1_B 8.84 [2.96, 14.72]，7.46 [6.31, 8.60]，7.21 [4.67, 9.76]，L3_B 8.46 [3.79, 13.12]，2.73 [1.35, 4.12]，6.70 [3.76, 9.65]．`asr_check_summary.csv` の excess：逆再生 0.0051〜0.0120，元の音声 0.7901〜0.7990．
- 6.1.1 節：`results/dai20_eval/summary/levels.csv` の AFEW の ret_V 70.70，71.93，ret_A 53.24，ret_T 45.76，prod 69.23，70.88．図 8 の最も強いモダリティのマスは日本語版の点検と同じ．
- 表 8：A の組の 6 行（prod，avg，M_F2_K，M_F6_K，tcal，wf1000）を `results/dai20_eval/summary/effects.csv`（a = prop，seeds = all）と，B の組と CREMA-D を `results/dai21_eval5/summary/effects.csv` と `levels.csv`（stud_prop − stud_prod，stud_prop − none_AVT／none_AV）と照らし，値，区間，印が一致した．「Strongest single」の 12 行は `select_strongest_levels.csv`（sel_K@adj）と `select_strongest_effects.csv` と一致した．本文の数え（11/12，区間が 0 を含まない 7，−0.09〜+0.83，−0.05〜+0.82，CREMA-D の +0.31〜+0.63，温度の較正で 8/12，+2.70〜+5.52，7 条件，幅 10.61〜12.01，+0.57〜+0.67，3/50 と 69.25，感情ごとの −6.42〜−4.39，−4.32〜−2.74，−0.21，全部を K 本で測った表で 9/12，重みの表を探すで 11/12）は表と結果ファイルから数え直して合った．
- 表 9：`effects.csv` の prop_noprof と prop_levelfree（4 コーパス），`dai21_eval5` の CREMA-D と一致した．6.2.2 節の割合（AFEW 0.94，0.96，1.00 など．MOSEI 0.30，0.36，0.38．ソースで測った強さの一致 0）は `results/dai20_eval/posthoc/strongest_from_k_summary.csv` の agree_rate と agree_ext_rate と一致した．
- 6.3 節の重みの平均（顔 1.17，声 0.93，言葉 0.90）：`summary/weights.csv` の AFEW，K = 5 の w_prop の 4 感情の平均（1.174，0.928，0.898）．
- 要旨，序論，結論の 4.69，5.26，38/81，7/210，1.4〜4.1，0.80，0.83，173，184，11/12，2.1〜3.5 は上の各表の値と一致した．

## 判定

直してから通す．重大は 1（番号 1．英訳で除外の括弧が要旨・序論・結論から抜けた），中は 2（番号 2，3），軽は 11（番号 4〜14）．表の値は日本語版と結果ファイルの両方と全欄一致し，英訳で値・符号・コーパス・K・分類器が変わった箇所は無かった．
