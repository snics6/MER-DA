# 編集者の事前検査：英語版 ver2（2026-11-17）

対象：`paper/main.tex` と `paper/sections/v2_01_introduction.tex`〜`v2_09_conclusion.tex`，`main.tex` が使う図 12 枚（すべて `figures/*_en.pdf`），`paper/references.bib`．組んだ PDF（`main_en_v2.pdf`，36 頁）．同じ原稿を scratchpad に写して pdflatex と bibtex で組み直し，36 頁になることと，未定義の参照と横にはみ出す箱（Overfull \hbox）が無いことを確かめた．

**判定：直してから回す**
理由：数値は本文・表・結果ファイルで互いに合っており，研究の設計に査読の前に返される欠陥は見つからない．
ただし，下書きの印が 2 か所残り（表 2 の `\PROV{90.98}` と注記，表 7 の脚注の「Appendix TBA」），組んだ PDF では図の文字が 5.1〜6.0 pt になり，カバーレターと論文の種類がまだ無い．
研究者がまだ書いていない欄（資金ほか）を埋めた上で，下の A の項を直せば査読に回せる．

---

## 0. 公式の投稿規定と `venue.md` の照合（2026-11-17）

投稿規定のページ（https://link.springer.com/journal/44163/submission-guidelines ）を取り直し，`docs/参照/DAI_paper/docs/venue.md` と照らした．食い違いは無い．補う点が 3 つある．

1. **カバーレターは全種類の論文で必須**と明記されている（"required for all article types"）．`venue.md` の 1 節にも「投稿に要るもの」として書いてあるが，原稿の側ではまだ用意されていない（A-5）．
2. 倫理の申告が要る対象に，調査（surveys），オンラインの投票（online polls），スマートシティのデータ，健康のデータが書き足されている．この論文に関わる点は変わらない．
3. 誌のページには生成 AI（LLM）の使用についての規定が無い．Springer Nature 全体の編集方針（https://www.springer.com/us/editorial-policies/artificial-intelligence--ai-/25428500 ）は，LLM の使用を Methods の節（Methods の節が無ければ別の適切な場所）に書くよう求めている．文法と読みやすさのための校正（AI-assisted copy editing）だけなら申告は要らない（B-14）．

Methodology の要件は，今のページでも "demonstrable advance on what is currently available" である．`venue.md` にある「十分に試されていること」に当たる文は，取得した要約の中では確かめられなかった．

---

## 未記入の欄（研究者が書く．一覧だけ）

| 欄 | 場所 | 今の文 |
|---|---|---|
| 著者名・所属・連絡先 | `main.tex` の `\author`，`\affil` | TODO |
| 謝辞 | `\bmhead{Acknowledgements}` | TODO |
| 資金（必須．無ければ無いと書く） | Declarations | TODO |
| 利益相反 | Declarations | "no competing interests. TODO (confirm)" |
| 倫理（各コーパスの提供者の承認と同意） | Declarations | TODO を含む |
| データの置き場（発話一覧・ラベルの対応・分割・AFEW の書き起こし） | Data availability | "will be released at TODO" |
| コードの置き場と版の DOI | Code availability | TODO |
| 著者の貢献（CRediT） | Declarations | TODO |

資金とデータの入手先は，空欄のままだと技術的な検査で返される．

---

## A. 差し戻しの理由になるもの

### A-1 下書きの印と内部の注記が表に残っている（重さ：高）

- **場所 1**：表 2（`v2_03_setup.tex`，`tab:loco`）の表題 "the CREMA-D value has not been recomputed from the split file" と，CREMA-D の Effective の欄の `\PROV{90.98}`（PDF ではオレンジ色の文字になる）．
- **場所 2**：表 7（`v2_06_rq2.tex`，`tab:comparators`）の脚注 "values in Appendix \TBA"．PDF では赤い "TBA" が出る．原稿には付録（Appendix）が無い．
- **問題**：作業中の印と，確かめていないことを示す注記が投稿の版に残っている．参照先の付録も存在しないので，読者は「ほかの比べの値」を確かめられない．
- **直し方**：CREMA-D の有効クラスタ数を分割のファイルから計算し直して置き換え，表題の注記を消す．表 7 の脚注で挙げた 4 つの比べ（他コーパスで選んだ指数の積，確信度で重みづけた平均，表による確率の読み替え，K 本で合わせたモダリティの重み）は，付録の表を作って入れるか，脚注を消す．付録を作る場合は，数値を `results/dai20_eval/summary/effects.csv` の行（`M_F6_ext`，`conf`，`cf_D_l05`，`M_F8` 対 `prop`）から取る．

### A-2 組んだ PDF の中で，図の文字が 8 pt を下回る（重さ：高）

- **場所**：12 枚の図すべて．
- **問題**：図のファイルは幅 493 pt（174 mm）で作られており，ファイルの中の文字は Arial の 8〜12 pt で規定に合う．しかし本文の幅は 372 pt（131 mm）なので，`width=\textwidth` で入れると 0.754 倍に縮む．組んだ PDF から文字の大きさを取り出すと，図の Arial は 6.0 pt が 282 字，最小は 5.1 pt である．
  - 図 2（3 つの設定）は `0.94\textwidth` で入れているので，最小は約 5.7 pt になる．
  - 図 7（逆再生）は `0.85\textwidth` で入れているので，最小は約 5.1 pt になる．
  - 規定（Step 6）は図の文字を 8〜12 pt にするよう求めており，図の不備は査読の前に返される主な理由だと書いている．
- **直し方**：図を本文の幅（131 mm，372 pt）で作り直し，その大きさで文字を 8 pt 以上にする．`figsize` の幅を 131 mm にして，図 2 と図 7 も `width=\textwidth` で入れれば，縮めずに入る．あわせて，規定に従って図のファイル名を `Fig1.pdf` の形にそろえる（直した方がよいもの）．

### A-3 要旨の語数が上限の 250 語に近く，数え方によっては超える（重さ：中〜高）

- **場所**：`main.tex` の `\abstract{...}`．
- **問題**：空白で区切って数えると 247 語，"1.4--4.1" の範囲を 3 語として数えると 251 語になる．規定は 250 語「未満」を求める．投稿の仕組み（Snapp）の数え方は分からないので，超えると数えられるおそれがある．
- **直し方**：230 語程度まで削る．削る候補は次の 3 つである．
  - 1 文目の "while labels from several annotators are costly"
  - 第 1 の結果の "with one source corpus" と "with all other corpora as sources" の重複した説明
  - 最後の文
- 削った分で B-1 の略語の定義と，B-2・B-3 の範囲の語を足す．

### A-4 論文の種類が決まっておらず，原稿の中にも書かれていない（重さ：中）

- **問題**：投稿時に Research か Methodology を選ぶ必要がある．原稿は解析（RQ1(a)，RQ1(b)）と手法（RQ2）の両方を含む．
- **Methodology には向かない理由**：Methodology は「今あるものより明らかに進んでいること」を求める．ところが手法には次の弱い点がある．
  - 温度の較正との差は条件によって符号が変わり，AFEW の K = 5 では区間が 0 を下回る（−2.94 [−5.6, −0.3]）．
  - 手法は評価に使ったコーパスの上で作り，CREMA-D は値を見た後に足した（Sect. 8）．
- **直し方**：**Research** で出す．カバーレターと序論の最後の段落に，主な貢献を解析（どの情報源がどの誤りを直すか，蒸留がいつ効くか）と，それに基づく手法の 2 つとして書く．

### A-5 カバーレターが無い（重さ：中）

- **問題**：規定は全種類の論文にカバーレターを求めている．研究の背景と重要さ，この誌に合う理由を短く書く必要がある．誌の目的と範囲は感情認識を名指ししていないので，範囲に合う理由を自分で書く必要がある．
- **直し方**：目的と範囲の項目のうち "machine learning and deep learning"（理論と方法論の項目），"AI in perception and multimedia sensing" と "human-robot/machine interaction"（応用の項目）を挙げる．この研究がラベルの少ない新しいデータへの転移（半教師ありのドメイン適応）を，音声・顔・言葉の 3 つのモダリティで調べるものであることを 3〜4 文で書く．序論の第 1 段落にも，AI の読者向けに 1 文で位置づけを足すとよい（例：新しい場面に感情認識の分類器を移すときにラベルをどう使うかという，知覚の AI に共通の問題）．

---

## B. 直した方がよいもの

### 主張の範囲

**B-1 要旨に，定義していない記号と略語がある**（重さ：中）
- 要旨の "UAR"，"per K"．本文でも UAR は序論の RQ1(b) で先に使い，定義は 3.6 節（Metric and intervals）にある．
- 直し方：要旨で "unweighted average recall (UAR)" と書き，"per K" は "per number of target labels per emotion (K = 5 and 25)" とする．序論で UAR を初めて使う箇所でも定義する．
- 要旨の "labels-only baseline" は本文の "K-label baseline" と語が違うので，本文の語にそろえる．

**B-2 「疑似ラベルの方が正確なときだけ UAR が上がった」の範囲が，要旨と考察で本文より広い**（重さ：中）
- 要旨："UAR rose ... only when the pseudo-labels were more accurate than the classifier before retraining"．考察 7.2："we found, on four corpora ... that UAR rose only when ..."．
- 本文 5.1.3 によると，360 の版から除いた 1 点（MOSEI，AVT，K = 25）は margin が 0 で，区間が 0 を上回った（+0.30 [+0.07, +0.54]）．
- 本文で確かめた事実は「margin が負の版（K = 5 で 173，K = 25 で 184）のうち，上がったものは 1 つも無い」である．これは結論の文の書き方と同じである．
- また，この解析は K = 5 と 25 だけで，計算の後に設けた記述の解析である．要旨は "per K" とだけ書いて K の値を書いていない．
- 直し方：要旨と 7.2 を "none of the variants whose pseudo-labels were less accurate than the classifier before retraining rose (K = 5 and 25; post hoc)" の形にする．

**B-3 要旨の最後の文の勧めが，声の分類器でしか確かめていない結果より広い**（重さ：中）
- 要旨："These results suggest collecting unlabelled multimodal recordings for distillation ..."．
- 表 5 で区間が 0 を上回ったのは，ほぼ声の分類器だけである．顔・言葉・AVT は多くが区間で決まらない．5.2.4 では，各コーパスで最も強いモダリティをほかの 2 つで学び直すと，8 点すべてで点推定が負だった．
- 7.3 の第 1 の根拠（K = 0 の声の分類器が K = 5，10 の基準より高い）には区間が無く，本文の約束（区間が無ければ "higher in point estimates" と書く）にも従っていない．
- 直し方：要旨と 7.3 の勧めを "for the audio classifier" に限る．7.3 の第 1 の根拠に "in point estimates; no intervals were computed" を足す．

**B-4 「上回った（outperformed）」が本文の語の約束より強い**（重さ：中）
- 要旨，6 節の見出し，結論の "outperformed the product in 11 of 12 conditions"．
- 本文 3.6 節は，区間が 0 を外れたときだけ rose/fell と書き，それ以外は "higher in point estimates" と書くと定めている．11 条件のうち区間が 0 を外れたのは 7 条件である（表 8）．
- 直し方："was higher than the product in 11 of 12 conditions in point estimates (95% interval above zero in 7)" の形にそろえる．見出しも同じ語にする．

**B-5 要旨に「手法は評価に使ったコーパスの上で作った」が無い**（重さ：中）
- `DAI/CLAUDE.md` の 2026-11-04 の決定は，要旨と限界の両方にこれを書くとしている．限界（Sect. 8）には書いてあるが，要旨には無い．
- 直し方：要旨の第 3 の結果の後に，"(the method was designed on the same four corpora)" のような 1 句を足す．A-3 で削った語数を使う．

### 報告の正確さ

**B-6 範囲の丸め方が表と食い違って見える**（重さ：低）
- 要旨・序論・結論の "1.4--4.1 points" と，表 5 の "+1.45" と "+4.15"．
- 結果ファイル（`results/dai1/stats/families.csv`）では 1.446 と 4.148 なので，"1.4--4.1" は正しい．しかし表だけを見た読者が丸めると 1.5〜4.2 になる．
- 直し方：本文の範囲を "1.45--4.15" と書き，表と同じ桁にする．

**B-7 「強さだけの重み」の比べと，「プロファイル無し」の除去は，12 条件のうち 8 条件で同じ値になる**（重さ：中）
- 表 8 の "Strength only" と表 9 の "no profile" は，次の 8 条件で数値と区間が同じである．
  - IEMOCAP，MOSEI，CREMA-D の全 K（9 条件のうち，CREMA-D は 12 条件の外の 5 番目のコーパス）
  - AFEW の K = 10，MELD の K = 25
- 違うのは AFEW の K = 5・25 と MELD の K = 5・10 で，選ばれた n0 = 20 の 4 条件と合う．
- 比べの相手の数が実際より多く見えるので，査読者が数え方を問うおそれがある．
- 直し方：表 7 か表 9 の表題に，"Strength-only weights coincide with the proposal without profiles when the selected n0 is 0" を 1 文足す．

**B-8 「強さだけの重みとの差」の区間の記述が不完全**（重さ：低）
- 6.2.1 の "with intervals above zero on MOSEI at all K"．表 8 では IEMOCAP の K = 5（+0.24 [+0.03, +0.5]*）と CREMA-D の 3 つの K も区間が 0 を上回る．
- 直し方："on MOSEI at all K, IEMOCAP at K = 5 and CREMA-D at all K" とする．CREMA-D を 4 コーパスの話から外すなら，そう書く．

**B-9 「最も強い 1 つ」の比べで，切片の移し替えが相手の UAR を下げている**（重さ：中）
- `results/dai20_eval/posthoc/select_strongest_levels.csv` を見ると，AFEW の「最も強い 1 つ」は移し替えの前は 69.25（K = 5）と 73.23（K = 25）で，移し替えの後は 68.65 と 72.79 になる．
- 提案の移し替えの前の値は 71.11（K = 5）である．移し替えの前どうしで比べると差は 1.86 で，表 8 の +2.70 より小さい．
- 本文は K = 5 の 69.25 だけに触れている．
- 直し方：表 8 の脚注か 6.2.1 に，移し替えの前どうしの差（AFEW の 3 つの K）を書くか，移し替えが相手を下げる向きであることを 1 文で書く．

**B-10 「自分の確率を疑似ラベルにした版は −1.0〜+0.5」の K が書かれていない**（重さ：低）
- 5.2.2．表 6（K = 5）から計算すると −0.3〜+0.5 なので，−1.0 はほかの K の値である．
- 直し方："over K = 5, 10 and 25" を足す．

**B-11 損失の形の比べで，条件の数と K の範囲が食い違って見える**（重さ：低）
- 5.2.1 の "form (b) in all 48 conditions with K = 5, 10, 25" と，その直後の "over the 16 conditions with K = 0, 5, 10, 25"．K = 0 でどの形が選ばれたかが書かれていない．
- 直し方：K = 0 で選ばれた形を書くか，16 条件を K = 5，10，25 の 12 条件にそろえる．

### 書式

**B-12 参考文献に DOI が 1 つも無い**（重さ：中）
- `references.bib` に doi の欄が 0 件ある．使っている `sn-nature.bst` は DOI を出す行を注釈で消している（`paper/CLAUDE.md` の未決）．規定は "always include DOIs as full DOI links" と求める．
- 直し方：DOI のある項目に `doi` の欄を足し，DOI を `https://doi.org/...` の形で出す様式を使う．`sn-vancouver-num` は NLM に合う見込みである．使う前に DOI が出ることを確かめる．

**B-13 arXiv だけで出ている文献がある**（重さ：低）
- [30] Marín ほか 2026（arXiv:2609.09924）は，関連研究と位置づけ（2.5 節）で手法の違いを示す相手に使っている．[43] Shi と Gao 2024（arXiv:2409.05007）もある．
- 規定は，参考文献の一覧に発表済みか採録済みの文献だけを載せるとしている．
- 直し方：採録されているか調べ，採録されていれば書誌を直す．未採録なら，arXiv の DOI（`https://doi.org/10.48550/arXiv....`）を付けて載せられるかを誌に確かめる．[39] RoBERTa と [16] Hinton ほか（ワークショップ）は慣例として通るとみる．

**B-14 生成 AI の使用の申告**（重さ：確認）
- 原稿の英語版は，日本語の下書きを AI の助けで英訳して作ったとみられる（`paper/ja_v2/README.md`）．
- Springer Nature の方針では，校正を超える使い方（訳や文の生成）は Methods の節などに書く．
- 直し方：使い方を研究者が確かめ，校正を超えるなら 3 節か Declarations に 1 文書く．

**B-15 略語の初出の定義が抜けている**（重さ：低）
- 次の略語を初出で展開していない．
  - IM（information maximisation）
  - T3A（test-time template adjuster）
  - CORAL（correlation alignment）
  - DANN（domain-adversarial neural network）
  - MME（minimax entropy）
  - AUC（6.1.2）
  - CCC（Sect. 8．concordance correlation coefficient）
  - RQ（序論で "research questions (RQs)" と書いていない）
- 直し方：4.1.3 か 5.1.2 で手法名を初めて出す箇所で展開する．

**B-16 見出しが実質 4 段になる箇所がある**（重さ：低）
- 6.1.2（`\subsubsection`）の中の太字の小見出し "Reliability."，"Strengths from the K labels."，"Profiles from the sources."，"Fusion and pseudo-labels." が 4 段目に見える．規定は 3 段までである．
- 直し方：太字をやめて段落の 1 文目で主題を書くか，6.1.2 を 2 つの `\subsubsection` に分ける．
- 節の見出しが結果の文になっていて長い（4 節は 25 語）．規定違反ではないが，DAI の掲載論文は短い見出しが多い．

**B-17 表 2 が本文で引かれていない**（重さ：低）
- `tab:loco` は表 1 の表題からしか引かれていない．規定は，表を本文で番号順に引くよう求める．
- 直し方：3.3 節の LOCO 設定の説明の中で "(Table 2)" と引く．

**B-18 表の脚注に印が無い**（重さ：低）
- 表 4 の脚注（"Halving δ ..."）と表 7 の脚注に上付きの小文字が無い．規定は，表の脚注を上付きの小文字（有意の印は星）で示すよう求める．
- 直し方：本文の該当箇所に a などの印を付ける．

**B-19 書誌を含む脚注**（重さ：低）
- 3.3 節の YOLOv8-face の脚注は，URL・許諾・学習データの引用を含む．規定は「書誌の詳細を書く脚注」を避けるよう求めている．
- 直し方：ソフトウェアとして参考文献の一覧に入れ，本文で番号を引く．

**B-20 図の中に説明文がある**（重さ：低）
- 次の図は，下端に説明の文を図の中に書いている．
  - 図 7："Values in UAR points. Visual D = ..."
  - 図 8："Reliabilities used to design the method ..."
  - 図 6："1 point = one ..."
- 規定は「図の中に題や説明を書かない」としている．
- 直し方：これらの文を図の説明（caption）に移す．パネルの見出し（(a)，(b)）は残してよい．

**B-21 「Sec.」と「Sect.」が混ざっている**（重さ：低）
- 図 1 の中は "Sec. 4"，本文は "Sect."．直し方：図の側を "Sect." にそろえる．

**B-22 本文の文字の大きさ**（重さ：低）
- 規定は 12 pt 以上と Helvetica か Arial を求めている．雛形のままだと本文は 10 pt の Computer Modern，表は約 7 pt である．
- LaTeX で出すと誌が組み直すので返される見込みは低いが，査読用の PDF は読みにくい．表 8 は 1 頁を占め，`\tabcolsep` が 2 pt である．
- 直し方：雛形の指定を確かめ，査読用 PDF の文字を大きくする選択肢があれば使う．

### データの入手先と倫理

**B-23 AFEW の書き起こしを公開するという文が，公開しない理由と食い違う**（重さ：中）
- Data availability："the Whisper transcripts of AFEW will be released"．一方で "Per-utterance transcripts of reversed speech are not released because they contain licensed corpus content"．
- AFEW の書き起こしは映画の台詞なので，同じ理由で許諾の制約を受けうる．
- 直し方：AFEW の使用許諾（EmotiW の配布の条件）を確かめ，書き起こしを出せないなら，作り方の台本と Whisper の設定だけを公開すると書く．
- あわせて，IEMOCAP と AFEW の入手の手続き（申請先）を 1 文ずつ書く．

**B-24 倫理の文の書き方**（重さ：低）
- 今の文は二次的な解析であることを書いている．TODO の箇所には，各コーパスの原典が記す承認と同意の扱いを 1 文ずつ書く．DAI の掲載例（10.1007/s44163-026-01768-1）は「元の提供者が倫理の承認と同意を得た公開データの二次的な解析」と書いている．
- AFEW は映画の切り出しで，出演者の研究への同意は無い．そのため「配布者の許諾の条件のもとで使った」と書く方が正確である．

### 長さ

**B-25 原稿が長い**（重さ：低）
- 本文の文章は約 10,000 語（表の行を除く．表と図の文字を含めると約 14,800 語），36 頁である．
- 規定に上限は無いが，`venue.md` 7 節の掲載論文 6 本は本文 3,500〜8,000 語である．査読者の負担が大きいと，査読者が見つからずに取り下げになるおそれがある（`venue.md` 4 節の 6）．
- 直し方：4.1.1（誤差の模型の検証の数値），5.1.4（逆再生の作り方の細部），6.1.1（予備調査の相関の範囲）の細部を補足資料（ESM）に移す．

---

## 確かめた数値（一致したもの）

次の数値は本文・要旨・表・結果ファイルの間で一致した．

- **表 8（`tab:rq2`）**：全行で「提案 − 行」の差が，提案の値から行の値を引いた値と 0.01 以内で一致した．* と † が区間と合うことも確かめた．
- **結果ファイルとの照合**：
  - 提案 − 積の 12 条件は `results/dai20_eval/summary/effects.csv` の `prop` 対 `prod` と一致した．
  - 学び直した AVT の差と CREMA-D の値は `results/dai21_eval5/summary/effects.csv` と一致した．
  - 「最も強い 1 つ」の差と幅（AFEW の区間の幅 10.61〜12.01）は `results/dai20_eval/posthoc/select_strongest_effects.csv` と一致した．
  - 表 9 の除去の差は `prop_noprof` と `prop_levelfree` と一致した．
- **文中の数え**：
  - 11/12（積との差），7（区間で上），8/12（温度の較正との差），9/12（12 個の表），11/12（表の探索），+2.70〜+5.52，積 − 最も強い 1 つの +2.04〜+5.02 と +0.57〜+0.67 は，すべて表 8 から再計算して一致した．
  - K = 5 で非視覚の分類器が選ばれたのは 3/50 回で，移し替えの前の値は 69.25 だった．どちらも結果ファイルと一致した．
- **表 5（`tab:gain`）**：K = 0，5，25 の値と区間は `results/dai1/stats/families.csv`（family 1）と一致した．K = 10 は照らしていない．
  - 表 6 の AVT の基準と蒸留は表 8 の K-label base. と Distill. (prod.) と一致し，その差は表 5 の AVT の行と合う．
  - 本文の範囲（+3.32〜+4.15 など），K = 0 の 53.4 対 50.1 などは表と合う．
- **表 4（`tab:allsource`）**：各列の合計は 38/7/1/35，32/5/0/44，194 + 16 = 210 で一致した．偶然の期待値は 81 × 0.025 = 2.02，210 × 0.025 = 5.25 で一致した．
- **表 1 と表 2**：感情ごとの数の合計が発話数と一致した．
- **5.2.3〜5.2.5**：margin の数え（173 + 187，184 + 176，16 + 20 = 36）と，D の差（19.3 − 2.0 = 17.3 など）が合う．
- **照らしていないもの**：
  - 4.2.4 の "341 of 373 pairs"
  - 6.1.1 の予備調査の相関の範囲（−0.954〜+0.403 など）
  - 表 3 の値
  - 表 5 の K = 10 の列
