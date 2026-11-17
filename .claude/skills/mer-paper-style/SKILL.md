---
name: mer-paper-style
description: MER-DA の論文（DAI に出す MER の論文．DAI/paper/）の文章を書く・直すときに必ず先に読む書き方の手順書．要旨，序論，関連研究，設定，結果の節，考察，限界，図表の説明，カバーレター，査読への返事を，日本語の下書きでも英語の原稿でも対象にする．有名な MER の論文（MulT，TFN，MISA，ADDoG，Busso 研など 29 本）と，岡田将吾（JAIST）・石井亮（NTT）の論文 16 本の書き方の傾向と，研究者の方針（何をしたかを前に出す，MER に位置づける，要旨と序論では結果を言い切る）をまとめたもの．
---

# MER の論文の書き方（2026-11-15 研究者の承認）

研究者の言葉（2026-11-15）：「あなたには基本的に彼らのように書いてほしいし，せっかく調べた書き方の傾向を忘れてほしくない」．論文の文章を 1 文でも書く・直す前に，この手順書を読む．
元のノート（原文の引用と出典の全部）：`DAI/paper/notes/書き方_まとめ_2026-11-15.md`，`書き方_有名なMER論文_融合系_2026-11-15.md`，`書き方_有名なMER論文_コーパス間系_2026-11-15.md`，`書き方_岡田とIshii_2026-11-15.md`，`書き方_MERとDAの要旨と序論_2026-11-15.md`，`書き方_Ryumina2025.md`，`書き方_DAIの手本_2026-11-14.md`．迷ったら元のノートの原文を引いて確かめる．

## 0. 研究者の方針（ほかの規則より優先する）

1. **何をしたかを前に出す．**先行研究の説明と主張の限定を増やしすぎない．先行研究は自分の位置を示す分だけ，1 本に 1〜2 文．
2. **MER に位置づける．**軸はコーパス間の MER，少数ラベルの MER，マルチモーダルの MER．DA・蒸留・半教師あり学習の一般の文献は方法の出典として短く引く．
3. **後半の提案手法を重く，前半の解析の道具は軽く．**解析は「何が分かったか」と「それが手法のどの決め事を決めたか」で書く．
4. **要旨と序論では結果を点推定で言い切る．**区間が 0 を含むかどうか，条件の数え方は結果の節で書く．MER の論文で区間まで出すものはほとんど無い（29 本中 0 本）ので，結果の節の区間はこの論文の強みとして書いてよい．
5. **データ → モデル → 実験 → 結果の順に事実を書く．**評価の手順は事実のとおりに書き，「探索」などのラベルや長い断り書きは付けない．範囲の限定は限界の節にまとめる．
6. **長さ：誌の版で 32 頁ほど（約 9,600 語）．**DAI の ML 論文は補足資料も付録もほとんど使わない（24 本中 ESM 1 本，付録 0 本）．
7. **論の組み立ては次節（0.5 節）の岡田・石井のエッセンスに従う．**「言い切りすぎない」は表に出た症状で，本体は 0.5 節の 6 点である（2026-11-15 研究者の指摘「言い切りすぎない，だとちょっと浅い理解」）．
8. 数値は照合を通した結果の文か結果ファイルからだけ取る（`DAI/paper/CLAUDE.md`）．嘘にならない言い切りに限る：点推定の符号を数えた「12 条件のうち 11 で上回った」は書いてよい．「4 コーパスすべてで有意に」は区間が支えないなら書かない．

## 0.5 岡田・石井のエッセンス（論の組み立て．2026-11-15 研究者の承認）

2 人は文ごとに語を弱めるのではなく，**主張の強さを問いの立て方と証拠の種類で決める**．証拠があるところは強く言い切る（Ishii 2022 の *"critically important"*，*"the first time that such a dissonance has been revealed"*，Okada 2019 の *"more accurate than previous models for 14 out of 15 traits"*）．

1. **問題を方法からではなく人や場面の側から立てる．**石井は「円滑な会話のために参加者は相手の意欲を見て振る舞いを変えなければならない」から始め，手法の動機も人についての考えで書く（*"humans are likely to control actual turn-changing on the basis of turn-management willingness"*，*"We believe that ..."*）．Sakuramoto 2026 は実環境の性質から始める．こちらは「感情の表れ方と付け方がコーパスで違う」「ラベルは複数の評定者が付ける」から始め，分類器の成績はその後に置く．
2. **新しさは「知られていることが，この場面でも成り立つか」で立てる．**Hayashi 2025（初対面で効いた性格の情報が友人どうしでも効くか．*"we investigate whether this findings applies to ..."*），Sakuramoto 2026（*"it remains unclear whether they can be directly applied to real-world environments"*），Kikuchi 2026（LLM が予測できるとき声が何を足すか）．既知の結果を土台にし，最後は *"the first step toward understanding why ..."* で結ぶ．こちらの既知の結果：ほかのモダリティの疑似ラベルで声の分類器が学べる（Albanie 2018）．
3. **問いを番号で立て，主張は問いへの答えとして書く．範囲は主張の文の中に入れる．**石井の Q1，Q2-1，Q2-2 → 手法の節 *"To address Q1"* → 考察 *"Answer to Q1"*．Hayashi 2025 は RO-1（効くか）と RO-2（なぜ効くか）を立て，考察の見出しを答えの文にする．範囲の語（*"in dyadic interactions between friends"*，*"in real-world HRI"*，*"on this dataset"*）を主張の中に入れるので，後から断り書きを足さない．
4. **「効くか」の次に「なぜ効くか」を，量を成分に分けて答える．**Hayashi 2025 は効いたことを示した後 *"demystify how BFFs improve"* と問い，評定を 3 成分に分け，どの入力がどの成分を捉えさせたかを示す．Kikuchi 2026 は 2 つの予測が重なる分散を測ってから「補い合う」と言う．こちらの「誤りを答えの偏りと残りの誤りに分け，どの情報源がどちらを直すか」はこの型に乗せて書く．
5. **動詞の強さを証拠の種類に合わせる．**測った結果は *"Our results show"*（自分の観察は過去形），仮説の検定は *"substantiates / supports / partially confirms"*，解釈は *"suggesting that"*，測っていない理由は *"Although we cannot say for certain ..., one possible explanation is ..."*，先行研究とは *"in line with"*，他人の結果は *"have been reported / shown"*．原因の候補は can / may で並べ，because で断定しない（ADDoG）．「多くの場合」「ふつう」のように測っていない頻度を言わない．設定は「本研究は〜の場合を考える」と書く．
6. **読者が持ち帰る判断を書く．**Hayashi 2025 *"RO-1 provides readers with information to judge whether performance gain is worth the cost of collecting Big Five personality traits"*，Okada 2019 *"we present guidelines for applying the framework"*．こちらは「新しいコーパスで手間をラベル付けとラベル無しの録画のどちらに回すか」の判断材料．

**文のつなぎ**：各文の頭で前の文の最後の語を受ける（成績が落ちる → *"Adapting ..."* → 費用 → *"To reduce this cost"* → 使われてきた → *"However, it remains unclear whether"* → 問い → *"To answer these questions"*）．既知の結果の文を前の文とつなげずに置かない．

## 1. 要旨（7〜9 文，150〜230 語．上限 250 語）

決まった要旨（第 7 版，2026-11-15．`DAI/paper/ja/00_要旨.md`）の流れ：

| 文 | 役割 | 型 | 出典 |
|---|---|---|---|
| 1 | 場面の事実と既知の問題 | コーパスは録音・誘発・注釈が違い，*"... have been reported to degrade on another"*（原因は言い切らず並べる） | ADDoG，エッセンス 1・5 |
| 2 | なぜ難しいか | *"Adapting ... requires its labels, which are costly because each needs several annotators"* | Busso 研 |
| 3 | 既知の対処 | *"To reduce this cost, pseudo-labels from other modalities have been used to ..."* | Albanie 2018，エッセンス 2 |
| 4 | 問い | *"However, it remains unclear whether and why this works when ..., and how ..."* | Sakuramoto 2026，エッセンス 2・3 |
| 5 | 成分に分ける | *"To answer these questions, we decompose the error of ... into ..."* | Hayashi 2025，エッセンス 4 |
| 6〜7 | 答え（観察） | *"Our results show that ... was removed ..., whereas ... decreased ... but rarely ..."*，過去形 | Kikuchi 2026，エッセンス 5 |
| 8 | 提案 | *"Based on this, we propose ... late fusion for pseudo-labelling ..."* | Ishii 2022，融合系 |
| 9 | 範囲と結果 | コーパス名，*"outperformed ... in 11 of 12 conditions, by 2.1–3.5 UAR points on AFEW"* | 融合系，Okada 2019 |

- 7〜10 文，150〜249 語（上限 250 語未満）．数値は 2〜3 組．
- **要旨は 9 の提案の結果で終える．**「これらの結果は〜を示唆する」「〜の判断に役立つ」のような読者への示唆の文を最後に置かない（2026-11-17 研究者：「これが結論ではなくないか？…新たな擬似ラベル作る方法を提案した，とか，あるいは分析の結果になるべき」）．示唆は考察の節に書く．
- 独自の語（切片のずれ，向きのずれ，gain，余地）は要旨で使わないか，括弧で 1 句の説明を付ける．内部の語（評価の回，条件の数え方，下調べ）は使わない．

## 2. 序論

全体の構成（問い Q1〜Q3，章立て，図 1〜9）は `DAI/paper/notes/構成の案_岡田石井の型_2026-11-15.md`（2026-11-15 研究者の承認）．図の決まりは `DAI/paper/notes/図_岡田石井と有名なMER論文_2026-11-15.md` の 4.0 節．

1. 1 段落目：コーパス間で成績が落ちることと原因を，出典つきで 1 文に並べる（ADDoG の 2 段落目の型）．同じ項目を 2 節のコーパスの表の列にする（EmoBox の型）．
2. 2 段落目：ラベルが少ない理由（評定者の知覚評価の費用，正解の主観性）と，ラベル無しの発話とほかのコーパスは手元にあること．このため半教師ありのドメイン適応の問題になり，MER ではほかのモダリティも使える．
3. 3 段落目：各情報源が MER で役に立つと報告されていること（ほかのコーパス：Ryumina 2025，ラベル無し：SSDA，ほかのモダリティ：Albanie 2018，Zhang 2021）と，分かっていないこと（*"it remains unclear ..."*）．
4. 問いを疑問文で立てる（石井の *"Q1) ...? Q2) ...?"*）．図 1 を問いか設定の概観にする（石井，融合系 8/15）．
5. 問いの箇条を置くときは，貢献の箇条は置かない（Ishii 2022．2026-11-17 の ver3 の設計図で研究者が承認）．代わりに "First, we analyse ... to address RQ1 ... Second, we propose ... to address RQ2. This method is motivated by ..." の段落に主な数値を書き，範囲つきの新しさの文は関連研究の最後の小節と結論に置く．貢献の箇条を置く場合は番号つきの 3〜4 項で，各項に数値を 1 つ（Okada 2019 の *"for 14 out of 15 traits"*）．
6. 扱わない範囲は目的のすぐ後に 1 文（Kikuchi 2026 の *"without claiming that ..."*）．

## 3. 関連研究

- 短く，MER の 3 つの流れ（コーパス間・少数ラベル，マルチモーダルの融合，ほかのモダリティからの学習）に分ける．
- 提案は late fusion として位置づける．TFN の late fusion への批判（*"assuming that simple weighted averaging is a proper fusion approach"*）を引き，ラベルが数本の設定で決定の段の融合を選ぶ理由を解析の結果から書く．近い先行研究は M3ER（発話ごとに信頼できる手がかりを強める），Self-MM（単一モダリティのラベルを自動で作る），Kaya 2017，Busso 2004．

## 4. 設定と手法

- 凍結した特徴の上の線形分類器は「解析のための設定」と書く：*"serves as a diagnostic ... rather than as a new ... architecture"*（Kikuchi 2026）．
- 話者の分け方は *"speaker-independent"* と書き，MOSEI はチャンネルか動画で分けることを同じ文で書く．
- UAR は初出で定義し，4 感情でのでたらめの水準 25% を添える．
- ハイパーパラメータを選んだデータを書く（仮のターゲットで選んで固定）．

## 5. 結果の節

- 結果の小節の見出しは比べの名にする（例：「(C1) − (C0)：蒸留の gain」）．答えの文は考察の見出しに置く（2026-11-17 に岡田・石井の雑誌 4 本を読み直して直した．`DAI/paper/notes/岡田石井の方法_言語化_2026-11-17.md` の 10〜12 節）．比べる条件には群ごとに続いた番号を振って条件の表にまとめ，結果の前に比べ方の小節で各比べが何を確かめるかを書く（Ishii 2022 の (1)〜(14)）．結果の小節の最後に「問い X への答えは次のとおりである」のまとめの段落を置く．
- 段落の型：「表 X は…を示す」→ 数値つきの観察（差と，差を作った 2 つの値を括弧に：*"28.7% (42.6% vs. 71.3%)"*）→ コーパスやモダリティの性質による解釈（Ryumina 2025）．
- 弱い結果は同じ段落で *"By contrast, ..."* と数値つきで書く（Kikuchi 2026）．区間が 0 を含む条件はここで書く．
- 差は「点」（percentage points）で書き，「%」と書かない．表の太字は「点推定の最大」と説明に書く．
- 感情ごと・モダリティごとの表と，取り違えやすい感情の組を示す（MELD，DialogueRNN，M3ER）．
- 測っていない原因は *"One possible explanation is that ..."* で書く．

## 6. 考察・限界・結論

- 考察は問いへの答えの形（石井の *"Answer to Q1"*．見出しは答えの文）．各小節は観察 → 意味 → 先行研究との一致と違いの順で，比べごとの解釈は繰り返さない．示唆は *"These findings suggest ..."*．手法の節は Motivation から書き出し，解析のどの値が設計のどこを決めたかを 1 対 1 で書く．
- 限界は独立の節．太字の短い見出しの段落を並べ，各段落に「分からないこと，理由，次に試すこと」を書く（Kikuchi 2026，Hayashi 2025）．
- 結論は何をして何が分かったかを数値つきで 1 段落．最後の文は提案とその結果（か解析の結果）にし，示唆の文で終えない（2026-11-17 研究者の指示）．

## 7. 使わないもの

- 比べた範囲を書かない "state-of-the-art"，"the first"，"by a large margin"．検定や区間の無い "significantly"．
- 指標ごとに最良の相手を選んだ差，1 回の計算だけの表，テストのデータでの調整．
- 測っていない原因を "We demonstrate that ... due to ..." と書くこと（Ryumina 2025 の要旨の型）．
- 日本語の下書きでも，`~/.claude/CLAUDE.md` の「AI臭い文章を書かない」と，語の取り決め（`DAI/paper/ja/README.md`，メモの term-cell-not-masu：条件，反復，gain，疑似ラベル，アンカーなど）に従う．

## 8. 書いた後

- 書いた節は `paper-numbers`（数値）と `paper-citations`（引用）に照らさせ，英語にした後は `paper-prose` に照らさせる（`DAI/.claude/agents/`）．`paper-prose` はこの手順書を基準に読む．
