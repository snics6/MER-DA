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
7. 数値は照合を通した結果の文か結果ファイルからだけ取る（`DAI/paper/CLAUDE.md`）．嘘にならない言い切りに限る：点推定の符号を数えた「12 条件のうち 11 で上回った」は書いてよい．「4 コーパスすべてで有意に」は区間が支えないなら書かない．

## 1. 要旨（7〜9 文，150〜230 語．上限 250 語）

| 文 | 役割 | 型（原文の句） | 出典 |
|---|---|---|---|
| 1 | 分野の問題 | 「1 つのコーパスでは良いが，新しいコーパスでは落ちる」の対比と原因：*"While methods trained and tested within the same dataset have been shown successful, they often fail when applied to unseen datasets."* 原因は recording conditions，elicitation（acted or natural），annotation，language | ADDoG（Gideon 2019） |
| 2 | なぜ難しいか | ラベルが少ない理由：*"labels ... are collected with perceptual evaluations from multiple evaluators ... cognitively intense and expensive"*．ラベル無しは多い：*"the abundance of unlabeled data"* | Parthasarathy & Busso 2020，Abdelwahab & Busso 2018 |
| 3 | 分かっていないこと | *"..., but it remains unclear ..."* を 1 文 | 岡田研 2023〜2026，Kikuchi 2026，Xue 2023 |
| 4 | 本研究の設定 | *"With linear classifiers on frozen pretrained ... encoders, we answer this in two steps."* | Kikuchi 2026 |
| 5〜6 | 解析で見つけたこと | *"Firstly, we find that ..., whereas ..."*．条件は同じ文に（*"when ..."*，*"but ... rather than universal"*） | Ishii 2022，Kikuchi 2026，Wagner 2023 |
| 7 | 提案 | *"Secondly, we propose ..."*．MER の語で名指す（late fusion / decision-level fusion） | Ishii 2022，融合系 |
| 8 | 評価の範囲と結果 | コーパスを名前で書く．数値は 1〜2 組．*"outperforms X in 11 of 12 conditions, by 2.1–3.5 UAR points on AFEW"* | 融合系 8/15，Okada 2019 の「何個のうち何個」 |
| 9 | 示唆 | *"These findings suggest ..."* | 岡田研 |

- 独自の語（切片のずれ，向きのずれ，gain，余地）は要旨で使わないか，括弧で 1 句の説明を付ける．内部の語（評価の回，条件の数え方，下調べ）は使わない．

## 2. 序論

1. 1 段落目：コーパス間で成績が落ちることと原因を，出典つきで 1 文に並べる（ADDoG の 2 段落目の型）．同じ項目を 2 節のコーパスの表の列にする（EmoBox の型）．
2. 2 段落目：ラベルが少ない理由（評定者の知覚評価の費用，正解の主観性）と，ラベル無しの発話とほかのコーパスは手元にあること．このため半教師ありのドメイン適応の問題になり，MER ではほかのモダリティも使える．
3. 3 段落目：各情報源が MER で役に立つと報告されていること（ほかのコーパス：Ryumina 2025，ラベル無し：SSDA，ほかのモダリティ：Albanie 2018，Zhang 2021）と，分かっていないこと（*"it remains unclear ..."*）．
4. 問いを疑問文で立てる（石井の *"Q1) ...? Q2) ...?"*）．図 1 を問いか設定の概観にする（石井，融合系 8/15）．
5. 貢献は番号つきの 3〜4 項．各項に数値を 1 つ（Okada 2019 の *"for 14 out of 15 traits"*）．動詞は find / propose / show / establish を書き分ける（岡田研）．
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

- 小節の見出しは発見の文にする（CMU-MOSEI の *"Priors in Fusion:"*，Hayashi 2025 の答えの文の見出し）．小節の最初の 1〜2 文で答えを書く（Wagner 2023 の "Answer:" の型）．
- 段落の型：「表 X は…を示す」→ 数値つきの観察（差と，差を作った 2 つの値を括弧に：*"28.7% (42.6% vs. 71.3%)"*）→ コーパスやモダリティの性質による解釈（Ryumina 2025）．
- 弱い結果は同じ段落で *"By contrast, ..."* と数値つきで書く（Kikuchi 2026）．区間が 0 を含む条件はここで書く．
- 差は「点」（percentage points）で書き，「%」と書かない．表の太字は「点推定の最大」と説明に書く．
- 感情ごと・モダリティごとの表と，取り違えやすい感情の組を示す（MELD，DialogueRNN，M3ER）．
- 測っていない原因は *"One possible explanation is that ..."* で書く．

## 6. 考察・限界・結論

- 考察は問いへの答えの形（石井の *"Answer to Q1"*）．示唆は *"These findings suggest ..."*．
- 限界は独立の節．太字の短い見出しの段落を並べ，各段落に「分からないこと，理由，次に試すこと」を書く（Kikuchi 2026，Hayashi 2025）．
- 結論は何をして何が分かったかを数値つきで 1 段落．

## 7. 使わないもの

- 比べた範囲を書かない "state-of-the-art"，"the first"，"by a large margin"．検定や区間の無い "significantly"．
- 指標ごとに最良の相手を選んだ差，1 回の計算だけの表，テストのデータでの調整．
- 測っていない原因を "We demonstrate that ... due to ..." と書くこと（Ryumina 2025 の要旨の型）．
- 日本語の下書きでも，`~/.claude/CLAUDE.md` の「AI臭い文章を書かない」と，語の取り決め（`DAI/paper/ja/README.md`，メモの term-cell-not-masu：条件，反復，gain，疑似ラベル，アンカーなど）に従う．

## 8. 書いた後

- 書いた節は `paper-numbers`（数値）と `paper-citations`（引用）に照らさせ，英語にした後は `paper-prose` に照らさせる（`DAI/.claude/agents/`）．`paper-prose` はこの手順書を基準に読む．
