# 引用の照合：2 データ，モデル，設定，誤りの分け方，評価と統計（`DAI/paper/ja/02_設定.md`）

検査役：paper-citations（2026-11-15，第 1 回）．照合した記録：候補（`DAI/paper/notes/引用の候補_2026-11-14.md`），Q3（`DAI_paper/参照論文/要点_問い3_同じ設定の研究.md`），RW（`docs/related-work.md`），F-SSDA・F-MER（`research_notes/有名な論文_2026-11-07/`），`data/README.md`（DR），Ryumina 2025 の PDF．`[@要追加]` の書誌は `citations_追加の書誌.md`．

## 判定：直してから通す（言い過ぎ 1，取り違え 1）

| # | 原稿の文 | 文献 | 記録の場所 | 判定 | 直し方 |
|---|---|---|---|---|---|
| 2-1 | （7 行）IEMOCAP，CMU-MOSEI，MELD，AFEW，RAVDESS，CREMA-D | busso2008iemocap，zadeh2018mosei，poria2019meld，dhall2012afew，dhall2013emotiw，livingstone2018ravdess，cao2014cremad | F-MER:65-180 | 合う | データの出典として引くだけなので可．dhall2012afew は記録が ○（本文未読）だが，細部を言わせていない |
| 2-2 | （7 行）台本の場面では 10 人の役者が同じ台詞を読み，言葉の分類器が台詞を覚えるだけで正解しうるので，即興と台本を別の群にした [@li2021acted] | li2021acted | Q3 2 節，RW:1475 | **取り違え** | Li ほか 2021 は MSP-IMPROV の発話の 4 区分（自然な会話，即興，即興の中の目標の文，読み上げ）をドメインとして扱った音声と映像の研究で，IEMOCAP も言葉の分類器も扱っていない．台詞を覚える理由はこの論文の理由であり，Li ほかの主張ではない．直し方：「…正解しうるので，即興と台本を別の群にした．演技と即興の発話を別のドメインとして扱う例は Li ほか [@li2021acted] にある．」 |
| 2-3 | （9 行）ラベルは Neutral，Happy，Sad，Angry の 4 つにそろえ [@ryumina2025] | ryumina2025 | PDF 4.4 節（"four emotions: Neutral, Happy, Sad, Angry, as IEMOCAP lacks certain ..."，MOSEI は単一ラベルの発話と，どの感情も付かない発話を Neutral にする），WR 3 節の 4 | 合う（注意） | 4 感情に絞ることは Ryumina に合う．MOSEI の 0.5 のしきい値と「2 つ以上ある発話は使わない」はこの論文の規則で，Ryumina と同じかは記録で確かめていない．引用は「4 つにそろえ」の直後に置いたままにし，MOSEI の規則に引用が掛からない現在の位置を保つ |
| 2-4 | （9 行）Whisper large-v3 [@radford2023whisper] の出力 | radford2023whisper | 候補 (b)（書誌は Ryumina 2025 の参考文献 [27] で確認．本文未読） | 合う | モデルの出典として引くだけなので可 |
| 2-5 | （51 行）WavLM-large，RoBERTa-base，ResNet-50 の表情モデル [@ryumina2022] | chen2022wavlm，liu2019roberta，ryumina2022 | F-MER:528-554，DR:445-452 | 合う | ryumina2022 は本文未読（書誌のみ）．重みの出典として引くだけなので可．bib の DOI 10.1016/j.neucom.2022.10.013 は Crossref で題・巻・頁と一致した |
| 2-6 | （51 行）HuBERT-large [@要追加: Hsu ほか 2021]，YOLOv8n-face [@要追加: YOLOv8-face] | — | F-MER:513（HuBERT ◎），DR:454-465（顔検出器） | 記録が無い（YOLOv8-face） | HuBERT は記録に書誌がある．YOLOv8n-face は論文ではなく配布物（derronqi/yolov8-face，WIDER FACE で学習）なので，文献としては WIDER FACE を引き，配布元は脚注の URL にする．書誌は追加の書誌の一覧 |
| 2-7 | （90 行）この形は，ソースの重みへの L2 の引き戻し [@daume2007] を多ソースに広げたものである | daume2007 | RW:1467，F-SSDA:81 | 合う（注意） | Daumé III 2007 はこの形（PRIOR）を比べる相手として使った．考案は Chelba & Acero 2004 である（RW:1467）．「Daumé III [@daume2007] が比べた PRIOR（ソースの重みへの L2 の引き戻し）を多ソースに広げたもの」と書くと出典の役割が正確になる |
| 2-8 | （116 行）共分散は Ledoit–Wolf [@要追加: Ledoit & Wolf 2004] で縮小推定 | — | 記録の中に書誌は無い（候補 (b)） | 記録が無い（方法の出典なので書誌だけで可） | 書誌は追加の書誌の一覧 |
| 2-9 | （132 行）テストに出る人物の実効の数が 20 未満のターゲットでは，百分位の区間が狭く出る [@要追加: Cameron, Gelbach & Miller 2008] | — | 記録の中に無い（候補 (c)） | **記録が無い・言い過ぎ** | Cameron ほか 2008 が示したのは，クラスタが少ない（5〜30）とき，クラスタを単位にした通常の漸近の検定が有意を出しすぎること（名目 5% で約 10%）と，クラスタのブートストラップ t がそれを直すことである（抄録，NBER 版の書誌の頁で確認）．「百分位の区間が狭く出る」は抄録の言い方ではない．直し方：「クラスタが少ないと，クラスタを単位にした推測は有意を出しすぎる [@CGM2008]．このため…jackknife の t 区間との和集合で判定する．」 |
| 2-10 | （132 行）Holm の補正 [@要追加: Holm 1979] | — | 記録の中に無い | 記録が無い（方法の出典なので書誌だけで可） | 書誌は追加の書誌の一覧 |

## bib の項目

busso2008iemocap，zadeh2018mosei，poria2019meld，dhall2012afew，dhall2013emotiw，cao2014cremad，livingstone2018ravdess，chen2022wavlm，ryumina2022，radford2023whisper，liu2019roberta，daume2007 の著者・年・題・掲載先は記録（候補の表と F-MER）と食い違わない．ryumina2022 は Crossref で確認した．

## 欠けている位置づけ

なし．
