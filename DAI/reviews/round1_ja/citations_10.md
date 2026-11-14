# 引用の照合：10 分析から作った手法とその評価（`DAI/paper/ja/10_手法と評価.md`）

検査役：paper-citations（2026-11-15，第 1 回）．照合した記録：候補 #21〜#30・#40〜#48，R-LEV（`research_notes/先行研究_水準はK本・相性はソース_2026-11-14/report.md` 3.3・3.4・4 節），R-CONF（`research_notes/先行研究_信頼度の表_2026-11-13/report.md` 2・3・4 節），R-IMP（`research_notes/先行研究_改善案_2026-11-13/report.md` 2.1 節），AlbA 3.1 節，AlbB 97 行，`DAI/docs/実験1_設計.md` 6 節（比べる手法の表と名前の約束），`DAI/docs/実験20_設計.md` 4 節（比べる相手の表）．

## 判定：直してから通す（言い過ぎ 2，取り違え 1）

| # | 原稿の文 | 文献 | 記録の場所 | 判定 | 直し方 |
|---|---|---|---|---|---|
| 10-1 | （80 行）積の規則と和の規則は Kittler ほか [@kittler1998] が整理し | kittler1998 | R-LEV 3.4 節 (b)（◎．抄録と 2 節の見出し） | 合う | — |
| 10-2 | （80 行）分類器 × クラスの重みで答えをまとめる形は Kuncheva と Rodríguez [@kuncheva2014] の枠組みに含まれる | kuncheva2014 | R-CONF 2 節の表（◎．重みつき多数決，recall combiner，naive Bayes combiner の入れ子） | 合う | — |
| 10-3 | （80 行）Kaya ほか [@kaya2017] と Kahou ほか [@要追加: Kahou ほか 2015] が AFEW でモデル × 感情の重みを検証データの上の乱数探索で決めた | kaya2017，Kahou 2015 | R-LEV 3.4 節 (a)（Kaya ◎ 3.5 節 "searched over a pool of randomly generated matrices"．Kahou ◎ 道具経由「感情ごとに単体の上で重みを乱数探索」） | 合う | Kahou は格子の探索も行ったが，乱数探索も行っているので可．Kahou の記録は道具経由なので，数値を足すなら PDF で確かめる |
| 10-4 | （80 行）Busso ほか [@busso2004] は感情ごとにモダリティの強さが違うことを報告し，取り違えの表から作った重みの融合（0.86）が積（0.89）を下回ったことも報告した | busso2004 | R-CONF 2 節の表（◎．本文 4 節と表 1〜7） | 合う | 数値は記録と一致する．データは女優 1 人のコーパス内の結果である（R-CONF 4.3 節）．位置づけの文なので書き足さなくてよい |
| 10-5 | （80 行）Marín ほか [@marin2026classwise] は，全ラベルで学んだ感情ごとのモダリティの重みをコーパスの中で使った | marin2026classwise | R-CONF 2 節の表・4.4 節（◎．重みは発話ごとに小さな MLP が出し，全ラベルで学ぶ） | 合う（注意） | 重みは感情ごとかつ発話ごとである．「全ラベルで学んだ MLP が発話ごとに出す感情ごとのモダリティの重みを，コーパスの中で使った」とすると，この論文の表（発話に依らない重み）との違いが正確になる |
| 10-6 | （80 行）データが少ないときに推す数を減らす考えは，temperature scaling が vector scaling と matrix scaling より良かった Guo ほか [@guo2017calibration] の結果にある | guo2017calibration | R-LEV 4.2 節（◎．主張「1 つの数の temperature scaling が vector scaling と matrix scaling より良い．較正のずれは低次元である」，ラベルの量は検証データの数千〜数万本，「同じ点：データが少ないときは推す数を減らす」はこちらの読み） | **言い過ぎ** | Guo ほかは数千本以上の検証データで較正しており，「データが少ないとき」の結果ではない．Guo ほかの理由づけは較正のずれが低次元であること（"network miscalibration is intrinsically low dimensional"）である．直し方：「推す数を減らす考えは，1 つの数の temperature scaling が，クラスごとの数を推す vector scaling や matrix scaling より良かった Guo ほか [@guo2017calibration] の結果にある．著者らは較正のずれが低次元であることをその理由に挙げた．」 |
| 10-7 | （80 行）評価するドメインを含まない計算でハイパーパラメータを選ぶ手順は，ドメイン汎化の leave-one-domain-out の選択 [@要追加: Gulrajani & Lopez-Paz 2021] と同じ形である | Gulrajani & Lopez-Paz 2021 | R-LEV 3.3 節（◎ 道具経由．3.1・5.1 節） | 合う | 掲載先（ICLR 2021）は ICLR の発表の頁で確かめた（追加の書誌の一覧）．同じ論文は LODO の選択が学習ドメインの検証より弱いと報告している（R-LEV 3.3 節）．査読で問われうる点として 11 節の限界に回すかは研究者の判断 |
| 10-8 | （103・104 行）積の規則・和の規則 [@kittler1998] | kittler1998 | 同上 | 合う | — |
| 10-9 | （106 行）確信度で重みづける平均の出典：確信度で重みづける後段の融合 [@要追加: Tellamekala ほか 2023（COLD Fusion）] | COLD Fusion | D20 4 節（「COLD Fusion が基準に使った型．原典は確信度の量を定めていない．最大の確率で測るのはこの設計の選択である」）．読書の記録は無い（候補 (b)） | **取り違え・記録が無い** | COLD Fusion の手法は確信度で重みづける平均ではなく，この型は COLD Fusion が比べる相手に使ったものである（D20）．表の出典の欄は「COLD Fusion [@Tellamekala] が比べる相手に使った型．確信度は各分類器の最大の確率で測る（この論文の選択）」と書く．この記述自体も読書の記録が無いので，引く前に COLD Fusion の比べる相手の節を原文で確かめる |
| 10-10 | （107〜109 行）分類器ごとの重み，recall combiner に近い数の持ち方，recall combiner の仮定の表 [@kuncheva2014] | kuncheva2014 | R-CONF 2 節，D20 4 節（「硬い票の多数決ではない」「naive Bayes combiner ではない」） | 合う | 原稿は「近い」「仮定の表」と書いており，D20 の限定に合う |
| 10-11 | （110 行）temperature scaling [@guo2017calibration] | guo2017calibration | R-LEV 4.2 節 | 合う | 分類器ごとに K 本で合わせる点，等しい重みの対数尤度，1 へ引く罰則は原典との違い（D20 4 節）．表の注に 1 行あるとよい |
| 10-12 | （111 行）重みつき対数線形プール [@heskes1997] | heskes1997 | R-LEV 3.4 節 (b)（◎ 継承） | 合う | — |
| 10-13 | （112 行）モデル × 感情の重みの乱数探索 [@kaya2017] | kaya2017 | R-LEV 3.4 節 (a)，D20 4 節（原典との違い：足す点，検証データの本数，候補の数） | 合う | D20 は「Kaya ほか 2017 の型」と書く．表の出典の欄も「の型」を付けるとよい |
| 10-14 | （113〜119 行）B の各行の出典：[@hinton2015distilling; @albanie2018]，[@shi2024audioguided]，[@zhang2021combining]，[@blum1998]（著者らの変形），[@sindhwani2008] | 6 本 | D1 6 節の「名前の約束」（「Albanie ほか 2018（Zhang ほか 2021，Shi & Gao 2024）の損失の形を合わせ，提案と同じ予測を受け取る版」と書き，原論文と同じ手法とは書かない），AlbA 3.1 節，AlbB 97 行 | **言い過ぎ**（読み手が原論文と同じ手法だと読む） | 共訓練の行だけに「著者らの変形」がある．表の注に 1 文足す：「B の各行は，原論文の損失の形を合わせ，積の予測を受け取るように変えた版であり，原論文と同じ手法ではない．」Zhang ほか 2021 の原論文の自分の予測は学習中に更新される予測で，ここでは固定した予測を使う（D1 6 節）．Shi & Gao の原論文は 3 つのモデルの予測の一致を使い，ここでは積と K 本だけの分類器の 2 つの一致を使う |
| 10-15 | （119 行）共正則化 [@sindhwani2008] | sindhwani2008 | RW:327（「ワークショップの予定表と Crossref の記録で書誌を確認．本文未読」），RW:655（「論文で引く前に本文を確かめる」） | 合う（確認水準が低い） | 本文を読んだ記録が無い．中身の欄は一般的な共正則化の説明なので細部の取り違えは起きにくいが，記録の指示に従い本文を確かめてから引く．共正則化の名前の出典としては bib にある sindhwani2005coreg（Sindhwani, Niyogi, Belkin 2005）を並べるのが標準的である（候補 #26 も 2 本を挙げる） |
| 10-16 | （120 行）DANN，CORAL，MME，自己学習，エントロピー最小化 | 5 本 | 6 節の照合と同じ | 合う | 6 節の 6-3 と同じ断り（変形であること）が 7 節の本文にあれば，ここは「7 節の標準の手法」で足りる |
| 10-17 | （200 行）Tukey の非加法性の 1 自由度 [@要追加: Tukey 1949] | Tukey 1949 | 記録の中に無い | 記録が無い（方法の出典なので書誌だけで可） | 「全体の平均 + 行の差 + 列の差 + λ × 行の差 × 列の差」は Tukey の 1 自由度の模型の形として正しい．書誌は追加の書誌の一覧 |
| 10-18 | （233 行）Gold Loss Correction の前向き補正 [@hendrycks2018glc] | hendrycks2018glc | R-IMP 2.1 節（◎．Ĉ の式と前向き補正 ℓ(Ĉᵀg(x), ỹ)） | 合う | 原稿の「学び直す分類器の確率にその表を掛けてから疑似ラベルとの交差エントロピーを取る」は前向き補正の形に合う．GLC は Ĉ を雑音のあるデータだけで学んだ分類器から推し，ここでは K 本の交差の推定で推す点が違う（R-IMP 2.1 節の「違いの 2 つ目」）．論文で「GLC の前向き補正の形」と書けば十分 |

## bib の項目

kittler1998（TPAMI 20(3):226–239），kuncheva2014（KAIS 38(2):259–275），kaya2017（IVC 65:66–75．題は bib で補われている），busso2004（ICMI 2004，pp. 205–211），guo2017calibration（PMLR 70:1321–1330），heskes1997（NIPS 10），hendrycks2018glc（NeurIPS 31），hinton2015distilling，blum1998（COLT 1998，pp. 92–100），sindhwani2008（ICML 2008，pp. 976–983）は記録の書誌と一致し，食い違いは無い．
- marin2026classwise：arXiv の記録の注に「ABAW（ECCV 2026 のワークショップ）に採録」とある（arXiv API で 2026-11-15 に確認）．掲載版が出ていれば note を更新する．
- shi2024audioguided：題 "Audio-Guided Fusion Techniques for Multimodal Emotion Analysis"，著者 Shi, Gao は arXiv API で一致した．候補の表は MRAC 2024 と書くが，掲載版の書誌は確かめていない．

## 欠けている位置づけ

候補 (b) にある「Explicit Modality Weighting With Unimodal Supervision for Categorical Multimodal Emotion Recognition」（IEEE TAC 2026）は本文も抄録も読めておらず，10.1.8 節の「見つからなかった」の文を査読で問われうる．引く文献を増やす提案ではなく，投稿前に抄録を確かめる 1 点として挙げる．
