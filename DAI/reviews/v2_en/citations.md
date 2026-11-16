# 引用の照合：英語の原稿 ver2（paper/main.tex，paper/sections/v2_01〜v2_09．2026-11-17）

対象：`paper/main.tex`（要旨と申告），`paper/sections/v2_01_introduction.tex`〜`v2_09_conclusion.tex`，`paper/references.bib`．組んだ PDF（`main_en_v2.pdf`）の参考文献の一覧も見た．
記録：`paper/notes/引用の候補_2026-11-14.md`（候補），`docs/related-work.md`（RW），`research_notes/有名な論文_2026-11-07/MER.md`（F-MER），`research_notes/先行研究_信頼度の表_2026-11-13/report.md`（R-CONF），`research_notes/先行研究_水準はK本・相性はソース_2026-11-14/report.md`（R-LEV），`paper/notes/書き方_有名なMER論文_コーパス間系_2026-11-15.md`（W-CC），`paper/notes/書き方_有名なMER論文_融合系_2026-11-15.md`（W-FU）．前の照合 `reviews/v2_ja/citations.md`（v2ja），`reviews/restructure_2026-11-16/citations_02.md`（RS），`reviews/round1_ja/citations_01.md`・`citations_02.md` の判定と比べた．
日本語の ver2（`paper/ja_v2/`）と文が同じで，前の照合で「合う」とした引用は判定を引き継ぎ，表には載せていない．

## 判定

**直してから通す．**

- v2ja の 12 個の指摘は，英語版にすべて入っている（下の「v2ja の指摘の反映」）．7.2 節と限界の節を短くしたことで，引用の文が記録と合わなくなった箇所は無い．
- 新しく見つけた取り違えが 3 か所ある（番号 1〜3）．番号 1 と 2 は 2.1 節を縮めたときに入った誤りで，日本語の ja_v2 の 2.1 節にも同じ誤りがある（v2ja は見落とした）．番号 3 は序論の 1 段落目の Gideon ほかの文で，これまで記録と照らされていなかった．
- 言い過ぎ（軽）が 1 か所（番号 4），軽い注意が 4 か所（番号 5〜8）ある．
- bib の書誌の誤りは無い．本文で引く 53 個の鍵はすべて bib にあり，PDF の参考文献も 53 本で，未解決の引用（[?]）は無い．bib には本文で引かない項目が 20 個ある（番号 9）．

## 表

| 番号 | 重さ | ファイルと行 | 原文 | 文献 | 記録の内容 | 判定 | 直し方 |
|---|---|---|---|---|---|---|---|
| 1 | 中 | v2_02_related.tex 5 行 | "For semi-supervised domain adaptation, in which the target has a few labels, methods have been proposed for each information source: learning representations that do not reveal the domain \cite{ganin2016dann}, aligning feature covariances \cite{sun2016coral}, and using the entropy or pseudo-labels of predictions for unlabelled data \cite{saito2019mme,grandvalet2004,lee2013pseudolabel}." | ganin2016dann，sun2016coral，grandvalet2004，lee2013pseudolabel | RS 番号 2・3・6（`有名な論文_2026-11-07/半教師ありDA.md` 295・315 行：DANN と CORAL はターゲットにラベルが無い教師なしの DA．161・179 行：Grandvalet と Lee はドメインのずれの無い半教師あり学習．484 行：半教師ありの DA の手法は MME だけ） | 取り違え | 「For semi-supervised domain adaptation ... methods have been proposed」は，5 本がどれも半教師ありの DA のために提案されたと読める．序論の同じ誤り（v2ja 番号 1）は直したが，2.1 節を縮めたときに同じ誤りが入った．日本語の ja_v2 の 2.1 節（「ターゲットに少数のラベルがある半教師ありのドメイン適応では，…が，使う情報源に応じて提案されてきた」）も同じなので，両方を直す．案："Methods usable in semi-supervised domain adaptation, in which the target has a few labels, have been proposed for each information source: learning representations that do not reveal the domain \cite{ganin2016dann} and aligning feature covariances \cite{sun2016coral} for unlabelled targets, entropy minimisation and pseudo-labels from semi-supervised learning \cite{grandvalet2004,lee2013pseudolabel}, and minimax entropy for semi-supervised domain adaptation \cite{saito2019mme}." |
| 2 | 軽 | v2_02_related.tex 5 行 | "Adding source data can also lower target performance (negative transfer) \cite{wang2019negative}. These studies compare methods that combine information sources and do not measure, in one setting, which part of the classifier error each source corrects." | wang2019negative | RS 番号 7・8（Wang ほかは負の転移を「ソースあり − なし」で定義して要因を調べた研究で，手法を比べる研究ではない．RW:869） | 取り違え | 「These studies」の直前に Wang ほかがあるので，Wang ほかも「手法を比べた研究」に入る．RS 番号 8 で直した文（「これらの手法の研究は」）が，縮めたときに元に戻っている．ja_v2 の 2.1 節も同じ．案："The method studies above compare methods that combine information sources and ..." |
| 3 | 軽〜中 | v2_01_introduction.tex 4 行 | "Factors that change across corpora and can affect emotion recognition include the recording environment and devices, acted versus natural expression, the number and attributes of speakers, and the annotation scheme \cite{gideon2021addog}." | gideon2021addog | W-CC 259 行（Gideon ほかの序論 2 段落目の原文："Some factors that change across datasets and can impact affect recognition include the environmental noise [4], the spoken language [3], the recording device quality [5], and the elicitation strategy (acted versus natural) [6]. Additionally, a mismatch in subject demographics between datasets can result in misclassification, due to the small numbers of participants ..."） | 取り違え | Gideon ほかが挙げた要因は，雑音，言語，録音機器の質，誘発の方法，話者の属性の違いである．「the annotation scheme」は Gideon ほかの一覧に無い．「the number ... of speakers」は，Gideon ほかでは要因ではなく「参加者が少ないので属性の違いが誤りを生む」という理由である．注釈の方式の違いは表 1 が示すこの論文の観察なので，引用から外す．案："Factors that change across corpora and can affect emotion recognition include environmental noise, recording devices, acted versus natural elicitation and speaker demographics \cite{gideon2021addog}, and the corpora also differ in how they were annotated (Table~\ref{tab:corpora})." 日本語の 01_序論の 1 段落目も同じ文なので，そろえて直す |
| 4 | 軽 | v2_02_related.tex 17 行 | "multimodal models trained on few samples have also been reported to fall below the best unimodal model \cite{huang2021}" | huang2021 | 候補 #35（「標本が少ないと融合は最良の単モダリティに負けうる（IEMOCAP で学習の標本を減らすと，3 モダリティが言葉だけに負ける）」．本文を読んだ，付録は未読） | 言い過ぎ（軽） | 「have been reported to fall below」は，標本が少なければ必ず下回ると読める．記録は「負けうる」で，日本語の ja_v2 も「下回りうる」と書いている．案："multimodal models trained on few samples can fall below the best unimodal model \cite{huang2021}" |
| 5 | 軽 | v2_02_related.tex 14 行 | "Zadeh et al.\ \cite{zadeh2017tfn} criticised late fusion, which trains unimodal classifiers separately and combines their answers, for assuming that simple weighted averaging is a proper fusion approach." | zadeh2017tfn | W-FU 500 行（原文："This prevents the model from learning inter-modality dynamics in an efficient way by assuming that simple weighted averaging is a proper fusion approach."） | 合う（注意） | 英語版は批判の中身（モダリティの間の関係を学べない）を落とし，仮定だけを残した．日本語の ja_v2 は「モダリティの間の相互作用を学べないと批判した」まで書いている．批判の向きを正確に伝えるには中身を残す．案："... for not learning inter-modality dynamics because it assumes that simple weighted averaging is a proper fusion approach." |
| 6 | 軽 | v2_06_rq2.tex 79 行（表 7 の Source の列） | "Random search of Kaya et al.\ \cite{kaya2017}" | kaya2017 | R-LEV:118（Kaya ほかは AFEW の開発データの上で，乱数で作った行列の候補から，モデル × クラスの重みを探した），v2ja 番号 10（日本語は「の型」で原典との違いを残していた） | 合う（注意） | 本文の比べる相手は，K 本の UAR で 1,000 個の表と等しい重みの表から選ぶ版で，Kaya ほかの手続き（開発データ，候補の数）とは違う．英語版で「の型」が落ちたので，Kaya ほかの手法そのものと読める．同じ表の「Close to a recall combiner」と同じく違いを残す．案："Random search as in Kaya et al.\ \cite{kaya2017}" |
| 7 | 軽 | v2_01_introduction.tex 8 行，v2_07_discussion.tex 14 行 | 序論 "Busso et al.\ \cite{busso2004} reported that audio and visual classifiers recognise different emotions well."／7.2 節 "Busso et al.\ \cite{busso2004} found modality strengths to differ by emotion within one corpus" | busso2004 | R-CONF 4.3 節（女優 1 人の 258 文，4 感情．顔はモーションキャプチャのマーカー．leave-one-out で全ラベル） | 合う（注意） | 2 つの文の中身は記録と合う．ただし Busso ほかの「顔」は動画ではなくモーションキャプチャで，データは女優 1 人である．本文は V を動画の顔の特徴の意味で使うので，序論の「visual classifiers」は動画の顔の分類器と読まれうる．直すなら，序論を "audio and facial-expression classifiers"，7.2 節の "within one corpus" を "in recordings of one actress" にする．必須ではない |
| 8 | 軽 | v2_08_limitations.tex 4 行 | "fine-tuned encoders have been reported to give better dimensional emotion prediction (CCC) \cite{wagner2023}" | wagner2023 | RW:776-779（MSP-Podcast で微調整し，コーパスの中で凍結より CCC が +0.03〜+0.10 高い） | 合う | v2ja 番号 6 の直しが入っている．差はコーパスの中の比べなので，気になるなら "within-corpus" を足す．必須ではない |
| 9 | 軽（書誌の整理） | references.bib | 本文で引かない 20 項目：alexandari2020，bendavid2010，daume2007，gulrajani2021，hendrycks2018glc，heskes1997，holm1979，joshi2012，kahou2015，kumar2022lpft，pasad2023，pepino2021，raudys1998，sindhwani2008，tellamekala2024，tukey1949，wei2021，yang2021superb，yu2021selfmm，yun2024telme | — | v2ja 番号 10（付録【TBA】を書くときに tellamekala2024，kuncheva2014，heskes1997 を戻す） | — | `\nocite` は無いので，PDF の参考文献には出ていない（PDF は 53 本で，本文の引用の数と一致した）．投稿の前に，付録で戻す予定の heskes1997・tellamekala2024 を除いて bib から外すか，そのまま残すかを決める．原稿の誤りではない |

## v2ja の指摘の反映（英語版で確かめたこと）

| v2ja の番号 | 英語版の箇所 | 状態 |
|---|---|---|
| 1（序論の「この問題には…提案されてきた」） | v2_01 6 行 "Methods usable in this problem have been proposed for each information source" | 直っている．ただし 2.1 節に同じ誤りが入った（番号 1） |
| 2（Gideon ほかは声だけ） | v2_01 4 行 "Emotion classifiers trained on one corpus ..." | 直っている．要旨の 1 文目 "MER classifiers ... have been reported to degrade" には引用が無く，序論の Ryumina ほか（MER）が支えるので可 |
| 3（蒸留の定義と温度） | v2_05 22 行 "(the pseudo-label probabilities are targets without temperature)" | 直っている |
| 4（損失の 5 つの形の出典） | v2_05 24 行 | 直っている（5 つの数，形ごとの出典，"our variant of the idea of \cite{blum1998}"，"is not identical to the original method"） |
| 5（Cameron ほかの位置） | v2_03 69 行 "because inference with few clusters is anti-conservative \cite{cameron2008}, jackknife $t$ intervals are used" | 直っている．引用は理由の文に掛かっている |
| 6（Wagner ほかは CCC） | v2_08 4 行 | 直っている（番号 8） |
| 7（7.2 節で「示した」のは Albanie ほか） | v2_07 14 行 | 直っている．英語版は 7.2 節から Zhang ほかを外し，Albanie ほかだけに "showed" を使う |
| 8（Zhang ほかとの違いを言葉に置く） | v2_02 17 行 "Unlike Albanie et al., who used facial predictions only, and Zhang et al., who mixed facial and audio predictions, our pseudo-labels are built from three classifiers including text." | 直っている |
| 9（見出しの「半教師ありのドメイン適応」） | v2_07 14 行 "Whereas domain adaptation studies compared methods ..." | 直っている（英語版は箇条をやめ，「domain adaptation studies」と書く） |
| 10（表 7 の出典の列） | v2_06 74〜79 行 | Kittler，Kuncheva，Guo は合う．Kaya は「の型」が落ちた（番号 6） |
| 11（5.4 節の Albanie ほか） | v2_05 142 行 | 合う |
| 12（Ryumina ほかと IEMOCAP の即興） | v2_02 8 行 "(IEMOCAP, MOSEI, MELD and AFEW; we use only the improvised part of IEMOCAP)" | 直っている．1.7〜6.6 点は PDF の値と一致する（v2ja の確認） |
| 位置づけの 1 行目（7.2 節で Xue ほかにふれる） | v2_07 14 行 "training on more accurate predictions of another modality has been reported not to always help \cite{xue2023mfh}" | 入っている．候補 #33（別のモダリティの教師の正解率が高くても生徒が良くなるとは限らない．表 1）と合う |

## 7.2 節と限界の節を縮めた影響

- 7.2 節（v2_07 14 行）：引用は albanie2018，xue2023mfh，ganin2016dann，saito2019mme，busso2004，ryumina2025 の 6 本で，文献に言わせていることは日本語の ja_v2 の 7.2 節と同じか，その一部である．日本語にあった Zhang ほかと Kaya ほかの文は外れたが，外したことで記録と合わなくなった文は無い．Xue ほか（正解率が高くても上がるとは限らない）と本研究の結果（当たる幅が負なら上がらない）は同じ向きの別の主張で，並べ方に矛盾は無い．
- 限界の節（v2_08）：引用は wagner2023 の 1 本だけで，記録と合う（番号 8）．DANN と MME を線形の版にしたことは本文の手続きの説明で，引用は付いていない．

## 本文で引いていない文献と bib に無い引用

- bib に無い引用：無い（53 個の鍵がすべて bib にある）．
- 本文で引いていない bib の項目：20 個（番号 9）．

## 書誌（`paper/references.bib`）

- 本文で引く 53 項目の著者・年・題・掲載先は，記録（候補，F-MER，W-CC，W-FU）と前の照合（round1_ja，RS，v2ja）で確かめた書誌と一致する．iscen2019（CVPR 2019，pp. 5065–5074）と he2025m4ser（IEEE Transactions on Audio, Speech and Language Processing 33:4055–4070）は，今回 Crossref でも確かめた．
- PDF の参考文献では，Póczos，Gürpınar，Marín，Rodríguez の文字は正しく組まれている．
- 前から残る注意：shi2024audioguided は arXiv だけで，掲載先（MRAC '24．ACM MM 2024 のワークショップ）を確かめていない．marin2026classwise は arXiv の注に ABAW（ECCV 2026 のワークショップ）採録とある．掲載版が出ていれば更新する．grandvalet2004 の年（会議 2004，刊行 2005）は全体でそろえるだけでよい．

## 位置づけに要る文献

- 欠けている文献は無い（研究者の方針により，先行研究の段落を増やすことは求めない）．
