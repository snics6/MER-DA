# 引用の照合：ja_v2 の序論と 2.5 節（2026-11-17）

対象：`paper/ja_v2/01_序論.md`（全体），`paper/ja_v2/02_関連研究.md` の 2.5 節（23 行）．bib は `paper/references.bib` の末尾の 2 件（jaritz2023xmossda，zhang2024amanda）と，対象の文が引く鍵．

照らした記録：
- R25：`research_notes/先行研究_Ryumina2025の新規性と蒸留の比べ_2026-11-17/report.md`
- RyPDF：`paper/references/2025_Ryumina_MultiCorpusEmotionRecognitionCrossModalGated_PRL.pdf`（本照合で表 1，2 節，表 5，表 7 を読み直した）
- AmPDF：Amanda の ACL Anthology の PDF（本照合で 4 節と表 1 を読み直した）
- F-MER：`research_notes/有名な論文_2026-11-07/MER.md`（TFN，Self-MM の項と 3.2 節の表）
- 書MER：`paper/notes/書き方_有名なMER論文_融合系_2026-11-15.md`，`paper/notes/書き方_有名なMER論文_コーパス間系_2026-11-15.md`
- Q1・Q3：`docs/参照/DAI_paper/参照論文/要点_問い1_別モダリティ蒸留.md`，`要点_問い3_同じ設定の研究.md`
- 信頼度：`research_notes/先行研究_信頼度の表_2026-11-13/report.md`
- 候補：`paper/notes/引用の候補_2026-11-14.md`
- 前回：`reviews/v2_ja/citations.md`
- M3ER と TelME の評価の分け方は，前の調べで落とした PDF のテキスト（M3ER の 5 節："split the CMU-MOSEI dataset into training (70%), validation (10%), and testing (20%) sets"）と RyPDF の表 7 で確かめた．

## 判定

**直してから通す．**重い指摘が 1 か所ある（番号 7：序論 2 段落目の「マルチモーダルでコーパス間を扱った研究は，ターゲットのラベルを使っていない」は，2.2 節で引く He ほか（M4SER）と食い違う）．中くらいの指摘が 2 か所（番号 1：TFN と Self-MM は感情極性の推定の論文，番号 12：Busso ほかの結果の「コーパスをまたいでも」の範囲），軽い指摘が 4 か所ある．研究者が挙げた Ryumina ほかの文（番号 3）は，原文の範囲と合う．範囲を「いずれかで評価された 12 手法」と書けば読み違いがなくなる．2.5 節の 3 本の説明（番号 16〜18）は記録と原文に合う．言い方を少し直す箇所がある．新しい 2 件の bib は Crossref の記録と一致する．

## 表

重さ：重＝査読で取り違えと指摘される／原稿の中で食い違う，中＝記録より強いか範囲が曖昧，軽＝語の選び方，—＝直さなくてよい．

| 番号 | 重さ | 行 | 原稿の文 | 文献 | 記録の内容 | 判定 | 直し方 |
|---|---|---|---|---|---|---|---|
| 1 | 中 | 01:5（1 文目） | 「マルチモーダル感情認識（MER）は，同じ発話の声・顔・言葉から話し手の感情を推定する課題であり，モダリティの情報をまとめる融合の方法が数多く提案されてきた [@zadeh2017tfn; @mittal2020m3er; @yu2021selfmm; @yun2024telme]」 | zadeh2017tfn，mittal2020m3er，yu2021selfmm，yun2024telme | F-MER 251 行（TFN の題は "for Multimodal Sentiment Analysis"．評価は MOSI の感情価 −3〜+3，620 行），344・621 行（Self-MM も MOSI・MOSEI の感情価で，題は "for Multimodal Sentiment Analysis"）．M3ER（IEMOCAP 4 感情，MOSEI 6 感情）と TelME（IEMOCAP，MELD）は感情のカテゴリ | 取り違え（4 本のうち 2 本） | TFN と Self-MM は感情のカテゴリではなく感情極性（感情価）を推定する論文で，「話し手の感情を推定する課題」の融合の方法の例としては範囲がずれる．案：「…推定する課題であり，感情極性の推定を含めて，モダリティの情報をまとめる融合の方法が数多く提案されてきた [同じ 4 本]」 |
| 2 | — | 01:5（2 文目の前半） | 「これらの方法は，1 つのコーパスの中で学んでテストする形で評価されることが多く」 | 同上の 4 本 | 4 本ともコーパスの中で評価した．TFN は MOSI の 5 分割の交差検証（話者独立，F-MER 620 行），M3ER は MOSEI を 70/10/20 に分け IEMOCAP も同じ形，Self-MM は MOSI・MOSEI・SIMS の公式の分割（F-MER 621 行），TelME は IEMOCAP と MELD（RyPDF 表 7） | 合う | 「多い」は 4 本と RyPDF の 12 手法から言える範囲に収まる |
| 3 | 軽 | 01:5（2 文目の後半） | 「Ryumina ほか [@ryumina2025] は，IEMOCAP，MOSEI，MELD，AFEW で評価された最近の方法がコーパス間では評価されていないことを指摘した」 | ryumina2025 | RyPDF 2 節冒頭 "This section summarizes the SOTA methods that are developed based on the corpora used in this research and can be compared with our method."，同節 "while these methods are evaluated on multiple corpora, cross-corpus evaluation is neglected"．表 1 の題は "Systematization of recent MER methods"（2019〜2024 年の 12 手法．TelME〔[21]〕を含む）．表 7 では，各手法は 4 コーパスのうち 1〜2 個で評価されている（MOSEI 4 本，IEMOCAP 4 本，MELD 4 本，AFEW 3 本．AFEW の 3 本は声と顔だけ）．R25 1.4 節も「表 1 の 12 手法についての文で，MER の研究全体についての文ではない」と書く | 合う（範囲を明記するとよい） | 主張の中身（最近の方法・コーパス間の評価をしていない・指摘した）は原文と合う．ただし「IEMOCAP，MOSEI，MELD，AFEW で評価された」は，各手法が 4 コーパスすべてで評価されたと読める．案：「Ryumina ほか [@ryumina2025] は，IEMOCAP，MOSEI，MELD，AFEW のいずれかで評価された最近の 12 手法が，コーパス間では評価されていないことを指摘した」 |
| 4 | — | 01:5（3 文目） | 「感情のコーパスは，周りの雑音，録音の機器，演技か自然な表出か，話者の属性が互いに違い [@gideon2021addog]」 | gideon2021addog | 書MER コーパス間系 260 行（ADDoG の 2 段落目 "the environmental noise [4], the spoken language [3], the recording device quality [5], and the elicitation strategy (acted versus natural) [6]. Additionally, a mismatch in subject demographics …"），287 行（結論で同じ 4 つを再掲） | 合う | なし．前回の照合で直した形（「感情の分類器」）のまま |
| 5 | — | 01:5（4 文目） | 「1 つのコーパスで学んだ感情の分類器は，ほかのコーパスでは成績が落ちることが報告されている [@ryumina2025; @gideon2021addog]」 | ryumina2025，gideon2021addog | RyPDF 表 5 と本文："the performance decreases by 28.7% (42.6% vs. 71.3%), 13.7% …, 31.8% …, and 39.1% … with single-corpus training and multi-corpus testing"（UAR．1 つのコーパスで学んで残りで測った）．ADDoG は声のコーパス間 | 合う | なし |
| 6 | — | 01:7（1 文目） | 「新しいコーパスに分類器を合わせる研究は，主に声の感情認識で行われてきた [@abdelwahab2018dann; @gideon2021addog; @feng2023]」 | abdelwahab2018dann，gideon2021addog，feng2023 | 3 本とも声だけ（書MER コーパス間系，Q3 1 節）．マルチモーダルの DA の例（Amanda，Yin 2020）もあるが「主に」で範囲を残している | 合う | なし |
| 7 | **重** | 01:7（2 文目） | 「マルチモーダルでコーパス間を扱った研究 [@ryumina2025; @lian2026merbench] は，ターゲットのラベルを使っていない」 | ryumina2025，lian2026merbench（暗に he2025m4ser） | 2 本については正しい（R25 1 節：Ryumina は LOCO で，外したコーパスはテストだけ．R25 2.1 節：MERBench は 1 コーパスで学んで別のコーパスで評価）．しかし Q3 0 節 1 は「『マルチモーダル × 少数ラベル × コーパス間』の先行研究はある（M4SER，Li ほか 2021）」と書く．Q3 3 節：M4SER は声と Whisper の書き起こしの言葉を使い，IEMOCAP ↔ MELD で，ターゲットのクラスあたり 5・10・20・60 本のラベルを学習に使った．原稿の 2.2 節も He ほか [@he2025m4ser] をターゲットに少数のラベルがある研究として引いている．候補 42 行は「Feng & Chaspari と M4SER を引かないと『初めて』と読める書き方になり，査読で指摘される」と書く | 言い過ぎ（原稿の 2.2 節と食い違う） | 主語の「マルチモーダルでコーパス間を扱った研究」が分野の全体を指すので，M4SER が反例になる．案：「Ryumina ほか [@ryumina2025] と Lian ほか [@lian2026merbench] のマルチモーダルのコーパス間の評価は，ターゲットのラベルを使っていない．ターゲットに少数のラベルがある設定を扱った He ほか [@he2025m4ser] は，声と書き起こしの言葉の 2 つを使う．」Amanda もラベルを使わないマルチモーダルのコーパス間の研究なので，1 文目の括弧に [@zhang2024amanda] を足してもよい（鍵は bib にすでにある） |
| 8 | 軽 | 01:7（3 文目） | 「同じ発話のほかのモダリティの予測を目標にして声の分類器を学ぶ研究があり [@albanie2018; @zhang2021combining]」 | albanie2018，zhang2021combining | Q1 3 節（Albanie：顔の教師の予測で声の生徒を学ぶ），R25 3.2 節（Zhang 2021：顔だけの DCMKT と，顔と声を混ぜた疑似ラベル．最良は顔と声の混合） | 合う（注意） | Zhang ほかの主な方法の目標には声自身の予測が混ざる．2.5 節で「顔と声の予測を混ぜた Zhang ほか」と書き分けているので，序論も合わせるなら「ほかのモダリティの予測を含む目標にして」とする |
| 9 | — | 01:7（3 文目） | 「1 つのコーパスの中では声と顔で当てやすい感情が違うことも報告されている [@busso2004]」 | busso2004 | 信頼度 94 行・4.3 節（女優 1 人，258 文，4 感情，leave-one-out．声は喜びと怒りを互いに取り違え，顔は喜びが 1.00．コーパス内） | 合う | なし |
| 10 | — | 01:9（1・2 文目） | 「感情のラベルは複数の評定者が…費用がかかる [@parthasarathy2020ladder; @abdelwahab2018dann]．一方，ラベルの無い録音は集めやすい [@abdelwahab2018dann]」 | parthasarathy2020ladder，abdelwahab2018dann | 書MER コーパス間系 208・209 行（Abdelwahab & Busso："Due to the prohibitive cost of annotating new data every time we change the target domain and the abundance of unlabeled data"）．前回の照合を通った文で，「新しいコーパスのたびに多く付ける」に変わったが中身は同じ | 合う | なし |
| 11 | — | 01:9（4 文目） | 「半教師ありのドメイン適応 [@saito2019mme] で使うラベル付きのほかのコーパス（ソース），K 本のラベル，ターゲットのラベル無しの発話の 3 つ」 | saito2019mme | 候補 116 行（この論文の設定は MME が定めた SSDA の設定にあたる） | 合う | なし．前回の番号 1 の取り違え（DANN などを SSDA の手法と読ませる文）は，この版では消えている |
| 12 | 中 | 01:9（最後の文の後半） | 「Busso ほかの結果がコーパスをまたいでも成り立つかも分かっていない」 | busso2004 | 信頼度 96 行：Lingenfelser ほか 2011 は DaFEx（俳優 8 人）と CALLAS で，感情ごとにモダリティの強さが違うことを報告した（例：DaFEx で声は恐れ 0.43 対 顔 0.11，顔は喜び 0.82 対 声 0.21）．同 103 行：「今回見た研究はすべてコーパス内」．Q1 0 節 4：Albanie は感情ごとの違いを考察で予想しただけで測っていない | 言い過ぎ（範囲が曖昧） | 「コーパスをまたいでも」は「ほかのコーパスでも」と読める．その読みでは Lingenfelser ほかが別のコーパスで同じ向きを報告しているので，「分かっていない」と言えない．記録で言えるのは「コーパス間の設定（ソースで学んだ分類器をターゲットで使う）での報告は見つからなかった」までである．案：「ソースのコーパスで学んだ分類器をターゲットで使うときにも，声と顔で当てやすい感情が違うかは報告されていない」 |
| 13 | — | 01:9（最後の文の前半） | 「Albanie ほかと Zhang ほかは，ターゲットのラベルが感情ごとに数本しか無いコーパス間の設定では評価しておらず」 | albanie2018，zhang2021combining | Q1 3 節・R25 3.2 節（Albanie：VoxCeleb で学んだ生徒を RML と eNTERFACE で，生徒の出力の上のアフィン写像を 10 分割の交差検証で学んで測った．数本のラベルの設定は無い）．R25 3.2 節（Zhang 2021：IEMOCAP のラベルを全部使う） | 合う | なし |
| 14 | — | 01:11 | 「…蒸留と呼ぶ [@hinton2015distilling; @albanie2018]」 | hinton2015distilling，albanie2018 | 前回の番号 3（合う．注意） | 合う | なし．前回の文から変わっていない |
| 15 | — | 01:24（貢献 2 の 2 文目），02:23（4 文目） | 「我々の知る限り，感情ごとに数本のラベルしか無いコーパス間の MER で，ほかのモダリティを含む疑似ラベルの蒸留と，ほかのモダリティを使わないドメイン適応・半教師あり学習・テスト時適応の手法を同じ条件で比べた研究は無い」 | （引用なし．2.5 節で zhang2021combining，jaritz2023xmossda，zhang2024amanda を並べる） | R25 5.3 節：「今回の検索と手元のノートの範囲では」見つからなかった．書くなら "To the best of our knowledge" を付け，コーパス間・感情ごとに数本のラベル・3 モダリティの条件を同じ文に入れ，同じ段落で Zhang 2021，xMUDA・xMoSSDA，Amanda との違いを 1 文ずつ書く．R25 5.2 節：「クロスモーダルの学習と単一モダリティの DA の比べは初めて」とは書けない | 合う | 範囲（コーパス間，感情ごとに数本，MER）は R25 5.3 節の条件を満たし，5.2 節の書けない形（課題を限らない「初めて」）になっていない．2.5 節に近い比べの 3 本がある |
| 16 | 軽 | 02:23（6 文目） | 「Zhang ほか [@zhang2021combining] は声の感情認識で顔の疑似ラベルと同じモダリティの半教師あり学習を比べたが，ターゲットのラベルを全部使い，ドメイン適応の手法とは比べていない」 | zhang2021combining | R25 3.2 節（◎）：ターゲットは IEMOCAP（8 話者で学び 1 話者で検証，1 話者で評価）と CH-SIMS．ラベルは全部使い，ラベル無しのデータは別のコーパス（EmoVoxCeleb，iQIYI-VID）から足す．比べた相手は FixMatch，π-model，Temporal ensembling と DCMKT（顔だけの疑似ラベル）．提案の疑似ラベルは顔と声の混合．DA の手法は無い | 合う（語の注意） | (a)「顔の疑似ラベル」は，同じ段落の前の文の「顔と声の予測を混ぜた Zhang ほか」と食い違って読める．(b) Zhang ほかは DA の設定ではないので「ターゲット」の語が合わない．案：「Zhang ほか [@zhang2021combining] は声の感情認識で，顔の予測を含む疑似ラベルと同じモダリティの半教師あり学習を比べたが，評価するコーパス（IEMOCAP）の学習用のラベルを全部使い，ドメイン適応の手法とは比べていない」 |
| 17 | 軽 | 02:23（7 文目） | 「Jaritz ほか [@jaritz2023xmossda] は，ほかのモダリティを使う学び方と単一モダリティの半教師ありのドメイン適応の手法を比べたが，課題は 3 次元の意味分割である」 | jaritz2023xmossda（関連して jaritz2020xmuda） | R25 3.2 節（◎．arXiv 2101.07253）："We compare xMoSSDA against eight baselines."，単一モダリティの相手は "three SSDA baselines (trained on S + Tℓ + Tu) adapted from uni-modal UDA baselines"（Deep logCORAL，MinEnt，PL）．結果 "outperforms all baselines by up to +2.2 in 8 out of 9 cases"．R25 は xMUDA（CVPR 2020）も同じ形の比べ（単一モダリティの UDA 手法と）としてまとめている | 合う | 中身は合う．相手の 3 手法は UDA の手法を SSDA に移した版なので，厳密には「単一モダリティのドメイン適応の手法を半教師ありの設定に移した版」．xMUDA は 2.3 節で [@jaritz2020xmuda] として引いているので，ここで [@jaritz2020xmuda; @jaritz2023xmossda] と並べると，教師なしと半教師ありの両方の比べを 1 文で示せる（文献は増えない）．どちらも任意 |
| 18 | 軽 | 02:23（8 文目） | 「Zhang ほか [@zhang2024amanda] は，IEMOCAP，MELD，MOSEI，MSP-IMPROV の間のマルチモーダルの教師なしのドメイン適応をドメイン適応の手法と比べたが，ほかのモダリティの疑似ラベルによる蒸留とは比べていない」 | zhang2024amanda | AmPDF 4 節（本照合で読み直した）："four widely used benchmark multimodal emotion recognition datasets, IEMOCAP …, MELD …, CMU-MOSEI …, and MSP-IMPROV … which all contain acoustic, visual and lexical modalities"，4 感情，"We compare our model, Amanda, with DANN …, CDAN …, MADA …, DALN …"，単一モダリティ用の手法は "we directly conduct domain adaptation for each modality"．表 1 はコーパスの対 7 組の重みつき F1．本文に distillation と pseudo の語は無い．R25 3.2 節：ターゲットのラベルは使わず，融合の分類器だけを評価 | 合う（語の注意） | (a)「教師なしのドメイン適応をドメイン適応の手法と比べた」は，何を比べたかが読み取りにくい．Amanda は手法を提案した論文である．(b) 同じ段落に「Zhang ほか」が 2 本あり，どちらかが分からない．(c) 本研究との違いとして，ターゲットのラベルを使わない点が要る．案：「Zhang ほか（2024）[@zhang2024amanda] は，IEMOCAP，MELD，MOSEI，MSP-IMPROV の間のマルチモーダルの教師なしのドメイン適応の手法（Amanda）を提案し，DANN などのドメイン適応の手法と比べたが，ターゲットのラベルは使わず，ほかのモダリティの疑似ラベルによる蒸留とは比べていない」．6 文目も「Zhang ほか（2021）」とする |
| 19 | — | 02:23（1〜3 文目，9 文目以降） | 2.5 節の残りの文（albanie2018，busso2004，ryumina2025，huang2021，M3ER，Marín ほか） | — | 前回の番号 8（3 文目は前回の案どおりに直っている）．ほかの文は前回の版から変わっていない（git diff で確かめた） | 合う | なし |

## bib の照合

| 鍵 | 照らした先 | 結果 |
|---|---|---|
| jaritz2023xmossda | Crossref（DOI 10.1109/TPAMI.2022.3159589）：題 "Cross-Modal Learning for Domain Adaptation in 3D Semantic Segmentation"，IEEE TPAMI 45(2):1533–1544，2023 年 2 月，著者 Jaritz，Vu，de Charette，Wirbel，Perez | 一致する（bib の P{\'e}rez は正しい綴り）．R25 は arXiv 2101.07253 の本文を読んでおり，TPAMI の掲載版とは照らしていない．引く文の中身（比べの相手，課題）が変わる見込みは小さい |
| zhang2024amanda | Crossref（DOI 10.18653/v1/2024.findings-acl.859）：題 "Amanda: Adaptively Modality-Balanced Domain Adaptation for Multimodal Emotion Recognition"，Findings of ACL 2024，14448–14458 頁，著者 Zhang，Sun，Hong，Li | 一致する |
| lian2026merbench | Crossref（DOI 10.1109/tpami.2026.3653457）：TPAMI 48(5):5793–5810，2026 年 5 月 | 一致する．候補 60 行の「IEEE TAC 採録と思われる」は古い記録なので，候補のノートを直すとよい（原稿は直さなくてよい） |
| zhang2021combining | Crossref（DOI 10.1016/j.knosys.2021.107340）：KBS 229:107340，2021 年 10 月，著者 7 人 | 一致する |
| ryumina2025 | RyPDF の 1 頁目：PRL 190 (2025) 192–200，著者 5 人 | 一致する |
| zadeh2017tfn，mittal2020m3er，yu2021selfmm，yun2024telme，gideon2021addog，abdelwahab2018dann，feng2023，albanie2018，busso2004，parthasarathy2020ladder，saito2019mme，hinton2015distilling | F-MER 685・691 行，候補の表，前回の照合 | 記録と合う．TelME と Busso 2004 は候補 232 行で「正式な題が記録に無い」とされたままだが，bib の題は TelME が NAACL 2024 の題，Busso 2004 が ICMI 2004 の題で，信頼度 94 行の PDF の記録と矛盾しない |

## 欠けている文献（1 行）

- 序論 2 段落目にターゲットの少数のラベルを使うマルチモーダルのコーパス間の研究として He ほか [@he2025m4ser] が要る（番号 7．bib にすでにある）．
