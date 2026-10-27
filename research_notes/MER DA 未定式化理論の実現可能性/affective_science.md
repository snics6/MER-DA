# 感情科学の理論のうち，コーパス間・文化間のずれを説明するが DA の学習法になっていないもの（2026-10-28 時点）

確かめた水準の記号：◎ 本文を読んだ／○ 抄録を読んだ／△ 書誌か検索結果（検索の要約）のみ．△ の数値は，引用の前に原典で確かめる．
`docs/related-work.md` に既にある文献（Schuller 2010，Milner 2019，Li ほか 2021，Feng & Chaspari，Tavernor & Mower Provost 2025，Larsen ほか 2026）は「既出」と書き，中身を繰り返さない．
「定式化されていない」は，下の 7 節末に書いた検索語と場所で探して見つからなかったという判定である．

## 1. Brunswik のレンズ模型と，表出・伝達・知覚の分解をコーパス間のずれや DA に使った研究

### Takeaway
レンズ模型は，話し手の感情 → 測れる手がかり（遠位の手がかり）→ 聞き手の受け取った印象（近位の知覚）→ 聞き手の判断，という段に分けて，どの段で情報が失われるかを数える模型である．感情の声の研究では Bänziger, Hosoya & Scherer 2015 が経路模型として実測し，Laukka ほか 2016 が 5 か国の比較に使った．ただし，この分解を DA の学習法（ソースとターゲットの分類器のずれを段ごとに分けて直す方法）にした研究は見つからなかった．

### Cited Findings
- （◎）Bänziger, Hosoya & Scherer 2015（PLOS ONE 10(9): e0136675）は，演技者の表した感情（怒り・恐れ・喜び・悲しみと覚醒）→ PRAAT で測った 6 つの音響量（平均の強さ，強さの幅，F0 の下限，F0 の幅，長さ，1000 Hz 未満のエネルギーの割合）→ 素人の聞き手が 6 つの尺度で評定した近位の知覚（抑揚・大きさ・高さ・粗さ・速さ・不安定さ）→ 聞き手の感情の判断，という経路模型を，2 つのコーパスを合わせた 304 本で推定した — [PMC4556609](https://pmc.ncbi.nlm.nih.gov/articles/PMC4556609/)
- （◎）同論文で，近位の知覚だけで説明できた聞き手の判断の分散は，覚醒 91%，怒り 64%，恐れ 49%，悲しみ 41%，喜び 27% だった．音響量を足しても上がり幅は感情で最大 0.04，覚醒で 0.009 だった — [PMC4556609](https://pmc.ncbi.nlm.nih.gov/articles/PMC4556609/)
- （◎）同論文で，表した感情から判断への効果のうち，音響量と知覚を通る経路で運ばれた割合は，怒り 0.33，恐れ 0.19，悲しみ 0.18，覚醒 0.84 だった．感情の種類では，測った手がかりを通らない直接の経路が 0.57〜0.67 と大きく，覚醒では 0.13 と小さかった — [PMC4556609](https://pmc.ncbi.nlm.nih.gov/articles/PMC4556609/)
- （◎）同論文は，レンズ模型の式（達成度＝生態学的妥当性 × 機能的妥当性 × 一致度 G ＋ 模型に入らない成分）で，表出側と知覚側の手がかりの使い方の一致度 G が高いと報告した．著者は，声は覚醒をよく伝え，快不快（喜び）をあまり伝えないと書き，機械学習による感情の検出もこの模型から得をしうると書いた．限界として，標本が少ないこと，演技の音声であること，中立の刺激が無いことを挙げた — [PMC4556609](https://pmc.ncbi.nlm.nih.gov/articles/PMC4556609/)
- （△）Laukka ほか 2016（Journal of Personality and Social Psychology 111(5)）は，英語圏 5 か国の職業俳優 100 人が 11 の感情を声で表した録音を，手がかりの音響分析とレンズ模型で調べた．手がかりの使い方は表出でも判断でも集団の間で似ていたが，系統的な差（著者のいう文化の「なまり」）もあり，集団内の優位は感情によって大きさが違った — [dspace.cus.ac.in](https://dspace.cus.ac.in/jspui/handle/1/6914)
- （○）MES-P（Xiao ほか，arXiv 1808.10095，2018．査読付きの掲載は未確認）は，中国語の感情音声に，話し手が意図した感情（遠位のラベル）と聞き手が知覚した感情（近位のラベル）の 2 層のラベルを付けたデータセットである．抄録には，2 層の差を使う学習法は書かれていない — [arXiv 1808.10095](https://arxiv.org/abs/1808.10095)
- （△）Busso & Narayanan 2008（Interspeech）のポスターは，遠位の手がかりと近位の知覚は伝達や解釈の途中でずれうると書き，自己評価を入れると評定者間の一致が下がると報告した — [Busso 2008 ポスター](https://www.lab-msp.com/MSP/publications/Busso_2008_4_poster.pdf)，[論文](https://msp.utdallas.edu/publications/Busso_2008_4.pdf)
- （△）Patel, Scherer らは，19 人の聞き手に 160 本の声を 12 の尺度（韻律 8，感情の次元 4）で評定させ，回帰で各尺度に対応する音響量を求めた（近位の知覚を音響と感情の間に置く設計） — [ASA Lay Language Paper](https://acoustics.org/pressroom/httpdocs/162nd/Patel_5aSCb.html)
- （△）Scherer の修正版レンズ模型を，感情と性格の行動の計算モデルに当てはめた図が流通している．検索で見つかったのは図の頁だけで，DA の手法ではない — [ResearchGate の図](https://www.researchgate.net/figure/Modified-version-of-Brunswik-lens-model-from-Scherer-3-which-has-been-applied-to-the_fig3_312584970)
- （△）「lens model」と「speech emotion recognition」「cross-corpus」「encoder decoder」を組み合わせた検索では，レンズ模型を自動の感情認識の学習法にした論文は出てこなかった．見つかった encoder-decoder の論文（脳波の被験者間の転移，EMBC 2024）は網の構造の名前で，レンズ模型とは無関係である — [EMBC 2024](https://pubmed.ncbi.nlm.nih.gov/40039562/)

### Inferences
- MER-DA の特徴（凍結したエンコーダの発話平均）は遠位の手がかりに当たり，多クラスのロジスティック回帰の重みは「手がかりの使い方」に当たる．コーパスのラベルが意図した感情（RAVDESS・CREMA-D の演技の指示）なら頭は表出側（生態学的妥当性），知覚した感情（IEMOCAP・MELD・MOSEI・AFEW の注釈者）なら頭は知覚側（機能的妥当性）の推定になる．MER-DA の 7 群は，ラベルがレンズのどちらの端にあるかで 2 つに分かれる．
- レンズ模型の一致度 G（表出側と知覚側の重みの相関）は，MER-DA の MH2 の向きの効率 e（ソースとターゲットの最適な頭のなす角）と同じ種類の量である．生態学的妥当性 R_e（手がかりが感情をどれだけ分けるか）は，MH5 の dprime（コーパスの感情の分かれ方の強さ）と同じ種類の量である．MH5 の「組の相性は dprime の差で決まる」は，レンズ模型の語では「生態学的妥当性の差で決まる」と書ける．
- Bänziger 2015 で直接の経路（測った手がかりを通らない部分）が感情の種類で 6 割前後あったことは，音響の手作りの量では感情の種類の情報の多くを取りこぼすことを示す．凍結した自己教師ありのエンコーダがこの取りこぼしをどこまで埋めるかは，文献からは分からない．

### Gaps
- Scherer 2003（Speech Communication 40）のレンズ模型の章と，レンズ模型の式の原典（Hammond, Hursch & Todd 1964 など）は，今回の検索で本文も抄録も取得していない．
- レンズ模型の各段を，2 つのコーパスの差（どの段でずれたか）の分解に使った研究は見つからなかった．Laukka 2016 は 5 か国の比較だが，本文を読んでいないので，段ごとの差の大きさは分からない．
- 「proximal percepts」を中間の表現にした機械学習の論文は 1 本（Amrita，2018，Berlin のデータ）が検索に出たが，題名と要約しか見ておらず，数値は確かめていない．

## 2. push 効果と pull 効果，演技・誘発・自然な感情表出の違い（理論と実証）

### Takeaway
Scherer の枠組みでは，push 効果は体の内側の変化が声や顔を押し出す部分，pull 効果は社会の決まりや伝えたい相手に合わせて形を整える部分である．実証では，演技の表出は自然な表出より強いと評定されやすいが，差は強さだけでなく，いくつかの手がかりの組み合わせ（どの感情でどの手がかりが動くか）にも出る（Juslin, Laukka & Bänziger 2018）．この区別を DA の学習法（強さを合わせる，型の違いを別に直す）にした研究は見つからなかった．

### Cited Findings
- （△）pull 効果は社会の決まりなど外の型に合わせて形づくられる表出，push 効果は表出の手がかりを決める体の内側の変化と定義される．Scherer の Interspeech 2015 基調講演の資料は，自然で抑えの効かない発声はおもに push 効果で決まり，手がかりの共変は push 効果，手がかりの配置は pull 効果を反映しやすいと書く — [Frontiers in Psychology 2019](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2019.00508/pdf)，[Scherer 2015 基調講演資料](https://www.ling.upenn.edu/courses/Spring_2018/ling620/SchererKeynoteInterspeech2015.pdf)
- （△）Banse & Scherer 1996（JPSP 70(3):614–636）は，職業俳優（12 人とする資料がある）の 14 感情の演技を聞き手に判断させ，224 本の音響の特徴を分析した．ジャックナイフ法の判別分析は高い的中率を示し，その取り違えの型は聞き手の取り違えとよく似ていた．的中率の数値は今回確かめていない — [Banse & Scherer 1996 PDF（Columbia）](http://www.columbia.edu/~rmk7/HC/HC_Readings/Scherer.pdf)，[Bonn 大学の PDF](https://psychologie.uni-bonn.de/en/department/departments/social-and-legal-psychology/uploads-fuer-die-allgemeine-i/1996bansescherer_acoustic-profiles-in-vocal-emotion-expression.pdf/@@download/file/1996BanseScherer_Acoustic%20Profiles%20in%20Vocal%20Emotion%20Expression.pdf)
- （◎）Juslin, Laukka & Bänziger 2018（Journal of Nonverbal Behavior 42:1–40）は，23 のデータセットから集めた 1,877 本の声で，自然な表出と演技の表出を 3 つの実験で比べた．自然な表出の方が本物らしいと評定された（0〜4 の尺度で 3.45 対 2.66）．強さをそろえても差は残り，強い表出では差が小さかった — [PMC5816122](https://pmc.ncbi.nlm.nih.gov/articles/PMC5816122/)
- （◎）同論文で，演技の表出は平均して強いと評定された（1.58 対 1.28，d = 0.39）．音響の 13 の手がかりのうち「刺激の種類 × 感情」の交互作用が有意だったのは，弱い強さで 5，中くらいの強さで 3 だった．例：演技の喜びは F0 の幅が広い，自然な怒りは話す速さが速い，演技の怒りはジッタが大きい，自然な悲しみは強さの幅が広い — [PMC5816122](https://pmc.ncbi.nlm.nih.gov/articles/PMC5816122/)
- （◎）同論文の自然な表出だけの判定（9 択）では，最多の回答への一致は強い表出で 72%，中くらいで 58%，弱いと 46% だった（でたらめなら約 11%）．著者は「演技の研究で得た原則が日常の音声の自動分類でうまくいかないのは，この差のためかもしれない」と書いたが，検証はしていない — [PMC5816122](https://pmc.ncbi.nlm.nih.gov/articles/PMC5816122/)
- （△）Wilting, Krahmer & Swerts 2006（Interspeech）は，Velten 法（感情の強い文を読ませる）で一方の群には本当の気分を誘発し，他方の群には演技させた．誘発の群は気分が変わったと答え，演技の群は中立のままだと答えた．聞き手は演技の感情（特に否定の感情）を本物より強いと判断した — [ISCA Archive](https://www.isca-archive.org/interspeech_2006/wilting06_interspeech.html)
- （△）Krahmer & Swerts 2008（Interspeech）は，経験のある俳優の演技は素人の演技より極端で，本物らしくないと評定されたと報告した — [ISCA Archive PDF](https://www.isca-archive.org/interspeech_2008/krahmer08_interspeech.pdf)
- （△）Jürgens, Grass, Drolet & Fischer 2015（Journal of Nonverbal Behavior）は，ラジオの取材から取った本物の感情の発話 80 本と，職業俳優と素人による再演 80 本を比べた．2 種類の再演は同じくらいの正解率で認識され，どちらも本物より F0 の動きが大きかった．職業俳優は素人と違う調音の型を示した — [Springer](https://link.springer.com/article/10.1007/s10919-015-0209-5)，[PMC4519627](https://pmc.ncbi.nlm.nih.gov/articles/PMC4519627)
- （△）本物か演技かは聞き手の認識を変え，怒りは演技のときに，悲しみは本物のときに正しく当てられやすいという予測を検証した行動と fMRI の研究がある — [Cognitive, Affective, & Behavioral Neuroscience](https://link.springer.com/article/10.3758/s13415-011-0069-3)
- （△）Kamiloğlu & Sauter 2020 は，肯定の感情の非言語の発声で，自然な発声と演技の発声のどちらがよく認識されるかは感情によって違ったと報告した — [ISCA Archive LW 2020](https://www.isca-archive.org/lw_2020/kamiloglu20_lw.html)
- （△）自然な対話と演技の対話の感情認識を比べた ACII 2015 の研究では，言いよどみと非言語の発声の特徴は自然な対話で手作りの低次の特徴より良かったが，演技の対話ではそうでなかった（演技の対話ではそれらが少ない）— [IEEE Xplore 7344645](https://ieeexplore.ieee.org/document/7344645/)
- 既出：Schuller 2010（演技・固定文の組どうしはある程度移り，自然発話はほぼ偶然水準），Milner 2019（誘発の IEMOCAP が演技と自然の中間），Feng & Chaspari（演技 → 自然の少数ショット）．どれも演技・誘発・自然を定性的な区分として使い，push/pull の量を式にしていない — `docs/related-work.md` 2.2・2.3 節

### Inferences
- push/pull の区別と Juslin 2018 の結果は，演技と自然の差を 2 つに分ける：(1) 強さの差（演技は強い．同じ方向に伸びる），(2) 型の差（感情ごとに動く手がかりが違う）．MER-DA の量では，(1) はクラスの平均の間隔，つまり dprime（MH5），(2) はソースとターゲットの最適な頭のなす角（MH2 の e）に当たる．E8 の「演技の録音か自然な会話か」の 1 本の軸は，この 2 つの和を見ている可能性がある．
- Juslin 2018 で自然な表出の一致が強さで 46〜72% と大きく動いたことは，自然なコーパスの中にも強さの幅が大きいことを示す．MER-DA で dprime をコーパス全体の 1 つの値で扱うと，この幅を平均してしまう．
- Jürgens 2015 は「素人の再演も職業俳優の再演も本物より F0 の動きが大きい」と報告しており，RAVDESS・CREMA-D（演技）と IEMOCAP 即興（誘発）・MOSEI・MELD（自然か台本の演技）の差は，演技者の経験より「演技かどうか」で決まる見込みが高い．

### Gaps
- Scherer の push/pull の原典（Kappas, Hess & Scherer 1991 など）と Scherer 2003 の該当箇所は，今回は二次資料と講演資料でしか確かめていない．
- Banse & Scherer 1996 の聞き手と判別分析の的中率の数値は取得していない．
- 演技 → 自然の転移で，強さ（dprime）を合わせる操作の効果を測った機械学習の研究は見つからなかった．強さのラベルを持つ RAVDESS・CREMA-D を使った強さ別の転移の研究も見つからなかった（`docs/related-work.md` 4 節の未調査の分野と同じ）．

## 3. 方言の理論と集団内の優位（文化間・言語間），それを感情認識の DA に使った研究

### Takeaway
方言の理論（dialect theory）は，感情の表し方には文化を越えて共通の部分と文化ごとの「なまり」があり，見る側・聞く側は自分の文化のなまりに合わせて読むので，同じ文化どうしで正解率が高くなる（集団内の優位）と説明する．声のメタ分析（Laukka & Elfenbein 2021）は，文化の距離が大きいほど正解率が下がり集団内の優位が大きいことを示した．音響の分類器でも，文化をまたぐと成績が 26〜34% 下がった（Neiberg ほか 2011）．しかし，方言の理論の「共通＋なまり」の分解を DA の学習法として書いた研究は見つからなかった．文化間の感情認識の DA（AVEC 2019 の文化間の課題）は，敵対学習などの一般の手法を使っている．

### Cited Findings
- （△）Elfenbein & Ambady 2002（Psychological Bulletin 128(2):203–235）のメタ分析は，感情は文化を越えて偶然より高く認識されること，同じ集団の表出の方が正しく認識されること（集団内の優位），互いに接する機会の多い集団どうしでは優位が小さいこと，多数派が少数派を判断する方が逆より不正確なことを報告した．模倣の表出を使った研究では文化間の正解率が高く，研究の設計は集団内の優位の大きさを変えなかった．優位の大きさを 9.3% とする二次資料があるが原典で確かめていない．対象の研究数は資料で 87 と 97 に分かれる — [SWOV の書誌と抄録](https://swov.nl/en/publicatie/universality-and-cultural-specificity-emotion-recognition-meta-analysis)
- （△）Matsumoto 2002 は，Elfenbein & Ambady の結論は方法上の基準を満たす研究が少なく支持されないと反論した — [Matsumoto 2002 PDF](https://www.davidmatsumoto.com/content/2002%20Matsumoto.pdf)
- （△）Elfenbein, Beaupré, Lévesque & Hess 2007（「Toward a dialect theory」）は，演技の表情の表出と認識の文化差を扱った方言の理論の論文である（題名を確かめただけ） — [ResearchGate](https://www.researchgate.net/publication/6453687_Toward_a_dialect_theory_Cultural_differences_in_the_expression_and_recognition_of_posed_facial_expressions)
- （△）Laukka & Elfenbein 2021（Emotion Review 13(1):3–11，2020 年 2 月に早期公開）は，声の感情の文化間の研究 37 本（表出者 26 文化，判断者 44 文化）をメタ分析した．文化間でも多くの感情が偶然より高く認識され，同じ文化の方が正解率が高く，Hofstede の次元で測った表出者と判断者の文化の距離は，正解率と負に，集団内の優位と正に相関した — [SAGE](https://journals.sagepub.com/doi/full/10.1177/1754073919897295)，[WashU の書誌](https://profiles.wustl.edu/en/publications/cross-cultural-emotion-recognition-and-in-group-advantage-in-voca/)
- （○）Neiberg, Laukka & Elfenbein 2011（Interspeech，pp. 1581–1584）は，VENEC コーパス（11 感情 × 3 段階の強さ）の音響の量で nu-SVM を学ばせた．同じ文化と別の文化の条件で学ばせたときの再現率は偶然の約 2.4〜3 倍（抄録の表記が乱れていて正確な値は不明）で，文化をまたぐ条件では相対的に，強い表出で 26%，中くらいで 32%，弱い表出で 34% 下がった．同じ文化の模型は，強さが弱いときに条件の食い違いに敏感だった — [ISCA Archive](https://www.isca-archive.org/interspeech_2011/neiberg11b_interspeech.html)
- （△）PLOS ONE 2022「Identifying the evidence of speech emotional dialects using artificial intelligence: A cross-cultural study」は，6 つの文化圏の 6 つの音声データセットで SVM を学ばせ，全体の正解率は同じ集団で一貫して高かったが，感情ごとの再現率では集団内の優位が一部しか支持されなかったと報告した．本文は PMC の閲覧確認の頁で取得できなかった — [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0265199)
- （△）Brooks ほか 2023（Nature Human Behaviour 7(2):240–250）は，中国・インド・南アフリカ・米国・ベネズエラの 4,031 人が 2,756 本の元の発声をまねた声で深層網を学ばせ，24 の感情に関わる音響の次元を見つけた．その意味は 5 か国・3 言語で 79% 保たれた — [IDEAS/RePEc](https://ideas.repec.org/a/nat/nathum/v7y2023i2d10.1038_s41562-022-01489-2.html)
- （△）国の違う顔画像データベースの間で，畳み込み網が表情の動作単位の差を示したという題名の研究と，深層学習で文化ごとの表情の意味を調べた研究がある（どちらも題名だけ確かめた） — [PMC9670153](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9670153/)，[PMC10906517](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10906517/)
- （◎）反対の結果：Soto & Levenson 2009（Emotion 9(6):874–884）は，4 つの民族集団の大学生 161 人が会話の録画の感情価と強さを連続で評定した課題で，共感の正確さに集団内の優位を見つけなかった（評定者 × 対象の民族の交互作用は有意でない）．生理の同調では中国系の評定者だけが中国系の対象と強く同調した（33.1% 対 25.6%）．著者は，実時間の対人の場面では相手を正しく読む必要が優位を上回る可能性と，他集団に接する経験の影響を挙げた — [PMC2877627](https://pmc.ncbi.nlm.nih.gov/articles/PMC2877627)
- （○）反対の結果（機械学習）：Mathur, Adolphs & Matarić（IEEE FG 2023．arXiv 2208.00344）は，6 つの文化の自然な 2 者の会話の映像で，別の文化で学んだ模型が同じ文化で学んだ模型と同等以上だったと報告した．顔の特徴が音声より役に立った．方言の理論と集団内の優位には触れていない — [arXiv 2208.00344](https://arxiv.org/abs/2208.00344)
- （△）AVEC 2019 の文化間の感情の副課題（CES）は，SEWA のデータを広げ，ドイツ語・ハンガリー語の話者で学んだ覚醒・感情価・好みの推定を中国語の話者に移す課題として組まれた．参加論文の 1 本は，文化の間の差を埋めるラベル無しの敵対的な DA を提案した．いずれも方言の理論は使っていない（検索結果の範囲） — [AVEC 2019（arXiv 1907.11510）](https://arxiv.org/pdf/1907.11510)，[敵対的 DA の論文（doi 10.1145/3347320.3357692）](https://unpaywall.org/10.1145%2F3347320.3357692)
- （△）SEWA DB は，録画された人と同じ文化の注釈者を文化ごとに 5 人集めて注釈の偏りを減らした — [SEWA DB（arXiv 1901.02839）](https://arxiv.org/pdf/1901.02839)
- （△）Cahyawijaya ほか 2023（Interspeech）は，高齢者の音声が少ないとき，その集団だけで学んだ模型は成績が悪く，言語と年齢の集団をまたいで推論・データ増強する方法を提案した — [ISCA Archive PDF](https://www.isca-archive.org/interspeech_2023/cahyawijaya23_interspeech.pdf)
- （△）中国語・英語・紹興方言・チベット語の声で，青年に集団内の優位を見つけた 2024 年の研究がある — [Frontiers in Psychology 2024](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2024.1437701/full)
- （△）刺激の実際の出所は感情の認識に影響したが，聞き手が信じた出所は影響しなかったという研究があり，集団内の優位が動機ではなく手がかりの文化差から来ることを支持する（検索の要約による．著者と年は未確認） — [PMC3812869](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3812869/)

### Inferences
- 方言の理論は，ある集団 g の感情 y の手がかりの平均を「共通の型 μ_y ＋ 集団のなまり δ_{g,y}」と書くことに当たる．線形の頭で書けば w_g = w_0 + v_g で，v_g を小さく保つ罰則をかけた学習は Daumé 2007 の特徴の拡張（共通の重みとドメインごとの重みを同時に学ぶ）と同じ形である．Daumé 2007 は `docs/related-work.md` 2.6 節に既出だが，方言の理論との対応を書いた論文は見つからなかった．
- Laukka & Elfenbein 2021 の「文化の距離が大きいほど集団内の優位が大きい」は，ソースとターゲットの距離から転移の損を予言する式（v_t − v_s の大きさが距離に比例する）として読める．MER-DA の 7 群はほぼ英語・北米で文化の差が小さいので，「文化」を「産出の場（演技の録音か会話か）」に読み替えると，E8 の軸が Hofstede の距離の代わりになる．
- Neiberg 2011 で文化をまたぐ損が弱い表出ほど大きかった（26% → 34%）ことは，MH5（dprime が小さいターゲットほど組の相性が強く効く）と同じ向きである．方言の差は，信号の弱いところで目立つ．
- Soto & Levenson 2009 と Mathur 2023 は，自然な対話の素材では集団内の優位が消えうることを示す．方言の理論の予言は，演技の素材（RAVDESS・CREMA-D）で強く，会話の素材（IEMOCAP・MELD・MOSEI）で弱い可能性がある．

### Gaps
- Elfenbein 2013（Emotion Review の方言の理論の論文）は，今回の検索結果に出ず，確かめていない．
- Dailey ほか 2010（Emotion 10(6)．日米の表情の認識の差を，文化ごとの経験で学ぶ神経網の模型で説明したとされる）は，書誌が資料で食い違い，本文も抄録も取得できなかった．方言の理論を学習の模型にした最も近い例である可能性があり，原典で確かめる必要がある．
- PLOS ONE 2022 の 6 データセットの研究は，文化ごとにデータセットが 1 つずつなので，文化の差とコーパスの差が分けられていない可能性がある（本文未読のため未確認）．
- 方言の理論を明示的に使った文化間の SER・FER・MER の DA は見つからなかった．

## 4. 表示規則，構成主義的な情動理論，文脈がラベルを変える知見

### Takeaway
表示規則（文化や場面が表出を強めたり抑えたりする決まり），構成主義的な情動理論（Barrett．感情は手がかりと文脈から脳が組み立てる判断であり，表情と感情は 1 対 1 に対応しない），文脈の研究（Aviezer．体や場面が顔の感情の読みを変える）は，どれも「同じ顔や声でも，文脈しだいで付くラベルが変わる」と予言する．機械学習では文脈を入れた認識や，LLM で理論を実装する試みがあるが，コーパス間の DA の学習法として式にした研究は見つからなかった．

### Cited Findings
- （△）Aviezer, Trope & Todorov 2012（Science 338:1225–1229）は，強い肯定・否定の場面（テニスの勝ち負けなど）で，体だけからは肯定と否定を見分けられたが顔だけからは見分けられなかったと報告した．同じ顔を肯定の体と否定の体に付け替えると，顔の感情価の読みが体に合わせて反転した — [Princeton の書誌](https://collaborate.princeton.edu/en/publications/body-cues-not-facial-expressions-discriminate-between-intense-pos/)
- （△）Le Mau ほか 2021（Nature Communications．Barrett が共著）は，文脈の豊かな筋書きに応じた職業俳優の表情の写真で，同じ感情でも顔の形がばらつき，恐れ・喜び・驚きだけがある程度の信頼性と特異性で表されたと報告した．観察者の判断は文脈に依存した — [PMC8376986](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8376986/)
- （△）Srinivasan & Martinez（IEEE TAC．doi 10.1109/TAFFC.2018.2887267．arXiv 1808.04399）は，31 か国の自然な写真で，文化をまたぐ表情は感情の種類と感情価が一貫して読まれるが覚醒は一貫せず，文化に固有の表情は感情価と覚醒が一貫して読まれるが感情の種類は一貫しないと報告した．表情の数は感情で大きく違い，喜びは 17，嫌悪は 1 だった — [arXiv 1808.04399](https://arxiv.org/abs/1808.04399)
- （△）ICMI 2022 の付随論文「Contextual modulation of affect: Comparing humans and deep neural networks」は，文脈を減らした版では人の覚醒・感情価の評定が系統的に下がったが，深層網は文脈がある場合も無い場合も人の評定の型を再現できなかったと報告した — [ACM DL](https://dl.acm.org/doi/10.1145/3536220.3558036)
- （△）構成主義的な情動理論を実装した例として，LLM で利用者の文脈をまとめる CHI 2025 拡張要旨がある．評価は面接 5 人の事例研究である — [ACM DL](https://dl.acm.org/doi/full/10.1145/3706599.3721205)
- （△）顔と視覚の文脈の相互作用のメタ分析（Psychonomic Bulletin & Review 2025）と，文脈の豊かな動く場面での感情の手がかりの統合が個人で違うという研究（Nature Communications 2025）がある（題名だけ確かめた） — [Springer](https://link.springer.com/article/10.3758/s13423-025-02678-6)，[Nature Communications](https://www.nature.com/articles/s41467-025-67466-1)
- 既出：Larsen ほか 2026 は，範疇の感情ラベルが文脈に依存する状態を 1 つのラベルに縮めていると論じた — `docs/related-work.md` 2.4 節，[arXiv 2603.23017](https://arxiv.org/html/2603.23017v2)

### Inferences
- 3 つの理論は，DA の語ではラベルの付け方のずれ（p(y|x) がコーパスで違うこと．Ben-David 2010 の λ に入る部分）として現れる．文脈の変数 c が特徴に入っていないと，p(y|x) = Σ_c p(y|x,c) p(c|コーパス) がコーパスの文脈の分布 p(c|コーパス) で変わる．MER-DA の発話平均の特徴には対話の前後も場面も入らないので，この部分は少数のターゲットのラベルでも特徴の変換でも取り除けない．
- MER-DA の群の文脈は，対話（IEMOCAP・MELD），独白（MOSEI），映画の場面（AFEW），文脈の無い固定文（RAVDESS・CREMA-D）と分かれる．構成主義の予言では，固定文の演技のコーパスは「文脈が無い分，表出の型が誇張された」データであり，Le Mau 2021 の「文脈を与えると俳優の表情がばらつく」と合わせると，文脈の豊かなコーパスほどクラス内のばらつきが大きく dprime が小さいと予言できる．これは MH5 の dprime の差の由来の 1 つの説明になる（推論で，検証はしていない）．
- Srinivasan & Martinez の「文化をまたいで覚醒が一貫しない」は顔についての結果で，Bänziger 2015 の「声は覚醒をよく伝える」とは逆の向きである．モダリティによって，コーパスをまたいで保たれる軸が違う可能性がある（6 節）．

### Gaps
- Ekman & Friesen の表示規則の原典，Barrett の theory of constructed emotion の原典（2017 など），Barrett ほか 2019（Psychological Science in the Public Interest），Aviezer ほか 2008（Psychological Science）は，今回の検索で本文も抄録も取得していない．
- 文脈を考慮した感情認識のデータセット（EMOTIC，CAER など）と，文脈のずれを DA として扱った研究は，今回は探していない．
- 表示規則を，文化やコーパスごとの「表出の強さの縮め方」として分類器に入れた研究は見つからなかった．

## 5. 知覚された感情・意図した感情・感じた感情の区別と，評定者の主観性（一致，曖昧さ，soft label）

### Takeaway
意図した感情（演技の指示），知覚された感情（他人の評定），感じた感情（本人の自己報告）は別の量で，本人と他人の一致は他人どうしの一致より低い．2026 年の研究（El-Tawil, Sampath & Mower Provost）は，他人のラベルで学んだ模型が本人の自己報告をほとんど当てられないことを示した（覚醒の CCC はほぼ 0）．評定者の主観性は，soft label・曖昧さによる学習の順序・注釈者ごとの頭として学習法になっている（定式化済み）．ただし，意図と知覚の差をコーパス間の DA の学習法にした研究は，査読前の 1 本（第二言語の SER）しか見つからなかった．

### Cited Findings
- （◎）El-Tawil, Sampath & Mower Provost 2026（arXiv 2601.21130．掲載先は本文に無い）は，他人のラベルだけを持つ MSP-Podcast v1.11（151,654 区間，2,172 話者）で学んだ模型で，自己報告を持つ MuSE（独白 273 本）と IEMOCAP（自己評価のある 2,411 区間）を評価した — [arXiv 2601.21130](https://arxiv.org/html/2601.21130)
- （◎）同論文で，自己報告と他人のラベルの一致（CCC）は，MuSE で覚醒 −0.039，感情価 0.299（他人のラベルを平均すると覚醒 −0.179，感情価 0.350），IEMOCAP（演技）で覚醒 0.561，感情価 0.781 だった．MSP-Podcast で学んだ模型の MuSE の自己報告への成績は，覚醒 0.03〜0.08，感情価 0.26〜0.32 だった．IEMOCAP の自己報告で追加学習しても確かな改善は無く，WavLM ではコーパス間の汎化が悪くなった．適応やラベルの写像の手法は提案していない — [arXiv 2601.21130](https://arxiv.org/html/2601.21130)
- （◎）同論文で，内容の個人的な重要さ（LLM で 1〜5 に採点．10% だけ人が確かめた）が最も強い調整変数で，重要さが最高の区分では感情価の CCC が 0.6〜0.8 に上がり，低い区分では MuSE でほぼ 0 だった — [arXiv 2601.21130](https://arxiv.org/html/2601.21130)
- （△）Zhang, Essl & Mower Provost 2016（ICMI）は，IEMOCAP で自己報告と知覚のラベルを深層信念網の多課題学習で同時に学ばせた．知覚のラベルの課題は曖昧な発話で，自己報告の課題は明確な発話で最も得をした — [ICMI 2016 PDF](https://emp.engin.umich.edu/wp-content/uploads/sites/67/2022/06/2016_Zhang_ICMI.pdf)
- （△）arXiv 2510.03060（2025）は，刺激の意図した感情に結びつく「描写の発話」と，本人の自己報告の感情に結びつく「表出の発話」を分けた．参加者は 96.63% で意図した感情を報告したが，20% 以上で最も強く感じた感情は別だった — [arXiv 2510.03060](https://arxiv.org/html/2510.03060v1)
- （△）第二言語の学習者の SER の査読前の論文（Research Square）は，英語を母語とする評定者が意図と知覚の両方の感情を付けたコーパスで，母語の偏りをドメイン敵対学習で取り除く枠組みを提案した — [Research Square rs-7347456](https://www.researchsquare.com/article/rs-7347456/v1.pdf)
- （△）Han, Zhang, Schmitt, Pantic & Schuller 2017（ACM MM，pp. 890–897）は，注釈者の不一致から知覚の不確かさを作り，感情の状態と不確かさを多課題学習で同時に予測した（コーパス内） — [Augsburg 大学の書誌](https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/index/index/docId/45069)
- （△）Lotfian & Busso（arXiv 1805.10339．IEEE の雑誌の体裁．掲載誌と年は未確認）は，人にとって曖昧な標本は計算機にとっても曖昧だとして，評定者の不一致で難しさを決め，易しい標本から学ばせる順序（カリキュラム学習）を作った（コーパス内） — [arXiv 1805.10339](https://arxiv.org/abs/1805.10339)
- （△）Interspeech 2026 の論文（arXiv 2606.27536）は，MSP-Podcast 2.0 で，多数決のラベルの代わりに票の分布で教え，分布のエントロピーで学習の順序を決めた — [arXiv 2606.27536](https://arxiv.org/html/2606.27536)
- 既出（定式化済み）：Tavernor & Mower Provost 2025（ASRU）は，MSP-Podcast の注釈者 1,998 人ぶんの頭から，新しいコーパスの新しい注釈者に最も似た頭を少数ラベル（注釈者あたり 5〜30 件）で選ぶ．Li ほか 2021 は，ソースのクラス間の似かたを soft label でターゲットに移した．Larsen ほか 2026 は IEMOCAP で注釈者全員の一致が約 20% と報告した — `docs/related-work.md` 2.2・2.4 節

### Inferences
- MER-DA の群のラベルは，RAVDESS・CREMA-D が意図した感情（演技の指示），IEMOCAP・MELD・MOSEI・AFEW が知覚された感情（注釈者）で，感じた感情の群は無い．El-Tawil 2026 は，演技の IEMOCAP では本人と他人の一致が高い（感情価 0.781）が，自然な独白の MuSE では低い（0.299）ことを示す．演技のコーパスでは意図と知覚がほぼ一致し，自然なコーパスでは離れるので，「意図のラベルのソース → 知覚のラベルのターゲット」のずれは自然なターゲットほど大きいと予言できる．これは E8 の軸と同じ向きである．
- 評定者の主観性は，観測ラベルを「真の状態をコーパスごとの写像 T_g（4×4 の確率行列）で写したもの」と書けば，ラベルの雑音の補正（遷移行列の推定）の形の DA になる．ターゲットの T_t の自由度は 4×3 = 12 で，K 本のラベルから推定できる大きさである．

### Gaps
- Mower ほか 2009（ACII．曖昧な感情表出の解釈），Truong ほか 2012（自己報告と観察者の感情の認識），Chou & Lee の注釈者を扱う一連の研究は，今回は確かめていない．
- MER-DA の 7 群それぞれで，発話ごとの評定者の票や自己評価がどこまで公開されているか（CREMA-D の知覚の票，RAVDESS の評定，IEMOCAP の自己評価，MELD・MOSEI・AFEW の注釈者の数）は，今回は文献で確かめていない．`data/README.md` と各コーパスの原典で確かめる必要がある．

## 6. 次元（覚醒・快不快）と範疇の対応を，コーパス間のラベルの写像や DA に使った研究

### Takeaway
覚醒と快不快の 2 軸に範疇を置く写像（Russell の円環模型の考え方）は，ラベルの体系が違うコーパスを合わせるための手で決めた対応表として使われてきた（Schuller 2010，Zhang ほか 2011）．DA の学習法の中で，軸ごとに転移の強さを変える使い方は見つからなかった．感情科学の実証からは，声は覚醒をよく伝え快不快を伝えにくい（Bänziger 2015），顔は文化をまたぐと覚醒の読みが一貫しない（Srinivasan & Martinez），自己報告では覚醒がほとんど当たらない（El-Tawil 2026），という軸ごとの違いが出ている．

### Cited Findings
- 既出：Schuller ほか 2010（IEEE TAC）は，6 コーパスの感情のラベルを覚醒・感情価の 2 値に手で写して合わせた — `docs/related-work.md` 2.3 節
- （△）Zhang, Weninger, Wöllmer & Schuller 2011（ASRU）は，複数のコーパスの範疇のラベルを覚醒と感情価の 2 値に写し，ラベル無しのデータを使う学習が教師ありの学習に近いか上回ることがあると報告した．平均で最良だったのは全コーパスのラベルを使う場合だった — [TUM の PDF](https://mediatum.ub.tum.de/doc/1107343/687488.pdf)
- （△）次元のコーパスで学び範疇のコーパスで試すときに，覚醒・感情価の平面に中立の領域を置く 3 クラスの写像を提案した研究がある — [ResearchGate](https://www.researchgate.net/publication/286681269_Location_of_an_emotionally_neutral_region_in_valence-arousal_space_Two-class_vs_three-class_cross_corpora_emotion_recognition_evaluations)
- （◎）Bänziger 2015 では，聞き手の覚醒の判断は近位の知覚で 91% 説明され，表した覚醒からの効果の 84% が手がかりを通っていた．喜びは 27% しか説明されなかった — [PMC4556609](https://pmc.ncbi.nlm.nih.gov/articles/PMC4556609/)
- （△）Srinivasan & Martinez は，文化をまたぐ表情では感情価が一貫して読まれ覚醒は一貫しないと報告した — [arXiv 1808.04399](https://arxiv.org/abs/1808.04399)
- （◎）El-Tawil 2026 で，他人のラベルで学んだ模型の自己報告への成績は，覚醒でほぼ 0，感情価で約 0.3 だった — [arXiv 2601.21130](https://arxiv.org/html/2601.21130)
- （△）Brooks ほか 2023 の発声の研究の前身（Cowen ほか 2019）は，発声の認識を整理するのは覚醒・感情価のような評価の次元より感情の範疇だと報告した — [IDEAS/RePEc（Brooks 2023）](https://ideas.repec.org/a/nat/nathum/v7y2023i2d10.1038_s41562-022-01489-2.html)，[WashU の書誌（Cowen 2019）](https://profiles.wustl.edu/en/publications/mapping-24-emotions-conveyed-by-brief-human-vocalization/)

### Inferences
- MER-DA の 4 感情は 2 軸に置ける：覚醒は Angry・Happy が高く Sad・Neutral が低い，快不快は Happy が高く Angry・Sad が低い．4 クラスのロジスティック回帰の頭は，この 2 軸の対比と残り（Neutral の位置など）に分けられる．
- 感情科学の結果からは，音声では覚醒の軸（Angry 対 Sad）がコーパスをまたいで保たれやすく，快不快の軸（Happy 対 Angry）は保たれにくいと予言できる．顔はその逆の可能性がある．これは MH5 が音声で成り立ち映像（V）で成り立たないこと（`docs/related-work.md` 1.2 節の表）と関係する可能性がある（推論で，検証はしていない）．

### Gaps
- Russell 1980（Journal of Personality and Social Psychology の円環模型）の原典は，今回は確かめていない．
- 範疇のラベルを次元の空間に置いて，その空間の上で DA（軸ごとの重みづけ，軸ごとの写像の学習）をした研究は見つからなかった．文章や画像では範疇から VAD への回帰の写像があるが（検索結果に出た画像の感情の研究．arXiv 2602.01173），音声・顔・マルチモーダルの DA では見つからなかった．

## 7. SER・FER・MER の DA で，これらの理論を明示的に式にした研究（2020〜2026 年を優先）

### Takeaway
2020〜2026 年の範囲で，レンズ模型，push/pull，方言の理論，表示規則，構成主義を，SER・FER・MER の DA の学習法として明示的に式にした研究は見つからなかった．定式化済みと判定できたのは，評定者の主観性を扱うもの（Tavernor & Mower Provost 2025 の注釈者の頭の選択，Li ほか 2021 の soft label，コーパス内の Han 2017・Lotfian & Busso のカリキュラム）と，意図と知覚の両方のラベルを使って敵対学習をする査読前の 1 本だけである．文化間の DA（AVEC 2019 CES）は一般の敵対学習で，方言の理論を式に入れていない．

### Cited Findings
- 定式化済み（評定者の主観性）：Tavernor & Mower Provost 2025（注釈者の頭を少数ラベルで選ぶ．覚醒・感情価の回帰）— `docs/related-work.md` 2.2 節，[arXiv 2509.12295](https://arxiv.org/pdf/2509.12295)
- 定式化済み（soft label による転移）：Li ほか 2021（演技 → 即興．ソースのクラス間の似かたをターゲットに移す）— `docs/related-work.md` 2.2 節
- 定式化済み（コーパス内の曖昧さ）：Han ほか 2017（知覚の不確かさの多課題学習）— [Augsburg](https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/index/index/docId/45069)；Lotfian & Busso（曖昧さで学習の順序を決める）— [arXiv 1805.10339](https://arxiv.org/abs/1805.10339)
- 定式化済み（意図と知覚．査読前）：第二言語の SER で，意図と知覚のラベルを持つコーパスでドメイン敵対学習 — [Research Square](https://www.researchsquare.com/article/rs-7347456/v1.pdf)
- 理論を使わない文化間の DA：AVEC 2019 CES の敵対的 DA — [arXiv 1907.11510](https://arxiv.org/pdf/1907.11510)
- 理論を検証する機械学習（DA ではない）：Neiberg 2011（文化をまたぐ損 26〜34%）— [ISCA](https://www.isca-archive.org/interspeech_2011/neiberg11b_interspeech.html)；PLOS ONE 2022（SVM で集団内の優位）— [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0265199)；Brooks 2023（意味の 79% が保たれる）— [RePEc](https://ideas.repec.org/a/nat/nathum/v7y2023i2d10.1038_s41562-022-01489-2.html)；Mathur 2023（別の文化で学んでも同等以上）— [arXiv 2208.00344](https://arxiv.org/abs/2208.00344)
- 評価の研究（手法なし）：El-Tawil 2026（他人のラベル → 自己報告）— [arXiv 2601.21130](https://arxiv.org/html/2601.21130)

### Inferences
- 6 つの理論のうち 4 つ（レンズ模型，push/pull，方言の理論，表示規則と構成主義）は，DA の学習法として未定式化である．残り 2 つ（評定者の主観性，次元と範疇）は，前者が学習法として定式化済み，後者がラベルを合わせる対応表としてだけ使われている．
- MER-DA の既存の量との対応は次のとおりである（推論）．レンズ模型の一致度 G ↔ MH2 の向きの効率 e，生態学的妥当性 ↔ MH5 の dprime，push/pull の強さ ↔ dprime，push/pull の型 ↔ 頭の向き，方言の「共通＋なまり」↔ 仮説転移の罰則の中心（`docs/related-work.md` 1.8.2 節）と Daumé 2007 の特徴の拡張，表示規則と文脈 ↔ Ben-David の λ のうち特徴で取り除けない部分，評定者の写像 ↔ ラベルの雑音の遷移行列．
- MER-DA の新しさに使うなら，「これらの理論が予言する量を，凍結した特徴と線形の頭の量として測った」ことが最も近い穴である．どの理論も感情科学の側では演技の少数のコーパスか人の判断で検証されており，7 群 × 3 モダリティの機械学習の地図で検証した研究は見つからなかった．

### Gaps
- IEEE TAC・Interspeech・ICASSP・ACII・ACM MM の 2020〜2026 年の目次を直接は当たっていない．検索エンジンの結果だけで判定した．
- Dailey ほか 2010 の神経網の模型（方言の理論に最も近い計算の模型の可能性）の中身は確かめられなかった．
- 2026 年の「データセットの違いを測る」単一モダリティの論文群（`docs/related-work.md` 5.1 節）が，これらの理論を引いているかは確かめていない．

### MER-DA の設定で学習法にするなら（候補ごと）

前提：4 感情，UAR，7 群，凍結したエンコーダの発話平均の特徴，多クラスのロジスティック回帰の頭，ターゲットのラベルはクラスあたり K 本（K = 0〜50），話者を分ける．

- **A. レンズ模型（表出側の頭と知覚側の頭の分解）**：意図のラベルの群（RAVDESS・CREMA-D）の頭を表出側，知覚のラベルの群（IEMOCAP・MELD・MOSEI・AFEW）の頭を知覚側とみて，ソースの頭を「表出側の頭＋知覚側との差の方向」に分け，ターゲットのラベルの種類に合う方を仮説転移の出発点にする．要るもの：同じ発話に意図と知覚の両方のラベル（CREMA-D の知覚の票，RAVDESS の評定，IEMOCAP の自己評価が候補．未確認），一致度 G を頭のなす角として測る定義（MH2 の e と同じか確かめる）．
- **B. push/pull（強さの差と型の差の分解）**：ソースのクラス平均を全体の平均へ s = d′_t / d′_s 倍に縮めてから合併する（強さを合わせる）．s は K 本で推定したターゲットの dprime から決め，残る損を頭の向きの差（型の差）として別に測る．要るもの：RAVDESS・CREMA-D の強さのラベルで「強さ別の dprime」と「弱い表出だけのソース」の転移を検算すること，強さのラベルの無い群での強さの代理量．
- **C. 方言の理論（共通の型＋群のなまり）**：頭を w_g = w_0 + v_g と分けて全ソースで同時に学び（Daumé 2007 の形），ターゲットでは v_t に K に応じた罰則をかける．E8 の軸で近いソースの v_g を v_t の初期値にする．要るもの：群の間の「方言の距離」の定義（文化の差が小さいので，E8 の軸か dprime の差で代える），CREMA-D の演技者の人種の情報で群の中の集団内の優位を検算すること．
- **D. 表示規則・構成主義・文脈（文脈によるラベルの付け方のずれ）**：最小の実装は，モダリティ（顔・声・言葉）の重みを群の文脈（対話・独白・映画・固定文）ごとに変える頭である．対話の前後を使える IEMOCAP・MELD では前の発話の特徴を入れる．要るもの：文脈の変数の定義と，文脈を足すと λ（ソースとターゲットの両方で最良の頭の誤り）が下がるかの検算．RAVDESS・CREMA-D は文脈が無いのでソースにだけ使う．
- **E. 評定者の主観性（ラベルの写像 T_g）**：観測ラベルを真の状態の 4×4 の写像 T_g で書き，ソースの頭を T_s で補正して学び，ターゲットの T_t（自由度 12）を K 本から推定する．要るもの：発話ごとの票か一致度（IEMOCAP・CREMA-D は候補．MELD・MOSEI・AFEW は要確認）．注釈者単位の版は Tavernor & Mower Provost 2025 で定式化済みなので，新しさは「群単位の写像」と「4 クラスの UAR」に置く必要がある．
- **F. 次元と範疇（軸ごとの転移の強さ）**：頭を覚醒の対比（Angry・Happy 対 Sad・Neutral）と快不快の対比（Happy 対 Angry・Sad）と残りに分け，軸ごとに罰則の重みか混ぜる割合 α を変える．モダリティで保たれる軸が違う（声は覚醒，顔は快不快）という予言を，既存の混同行列で損が軸ごとに非対称かを見て先に確かめる．要るもの：対比の基底の定義と，その基底での損の分解．

### 探した検索語と場所

場所：ウェブ検索（英語，標準と拡張），PubMed Central（PMC），arXiv，ISCA Archive の抄録頁．PMC8929654（PLOS ONE 2022）は閲覧確認の頁で本文を取得できなかった．

検索語：
- `Brunswik lens model speech emotion recognition cross-corpus machine learning`
- `"lens model" Brunswik "emotion recognition" automatic affective computing encoder decoder perception model`
- `"distal cues" "proximal" speech emotion recognition perceptual features listener ratings intermediate representation machine learning`
- `lens model analysis vocal emotion five nations acoustic features JPSP 2016 Laukka`
- `acted versus spontaneous emotion push pull effects Scherer recognition model transfer`
- `Juslin Laukka Bänziger 2018 mirror to our soul spontaneous posed vocal expression`
- `Juslin Laukka 2003 Psychological Bulletin communication of emotions vocal expression music performance different channels same code meta-analysis decoding accuracy`
- `Banse Scherer 1996 acoustic profiles vocal emotion expression 14 emotions actors judges accuracy discriminant analysis`
- `Wilting Krahmer Swerts 2006 real vs acted emotional speech Interspeech Velten mood induction`
- `Jürgens Grass Drolet Fischer 2015 acting experience emotion expression recognition voice non-actors Journal of Nonverbal Behavior`
- `dialect theory in-group advantage cross-cultural speech emotion recognition domain adaptation`
- `Laukka Neiberg Elfenbein 2014 cultural dialects vocal emotion acoustic classification within and across five nations`
- `Elfenbein Ambady 2002 meta-analysis universality cultural specificity emotion recognition in-group advantage percent studies`
- `Laukka Elfenbein 2021 Emotion Review vocal in-group advantage meta-analysis Hofstede distance accuracy results`
- `Dailey Cottrell 2010 evidence and computational explanation cultural differences facial expression recognition neural network Japanese American`
- `Brooks Cowen deep learning vocal bursts cultures Nature Human Behaviour 2023 emotion dimensions preserved`
- `cross-cultural speech emotion recognition culture-aware adaptation 2024 2025 in-group annotator culture`
- `AVEC 2019 cross-cultural emotion sub-challenge SEWA adversarial domain adaptation German Hungarian Chinese results`
- `annotator cultural background emotion perception labels speech emotion dataset raters from different countries recognition model bias`
- `Srinivasan Martinez cross-cultural culture-specific production perception facial expressions in the wild IEEE Transactions on Affective Computing`
- `Le Mau 2021 Nature Communications professional actors variability not stereotypical expressions`
- `Aviezer Trope Todorov 2012 Science body cues not facial expressions discriminate intense positive negative emotions`
- `theory of constructed emotion machine learning emotion recognition context Barrett implications affective computing Aviezer context`
- `speech emotion recognition intended versus perceived emotion speaker self-report observer labels gap model 2024`
- `soft labels annotator disagreement cross-corpus speech emotion recognition generalization ambiguity transfer`
- `Lotfian Busso curriculum learning speech emotion recognition crowdsourced labels ambiguity TASLP 2019`
- `Han Zhang Schmitt Pantic Schuller 2017 from hard to soft perception uncertainty emotion recognition ACM Multimedia`
- `cross-corpus emotion recognition label mapping categorical to valence arousal unified label space different taxonomies`

本文を取得したページ：PMC4556609（Bänziger 2015），PMC5816122（Juslin 2018），PMC2877627（Soto & Levenson 2009），arXiv 2601.21130（El-Tawil 2026）．抄録の頁を取得したもの：arXiv 1808.10095（MES-P），arXiv 2208.00344（Mathur），ISCA の Neiberg 2011．Srinivasan & Martinez（arXiv 1808.04399）は検索結果の抄録だけを見た．
