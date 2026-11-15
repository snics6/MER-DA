# 査読報告（感情計算の査読者）

対象：「What Few Target Labels Fix First: Threshold-versus-Direction Analysis and Cross-Modal Weak-Label Transfer for Cross-Corpus Multimodal Emotion Recognition」（本文 12 頁，補足資料 6 頁）．
読んだもの：`main.pdf`，`supplementary.pdf`，`main.tex`，`sections/*.tex`，`supplementary.tex`，`references.bib`．数値はすべて PDF から読んだ．

## 1. 論文の要約

凍結した音声（WavLM），顔（AffectNet で表情認識に学習した ResNet-50），言語（RoBERTa）の特徴の上に線形の 4 クラス分類器を置き，7 群（6 コーパス）の全順序対で，ターゲットのラベルがクラスあたり K 件のときにソースを混ぜた得を測る（Part I）．
各 2 クラスの決定を「閾値のずれ o」と「方向による分離 d」に分け，単一モダリティの音声と顔では K ≤ 5 のラベルが足す改善の 0.54〜0.83 が閾値の修正から来ること，音声の方向は実効次元 q = 16.1〜22.2 に応じて q/K の速さでしか良くならないこと，ソース 1 つの方向がターゲットのラベル 8.5〜11.8 件ぶんに当たることを述べる．
Part II では，ソースだけで学習した融合分類器などを教師にしてターゲットの全ラベル無し発話に重みつきの弱いラベルを付け，ソースから作った「開始点」に固定した線形の生徒を学習する手法を提案する．
同じ疑似ターゲットの手続きで調整した 9 種の半教師あり学習・蒸留の手法と比べ，基本版は 22 セル中 19，拡張版は 11 の生徒すべてで最強の競合を平均 0.8〜3.0 UAR ポイント上回ると報告する．

## 2. 推薦

**Major Revision．**
Part I の測定（全順序対の得の地図，閾値と方向への分解，ソースの価値をラベル数に換算する式）は TAC の読者にとって新しく，測定の手続きも丁寧である．
一方，Part II の手法の優位という主張には次の問題がある．ターゲットは 4〜6 で，そのうち 2 つが IEMOCAP である．統計の支持は限られる（同じテストの折りを共有する反復を平均すると，K=0 で 11 の生徒のうち 3 つしか有意でない）．比べた相手は同じ線形の生徒を学習し直す半教師あり学習と蒸留の手法に限られ，コーパスをまたぐ感情認識で標準のドメイン適応の手法が入っていない．拡張版だけが強い教師を受け取っている．凍結した特徴という設定が実用上代表的であることの裏づけ（微調整との比較，層の選び方）も弱い．
これらは追加の実験と記述の弱化で直せる範囲であり，Part I の価値を考えると Reject には当たらない．

## 3. 強み

- **S1．測定の設計が対になっている．**全手法が同じ反復の同じテスト話者・同じラベルの抽選を共有し（3 節冒頭，sections/3-setting.tex:4），K をまたいでラベルの抽選が入れ子になっている（3.3 節，同:62）．差を対で比べられる設計であり，コーパスをまたぐ感情認識の論文ではこの水準の対の比較は多くない．
- **S2．得の地図の範囲が広い．**3 モダリティとその融合 8 構成，7 群の全順序対，K = 1〜全ラベルにわたって，ソースを混ぜたときの得を 30 反復で測っている（4.1 節，Fig. 2）．顔を含むコーパスをまたぐ地図は，音声だけの先行研究（Schuller et al., 2010；Gideon et al., 2021）にない．
- **S3．閾値と方向への分解が実務の手順に結びついている．**閾値の損失の多くはコーパスの平均のずれであり，ラベル無しの中心化で 4.9 ポイント（WavLM）上がり，中心化の後は切片の再推定と混ぜる操作の得が 0.3 と 0.6 ポイントに縮む（4.3 節，Fig. S2）．「まず中心化する」という助言（7 節）が測定で裏づけられている．
- **S4．数式の予言を実データで照合している．**式 (4) の λ と測定値の順位相関 0.84〜0.91（4.4 節，Fig. S1），式 (5) の n と測定値の順位相関 0.77〜0.90（4.5 節，Fig. S3）を示し，平均の過大評価も隠さず書いている．
- **S5．競合の扱いが公平寄りである．**競合も同じ疑似ターゲットで K ごとに調整し（6.1 節），表 3 の「最強の競合」はテストの平均で事後に選んだもので，競合に有利に働く．原論文の値のままのエントロピー最小化が，融合の生徒で構成とターゲットによって大きく変わること（同じ IEMOCAP の即興がターゲットでも AVT では +4.6，AV では −28.3 ポイント．6.3 節，Fig. S8）も示している．
- **S6．限界の書き方が正直である．**MOSEI の話者分離が不完全であること，ターゲットが 4〜6 で検定が他のコーパスへの一般化を表さないこと，疑似ターゲットでの選択で失う量（6.4 節，8 節）を本文で認めている．

## 4. 指摘

### 4.1 Major

### [Major-1] 手法の優位を「11 の生徒すべてで上回る」と書いているが，ターゲットの数，検定の結果，手法を作った手順のいずれも，この書き方を支えていない
- 該当箇所: main.tex:35（要旨）／ sections/1-introduction.tex:30（貢献 4）／ sections/6-comparison.tex:12, 17, 23–25 ／ sections/9-conclusion.tex:6 ／ sections/5-part2-mechanism-to-method.tex:84, 91 ／ 補足 Table S3
- 現状: 要旨は "an extended version beats them for all eleven students, second encoders included, by 0.8 to 3.0 UAR points" と書き，結論も "exceeds the strongest competitor tuned under the same protocol for every student at K=0 and K=5" と書く．6.2 節は，Wilcoxon 検定で有意な生徒が K=0 で 8，K=5 で 7 であり，同じテストの折りを共有する反復を平均すると 3 と 7 に減ると書く（6-comparison.tex:23–24）．同じ節で "these tests describe the variation within our corpora, not generalization to other corpora" とも書く（同:25）．
- 問題:
  1. **ターゲットが少なく，偏っている．**AVT の 4 つの生徒（A，V，T，融合）は 4 ターゲットで比べられ（6-comparison.tex:12），その 4 つのうち 2 つは IEMOCAP の即興と台本である．したがって比較の中身は 3 コーパスである．AVT の構成を AVT の 4 群だけで作っているなら（Fig. 6 と Fig. S6 の "AVT world" はそう読める．本文には明記がない），IEMOCAP をターゲットにするとき，ターゲットのコーパスの外にあるソースは CMU-MOSEI と MELD の 2 群だけになる．3.6 節の疑似ターゲットの手続き（3-setting.tex:94）では，さらにそのうち 1 群を疑似ターゲットにするので，選択に使う手法は残りの 1 群だけから作られる．この条件で選んだハイパーパラメータが他のコーパスでも働くかは分からない．
  2. **ターゲットごとに見ると「すべてで上回る」とは言えない．**Table S3 の "Pos." の列では，AVT の T の生徒は K=0 と K=5 のどちらでも 4 ターゲットのうち 2 つでしか最強の競合を上回らない．AVT の V の生徒（K=0, K=5），AVT の融合の生徒（K=0），第 2 エンコーダの V と T の生徒（K=0, K=5）と融合の生徒（K=0）も 4 つのうち 3 つであり，AV の融合の生徒は K=5 で 6 つのうち 4 つである．7 節は 92 のターゲットとセルの組のうち 10 で ours-ext が開始点さえ下回ると書く（7-discussion.tex:12）．要旨の "for all eleven students" は，ターゲットの平均をとった値についてだけ成り立つ．
  3. **検定の単位が一般化の単位と合っていない．**Wilcoxon 検定はターゲットと反復の組（4 ターゲット × 10 反復）を標本にしている．反復は同じテストの折りを共有し（3-setting.tex:60 の "iteration r tests on fold r mod 5"），ターゲットの間の違いと反復の間の違いが 1 つの検定に混ざる．折りを平均すると K=0 で有意な生徒は 11 のうち 3 である．著者自身が認めるとおり（6-comparison.tex:25），この検定は他のコーパスへの一般化を示さない．
  4. **手法を作ったターゲットと評価したターゲットが同じである．**教師の候補は "a preliminary check over two iterations in AV" で，プールのラベルを使って測った J を見て決めている（5-part2-mechanism-to-method.tex:84）．拡張版は，基本版が K ≥ 10 で開始点を下回った後に足されたように読める（同:90–91）．しかも VT のテキストと AT の音声の生徒は拡張版で回していない（同:91 の "the VT text and AT audio students were not run with it"）．評価したターゲットを見ながら手法を作っているので，比較の数値は手法を作るときに見ていないコーパスでの性能を表さない．
  5. **比較に入れた生徒が選ばれている．**単一モダリティの生徒は AVT の 4 ターゲットでしか比べていない．5.4 節で回している AV の音声と顔の生徒（6 ターゲット．AFEW と CREMA-D を含む）は，ターゲットがいちばん多い構成なのに 6 節の比較に入っていない．
- 求める対応:
  1. 要旨・序論・結論の優位の書き方を，ターゲットの平均の差であること，ターゲットの数（AVT は 4，うち 2 つが IEMOCAP），上回ったターゲットの数，折りを平均した後に有意な生徒の数（K=0 で 3/11，K=5 で 7/11）がわかる形に弱める．
  2. AV の音声と顔の生徒（6 ターゲット）を 6 節の比較に加える．
  3. 手法の設計を固定したうえで，設計に使っていないコーパスで 1 回だけ評価する．同じ 4 感情（neutral，happy，sad，angry）と音声・映像をもつ MSP-IMPROV（Busso et al., 2017）はその候補になる．回さないなら，設計と評価に同じターゲットを使ったことを 8 節に書く．
  4. ターゲットごとの差を本文の表か図に出す（いまは Fig. S6 の髭で範囲が見えるだけである）．折りを平均した後の検定の表（p 値と d_z）も補足に出す．
- 著者のデータが必要か: 要（追加のターゲット・生徒の実験と，折りを平均した検定の表）．記述を弱めるだけなら不要．

### [Major-2] 拡張版だけが強い教師を受け取っており，比較の差のうちどれだけが学習の仕方から来て，どれだけが教師の違いから来るのかが分からない
- 該当箇所: sections/6-comparison.tex:9–10 ／ sections/5-part2-mechanism-to-method.tex:85, 91 ／ main.tex:35（"identically tuned competitors"）
- 現状: 6.1 節は "Competitors that need a teacher receive the fused start ..., the fused student refit on the K labels (generalized distillation), or the product of the single-modality starts ...; none receives a two-stage reference, so the margins include the choice of teacher" と書く（6-comparison.tex:9）．拡張版は，K 件のラベルで学び直した融合の生徒と，それが教える単一モダリティの生徒を組み合わせた二段の参照を教師にし，弱いラベルの重みのつけ方（均等，不均等，閾値を超えたものだけ硬いラベル）も K ごとに選ぶ（5-part2-mechanism-to-method.tex:91）．
- 問題: 要旨は競合を "identically tuned" と書くが，競合の教師の候補は拡張版より少なく，二段の参照を受け取らない．表 3 の ours-ext と最強の競合の差 0.8〜3.0 ポイントには，教師の候補の数と強さの差が含まれる．一般化蒸留（generalized distillation，GD）は拡張版と同じ「K 件で学び直した融合の生徒」を教師にもつので，二段の参照と重みのつけ方の選択を除いた ours-ext と GD の差を見れば，学習の仕方の寄与が分かるはずである．その比較が無い．あわせて，各手法が疑似ターゲットで選ぶ候補の数（格子の大きさ）が本文にも補足にも書かれていない．候補の数が違うと，同じ選択の手続きでも有利不利が生じる．
- 求める対応:
  1. 少なくとも GD，温度つき蒸留（KD），硬い疑似ラベル（HPL）に ours-ext と同じ教師の候補（二段の参照を含む）を与え，同じ疑似ターゲットで選んだ結果を表 3 に加える．または，ours-ext の教師を競合と同じ候補に制限した版を示し，教師の寄与と学習の仕方の寄与を分ける．
  2. 各手法の候補の格子（教師，m，温度，閾値，重み）とその大きさを補足に表で示す．
  3. 要旨の "identically tuned" を，選択の手続きが同じで教師の候補は異なることがわかる書き方に直す．
- 著者のデータが必要か: 要

### [Major-3] コーパスをまたぐ感情認識で標準のドメイン適応の手法と比べておらず，ラベル無しの適応で方向を直せないことも確かめていない
- 該当箇所: sections/6-comparison.tex:8 ／ sections/1-introduction.tex:15（Q3）／ sections/2-related-work.tex:5–9 ／ sections/5-part2-mechanism-to-method.tex:55–59 ／ main.tex:35（"Neither few labels nor label-free selection repairs a strong start's direction"）／ キーワード（main.tex:39 の "domain adaptation"）
- 現状: 9 つの競合は，自己学習，硬い疑似ラベル，エントロピー最小化，温度つき蒸留，生成した蒸留，共訓練，共正則化，EM による事前確率の調整，出力の再均衡である（6-comparison.tex:8）．どれも同じ線形の生徒を学び直すか，開始点の出力を直す．ラベル無しで開始点の方向を直せないことは，ラベルを付ける発話の選び方（7 つの規則）でだけ確かめている（5-part2-mechanism-to-method.tex:55–59）．関連研究は ADDoG（Gideon et al., 2021）と Feng と Chaspari（2023）を挙げるが，比較はしていない．
- 問題:
  1. 提案法はラベル無しのターゲットのプールを使う．コーパスをまたぐ音声感情認識で同じ情報を使う標準の手法は，ラベル無しのドメイン適応である．たとえば，凍結した特徴の上でそのまま使える CORAL（Sun et al., 2016．ソースとターゲットの共分散を揃える），敵対的なドメイン適応（Ganin et al., 2016；音声感情では Abdelwahab と Busso, 2018；Latif et al., 2023），線形の回帰でコーパスの違いを吸収する DaLSR（Zong et al., 2016），話者ごとの正規化（Schuller et al., 2010；Busso et al., 2013 の反復的な特徴の正規化）がある．TAC の読者は，6 節の題 "Comparison with Standard Methods" に，これらの手法が含まれていると受け取る．
  2. Part I の結論「閾値の損失の多くはコーパスの平均のずれで，中心化で直る」（4.3 節）は，平均だけを揃える操作の結果である．共分散まで揃える CORAL や，話者ごとの正規化は，ラベル無しで方向（W の向き）も変えうる．したがって，要旨の「ラベル無しでは強い開始点の方向を直せない」という主張を確かめるには，選び方の規則だけでなく，こうしたラベル無しの適応も試す必要がある．提案法の計量 M（ターゲットのプールの散らばりから作る．5-part2-mechanism-to-method.tex:25）はこの方向の操作の一部なので，CORAL との違いを示す意味は大きい．
  3. 多モダリティのドメイン適応（例：Amanda, Zhang et al., 2024）と，テスト時にモダリティが欠けた感情認識（例：MMIN, Zhao et al., 2021）も，「学習時には複数のモダリティがあり，配備時には 1 つしかない」という設定の近くにある．関連研究で位置づけていない．
- 求める対応:
  1. 少なくとも CORAL と話者ごとの正規化を，(a) 開始点を作る前の前処理として，(b) 競合として，同じ疑似ターゲットの手続きで加える．どちらも凍結した特徴の上で閉じた形か軽い計算で回る．
  2. 敵対的なドメイン適応を 1 つ（凍結した特徴の上の小さな網で十分である），同じ選択の手続きで加える．
  3. 回さないなら，その理由を 6.1 節に書き，Q3・6 節の題・要旨の主張を「半教師あり学習と蒸留の手法との比較」に絞る．要旨の "label-free selection" が「ラベルを付ける発話の選び方」の意味であることも明示する．
- 著者のデータが必要か: 要（追加の比較）．主張を絞るだけなら不要．

### [Major-4] 凍結した特徴と線形の分類器が実用上代表的な設定であることの裏づけが弱く，「微調整より強い」という序論の主張は補足 S7 節の記述では確かめられない
- 該当箇所: sections/1-introduction.tex:12 ／ sections/6-comparison.tex:70–72 ／ supplementary.tex:154–161（S7 節）と Fig. S9 ／ sections/3-setting.tex:49 ／ supplementary.tex:35
- 現状: 序論は "a setting simple enough to analyze and, in a check with low-rank adaptation of the audio encoder on four targets at K ≤ 5, stronger than fine-tuning" と書く．S7 節は，WavLM を LoRA で微調整すると K=5 で凍結した開始点より 7.3 ポイント低い（ターゲットごとに 6.3〜9.1）と書く．その根拠は 4 ターゲット × 3 反復である．音声の特徴は WavLM-Large の最終層を時間で平均したものである（3-setting.tex:49）．
- 問題:
  1. **微調整の手順が書かれていない．**LoRA の階数，挿入した層，学習率，エポック数，止め方，ハイパーパラメータの選び方，クラスの重み，そして何のデータで学習したか（ソースのコーパスか，K 件のラベルだけか）が S7 節にない．Fig. S9(a) の K=0 の "LoRA" の棒は，図から目で読むと約 40% で，凍結した開始点（約 47%）より低い．K=0 ではターゲットのラベルが無いので，この棒はソースだけで微調整した模型のはずだが，その説明もない．K=5 の "Fine-tuning on K=5 labels per emotion" が，ソースで微調整した模型をさらに 20 件で調整したものか，20 件だけで調整したものかも読み取れない．後者なら，数千件のソースで学習した凍結した開始点と比べること自体が公平でない．
  2. **比べるべき微調整の手順がある．**少数ラベルの音声感情認識では，ソースの感情コーパスで先に微調整し，その後でターゲットの少数のラベルで調整する手順（Chen と Yu, 2023．本文の [13]）や，LoRA などのパラメータ効率のよい微調整（Feng と Narayanan, 2023）が報告されている．wav2vec 2.0 と HuBERT を感情に微調整した模型を学習に使っていないコーパスで評価した研究（Wagner et al., 2023）もある．S7 節の確認はこれらの手順と対応づけられていない．
  3. **凍結した特徴の取り方が感情認識の慣習と違う．**凍結した自己教師ありの音声模型で感情を認識するとき，SUPERB の評価（Yang et al., 2021）は全層の出力の重みつき和を使い，Pepino et al.（2021）は複数の層の出力を学習した重みで合わせると最終層の出力だけより感情認識に有利な表現になると報告している．最終層だけの平均は，凍結した設定の性能を低めに見積もっているおそれがある．q（16.1〜22.2），n（8.5〜11.8），閾値の割合（0.54〜0.83）は層の選び方に依存しうるので，これらの数値が凍結した設定一般の性質なのか，最終層の性質なのかが分からない．
- 求める対応:
  1. S7 節に微調整の手順をすべて書き，ソースで先に微調整してから K 件で調整する手順を，同じ疑似ターゲットでハイパーパラメータを選んで比べる．
  2. WavLM の層の重みつき和か，疑似ターゲットで選んだ中間層の特徴で，少なくとも開始点・ours-ext と Part I の主要な量（q，閾値の割合）を出し直す．
  3. 1 と 2 を回さないなら，序論の "stronger than fine-tuning" を削り，「1 つの音声エンコーダ，4 ターゲット，3 反復の確認では」と範囲を書いた弱い言い方に替え，最終層の平均という選択を 8 節の限界に書く．
- 著者のデータが必要か: 要（追加の実験）．記述を弱めるだけなら不要．

### 4.2 Minor

### [Minor-1] 話者の折りの作り方が二通りに読め，区間の幅が妥当かどうか判断できない
- 該当箇所: sections/3-setting.tex:60, 99–100 ／ sections/4-part1-what-few-labels-fix.tex:10, 18（Fig. 2 の説明）／ sections/6-comparison.tex:22
- 現状: 3.3 節は "we split the speakers into five folds, and iteration r tests on fold r mod 5" と書き，3.6 節は "The ten IEMOCAP speakers yield only five (improvised) or seven (scripted) distinct test folds" と書く．4.1 節は区間を "treating the thirty iterations as independent" で計算し，Fig. 2 の説明は IEMOCAP のターゲットだけ区間が狭すぎると書く．
- 問題: 話者を折りに割り当てる分け方が全反復で固定なら，どのターゲットでも異なるテストの折りは 5 つしかなく，30 反復は同じ折りを 6 回ずつ使う．その場合，区間が狭すぎるのは IEMOCAP だけではない．分け方を反復ごとに引き直すなら，5 つの折りから台本の IEMOCAP で 7 つの異なる折りが生じる理由が分からない．6.2 節の Holm 補正の族（"all students and budgets"）が K = 0, 5 だけか，Fig. S7 の全 K を含むかも書かれていない．
- 求める対応: 分け方を反復ごとに引き直すかどうか，引き直すならその乱数の種，ターゲットごとの異なるテストの折りの数を書く．折りを共有するターゲットがあるなら，異なる折りを単位にした区間（または折りを単位にした階層のブートストラップ）で 4.1 節の数（42 対のうち 40，27 対）を出し直す．Holm 補正の族の大きさを書く．
- 著者のデータが必要か: 要（区間の出し直し）．書き足すだけなら不要．

### [Minor-2] 顔のエンコーダだけが感情のラベルで学習されているため，モダリティの違いとエンコーダの違いが区別できない
- 該当箇所: supplementary.tex:30 ／ sections/3-setting.tex:48–50 ／ sections/4-part1-what-few-labels-fix.tex:45, 64–66, 74 ／ Table 2 ／ sections/8-limitations-ethics.tex:7
- 現状: 顔の ResNet-50 は "trained on VGGFace2 and fine-tuned on AffectNet" である．音声（WavLM）と言語（RoBERTa）のエンコーダは感情のラベルを見ていない．Part I の第 2 エンコーダは音声（HuBERT）だけで，顔（CLIP）と言語（DeBERTa）は Part I に出てこない．8 節は "why the threshold share is large for the face, whose two losses are about equal, remains unexplained" と書く．
- 問題: AffectNet には neutral，happy，sad，angry が含まれるので，顔の特徴は外部のコーパスの感情ラベルで作られた表現である．顔の q が 1.9 と小さいことと，K=1 での閾値の割合が 0.83 と高いことは，顔というモダリティの性質ではなく，感情で教師あり学習したエンコーダの性質である可能性がある．Part II の融合の教師の強さにも，AffectNet のラベルの情報が入っている可能性がある．3.2 節の "No encoder is fine-tuned on the seven groups" は正しいが，読者は 3 つのモダリティの特徴が同じ条件で作られたと受け取りやすい．
- 求める対応: 3.2 節に，顔のエンコーダだけが感情のラベルで学習されていることを明記する．Table 2 と Fig. 4 に V（CLIP）の行を加え，顔についての Part I の主張を 2 つのエンコーダで支える．8 節の「顔で閾値の割合が大きい理由は説明できない」の箇所で，エンコーダの学習の違いを候補として挙げる．
- 著者のデータが必要か: 要

### [Minor-3] 4 感情への写像と各コーパスのラベルの決め方の記述が足りず，「閾値」の解釈に別の説明が残る
- 該当箇所: sections/3-setting.tex:11–12 ／ Table 1
- 現状: "happy (merged with excited in IEMOCAP)" と "A CMU-MOSEI utterance receives an emotion when exactly one of its six intensity ratings is at least 0.5 and neutral when none is" だけが書かれている．
- 問題:
  1. CMU-MOSEI の 0.5 という閾値と「ちょうど 1 つ」という条件は著者が決めたもので，理由も感度も示されていない．「どの感情も 0.5 未満」を neutral とする定義は，演じた neutral（IEMOCAP，CREMA-D，RAVDESS）とも，対話の文脈で付けた neutral（MELD）とも意味が違う．neutral と他の感情の間の閾値のずれには，コーパスの平均のずれ（4.3 節の解釈）だけでなく，ラベルの定義の違いも入りうる．
  2. CREMA-D のラベルが演者に指示した感情か，評定者が知覚した感情かが書かれていない（Table 1 の 1,271 件ずつという数は指示した感情を示唆するが，本文で確かめられない）．IEMOCAP の評定者の多数決の規則と同票の扱い，RAVDESS の 2 つの強さの両方を使ったか，も書かれていない．
- 求める対応: 各コーパスのラベルの決め方を 3.1 節か補足に書く．CMU-MOSEI の規則について理由を書き，閾値を変えた感度の確認を 1 つ示す．閾値の損失を感情の対ごとに（neutral を含む対とそれ以外で）分けて示し，neutral の定義の違いが閾値のずれを作っているかを確かめる．
- 著者のデータが必要か: 要（感度の確認と対ごとの内訳）

### [Minor-4] 多人数の場面の顔の取り方と，RAVDESS の顔を使わない理由が書かれていない
- 該当箇所: supplementary.tex:27–29 ／ sections/3-setting.tex:14
- 現状: "we keep the largest face of each frame" と書き，RAVDESS については "we did not extract face features for RAVDESS" とだけ書く．
- 問題: MELD（テレビドラマの多人数の場面）と AFEW（映画の切り抜き）では，各フレームで最も大きい顔が話者でない場合がある．顔の特徴の質がコーパスごとに違えば，V と AV の結果と融合の教師の強さもコーパスごとに違ってくる．RAVDESS は正面の顔の映像を含むコーパスとして音声と映像の感情認識で広く使われており，使わない理由が無いと V と AV のターゲットが 1 つ減った理由が読者に分からない．
- 求める対応: コーパスごとに，顔が検出された発話の割合と，最大の顔が話者である割合（無作為に選んだ発話を目で確かめる程度でよい）を補足に示す．RAVDESS の顔を使わない理由を書くか，加える．
- 著者のデータが必要か: 要

### [Minor-5] 改善を閾値の部分と方向の部分に分ける式が書かれておらず，見出しの数値 0.54〜0.83 の定義が読み取れない
- 該当箇所: sections/4-part1-what-few-labels-fix.tex:74 ／ sections/3-setting.tex:88–89
- 現状: "Split into a threshold part and a direction part with (1), the improvement of the pooled classifier over the source-only classifier is dominated by the threshold part" と書くが，分け方の式はない．Δ は "a shrinkage linear discriminant on one half of the target speakers and measured on the other half" で推定する．
- 問題: a(d, o) は非線形なので，o を先に動かすか d を先に動かすかで二つの部分の大きさが変わる．要旨の 0.54〜0.83 はこの分け方で決まる．Δ は推定値なので，小さいターゲット（RAVDESS は 24 話者 672 発話，AFEW は 795 発話）では一部の分類器の d が Δ を上回り，方向の損失が負になりうるが，その扱いも書かれていない．分析は対ごとの正解率の点数で行い，手法の評価は 4 クラスの UAR で行うので，二つの尺度の関係も示されていない．
- 求める対応: 分け方の式を書き，二つの順序の両方（またはその平均）で閾値の割合を示す．方向の損失が負になった対の割合と扱いを書く．対ごとの正解率の変化と UAR の変化の対応（例：対の平均の変化と UAR の変化の相関）を補足に示す．
- 著者のデータが必要か: 要

### [Minor-6] テキストと融合の構成で q を示していない理由と，式 (3) の当てはまりが書かれていない
- 該当箇所: Table 2（sections/4-part1-what-few-labels-fix.tex:105–108）／ 同:45
- 現状: q は A，A（HuBERT），V，AV で示し，T，VT，AT，AVT は "–" である．
- 問題: 方向の損失がいちばん大きいのはテキスト（11.5 ポイント）であり，テキストの q は Part I の主張にとって最も知りたい値である．空欄の理由が書かれていない．式 (3) を学習曲線に当てはめた残差も示されていない．
- 求める対応: 空欄の理由を書くか値を示す．当てはめの残差（K ごとの実測と予測の差）を補足に示す．
- 著者のデータが必要か: 要

### [Minor-7] 序論の「全音声ソースを混ぜると 20.2〜30.5」の上端が本文に出てこない
- 該当箇所: sections/1-introduction.tex:29 ／ sections/4-part1-what-few-labels-fix.tex:145
- 現状: 序論の貢献 3 は "20.2–30.5 for all audio sources pooled" と書き，4.5 節は "n is 20.2 and 28.9 labels per emotion for WavLM audio at K=5 and K=10" と書く．
- 問題: 30.5 が本文のどこにもないので，読者は序論の値を確かめられない（HuBERT の値かと推測するしかない）．
- 求める対応: 4.5 節に HuBERT の値を書くか，序論を 4.5 節の値に合わせる．
- 著者のデータが必要か: 不要

### [Minor-8] 「22 セルのうち 19」の数え方で，同点の扱いが一貫していない
- 該当箇所: main.tex:35 ／ sections/1-introduction.tex:30 ／ sections/6-comparison.tex:19 ／ Table 3 ／ Table S2
- 現状: Table 3 の AVT の融合の生徒（K=5）は最強の競合も基本版も開始点から 0.8 ポイントで，Table S2 でも ST と Ours はともに 54.6% である．本文はこのセルを勝ちに数えているように読める．一方，AT の融合の生徒（K=5）は表 3 で両方 0.1 ポイントで，本文は "level" と書く．
- 問題: 19 という数は 0.05 ポイント未満の差で決まっており，同じ見かけの二つのセルを別々に扱っている．
- 求める対応: 同点の規則（例：差が ±0.1 ポイント以内，または検定で有意でないものを同点とする）を決め，「勝ち x，同点 y，負け z」の形で書く．
- 著者のデータが必要か: 不要

### [Minor-9] Table 2 の HuBERT の行だけ対の集合が違い，4.3 節の見出しと中央値の記述が合わない
- 該当箇所: sections/4-part1-what-few-labels-fix.tex:90（Table 2 の説明）, 64–66
- 現状: Table 2 の説明は "(for A (HuBERT) the losses include all pairs)" と書く．4.3 節の見出しは "Source-only classifiers lose mainly through the direction, except on the face" で，直後に "The median over pairs of the threshold's share of the loss is 0.10–0.27 across configurations" と書く．
- 問題: HuBERT の損失だけ同じコーパスの対を含むので，Table 2 で A と A（HuBERT）を比べられない．顔の平均の損失は閾値 2.5，方向 2.4 で，閾値の割合は約 0.5 になるが，全構成の中央値が 0.10〜0.27 なら顔の中央値もこの範囲にある．見出しの「顔を除いて」は平均にだけ当てはまる．
- 求める対応: HuBERT の行を他の行と同じ「異なるコーパスの対」で計算し直す．顔の閾値の割合の中央値を別に書き，平均と中央値のどちらで述べているかを見出しと本文で揃える．
- 著者のデータが必要か: 要（HuBERT の行の出し直し）

### [Minor-10] 手法の名前と記号が定義なしに出てきて，TAC の読者が基本版と拡張版を組み立て直せない
- 該当箇所: sections/5-part2-mechanism-to-method.tex:75（"belief"）, 83–85（"product of the single-modality starts"，"down-weighted where these starts disagree"，"two-stage reference"）, 91（"extended family"）／ Fig. 6 の凡例（"AVT world"）／ Fig. 1，Fig. 5 ／ main.tex:35（要旨）
- 現状: 例として 5.4 節は "For a fused student, ... so its candidates add a two-stage reference: single-modality students, trained on the pool and the K labels with the fused start as teacher, are combined, and the product of their predictions either teaches the fused student or replaces it." と書く．5.3 節の "A belief that combines the per-source classifiers, each weighted by its confusion pattern on pseudo-targets" の "belief" は式で定義されていない．"AVT world"（ある構成を作るのに使う群の集合）は図にしか出てこない．
- 問題: 本文 12 頁に，Part I の理論と測定，Part II の手法，比較が入っている．各段落が 3〜6 個の数値の範囲を並べ，要旨にも 8 個の数値がある．感情計算の読者は，どの生徒にどの教師の候補が与えられ，何を K ごとに選ぶのかを本文から組み立て直せない．Fig. 5 と Fig. 1 の文字は印刷の大きさでは小さい．
- 求める対応: 基本版と拡張版を手順の枠（algorithm）で示し，生徒の種類ごとの教師の候補，重みのつけ方，選ぶもの，格子を並べる．記号の表を 1 つ置く．"world" を本文で定義する．要旨の数値を主要なものに絞る．頁が足りなければ，4.4 節のニュートン法の 1 歩による導出を補足に移す．
- 著者のデータが必要か: 不要

### [Minor-11] 競合の格子，原論文の値，選び方の 7 つの規則，コードの公開が書かれていない
- 該当箇所: sections/6-comparison.tex:10（"the fixed version uses the values of the original papers"）／ sections/5-part2-mechanism-to-method.tex:55, 68（"a small grid"）／ 補足 S5 節
- 現状: 補足 S5 節は開始点の 5 つの値の格子を示す．競合の格子，固定版で使った原論文の値，ラベル無しの 7 つの選び方の定義，教師と m の格子，硬いラベルの閾値 τ は本文にも補足にもない．コードやデータ処理の台本を公開するかどうかも書かれていない．
- 求める対応: これらを補足に表で示す．コードの公開の予定を書く．
- 著者のデータが必要か: 不要（記録の転記）

### [Minor-12] 上がり幅の実用上の大きさを判断する基準が示されていない
- 該当箇所: Table 3 ／ sections/7-discussion.tex:5
- 現状: Table 3 は開始点の UAR（42.1〜62.0%）と，そこからの差（0.0〜4.6 ポイント）を示す．
- 問題: 読者は，ターゲットに全ラベルがあるときとの差のうちどれだけを手法が埋めたのか，また 1〜3 ポイントがラベル何件ぶんに当たるのかを判断できない．著者は 4.5 節で学習曲線の上の位置からラベル数に換算する道具を持っている．
- 求める対応: Table 3 に，プールの全ラベルで学習した分類器（5.2 節の W* に当たるもの）の UAR を上限の参照として加える．ours-ext の得を，4.5 節と同じ方法でクラスあたりのラベル数に換算して示す．
- 著者のデータが必要か: 要

### [Minor-13] 先行研究との差の書き方が不正確な箇所があり，コーパスをまたぐ感情認識の主要な先行研究が抜けている
- 該当箇所: sections/2-related-work.tex:29–30, 5–9 ／ sections/1-introduction.tex:32
- 現状: 関連研究は "In these studies the teacher is trained on the target corpus or a third corpus, or the student uses all modalities at inference; the teachers of our basic method are trained on the source corpora alone" と書く．
- 問題:
  1. Albanie et al.（2018）は，VGGFace2 で事前学習し FERPlus で表情に調整した SE-ResNet-50 を教師にして，ラベル無しの VoxCeleb の動画に表情のラベルを付け，音声だけの生徒を学習した．教師の作り方は本論文の顔のエンコーダ（VGGFace2 で学習し AffectNet で調整した ResNet-50）とほぼ同じである．「第三のコーパスで学習した教師」と「ソースのコーパスで学習した教師」の違いは小さい．本論文の実際の違いは，(i) 融合した教師を使う，(ii) ターゲットの少数のラベルとソースから作った開始点への固定を組み合わせる，(iii) K ごとに疑似ターゲットで教師と重みを選ぶ，(iv) 感情ごとに重みを揃える，の 4 点である．
  2. コーパスをまたぐ音声感情認識のドメイン適応（Abdelwahab と Busso, 2018；Zong et al., 2016；Latif et al., 2023），話者ごとの正規化（Busso et al., 2013），多モダリティの感情認識のドメイン適応（Zhang et al., 2024），モダリティが欠けた感情認識（Zhao et al., 2021），自己教師ありの音声模型の感情への微調整（Wagner et al., 2023；Feng と Narayanan, 2023）が引用されていない．
  3. 「閾値の損失の多くはコーパスの平均のずれで，中心化で直る」という Part I の結果は，コーパスごと・話者ごとの正規化が効くという従来の知見（Schuller et al., 2010；Busso et al., 2013）の近くにある．本論文の新しさは，その効果を K の関数として測り，ラベルが足す改善の何割に当たるかを示した点にあるので，そう書くべきである．
- 求める対応: 2 節の差の文を上の (i)〜(iv) の形で書き直す．上の文献を加え，Part I の閾値の結果を正規化の先行研究に対して位置づける．
- 著者のデータが必要か: 不要

### [Minor-14] エントロピー最小化の失敗を「クラスがまとまりを作るか」で説明しているが，まとまりを測っていない
- 該当箇所: sections/7-discussion.tex:10–11
- 現状: "Entropy minimization relies on classes that form separate clusters in the unlabeled data, which holds on the improvised IEMOCAP sessions in AVT and AT but not on MELD, CMU-MOSEI, or the same target in VT and AV" と書く．
- 問題: どのターゲットでクラスがまとまりを作るかを，エントロピー最小化が効いたかどうかから逆に推しており，説明が結果の言い換えになっている．同じターゲット（IEMOCAP の即興）でも AVT・AT では効き VT・AV では効かないことは，まとまりよりも音声とテキストの組み合わせの有無を示しているようにも読める．
- 求める対応: プールでのまとまりの指標（例：真のラベルを使ったシルエット係数，決定境界の近くの密度）を構成ごとに示すか，この段落を仮説として弱く書く．
- 著者のデータが必要か: 要（指標を示す場合）．弱めるだけなら不要．

### [Minor-15] 倫理の節が，コーパスの利用許諾，野外の映像に映る人の同意，属性ごとの性能，顔の埋め込みの扱いに触れていない
- 該当箇所: sections/8-limitations-ethics.tex:18–21
- 現状: "We used only public corpora under their distribution terms and recorded no new data. Faces enter the pipeline only as embeddings of detected face crops, and we release no identifying data." と書く．
- 問題: IEMOCAP や AFEW などは利用許諾の申請を経て配布されるコーパスであり，"public" とひとまとめにすると読者は誤解する（コーパスごとの条件は著者が確かめて書くべきである）．CMU-MOSEI（YouTube），MELD（テレビドラマ），AFEW（映画）に映る人は感情の分析に同意していない．顔の埋め込みは顔の同定用のデータ（VGGFace2）で事前学習した網から作られており，同定に使える情報を含みうる．性別などの属性ごとの性能も報告されていない．顔の表情認識では属性による性能の差が問題になるので，TAC の読者はこの点を気にする．
- 求める対応: コーパスごとの利用条件を書く．野外のコーパスの同意の問題と，顔の埋め込みを公開するかどうかとその危険を書く．性別の情報があるコーパス（IEMOCAP，CREMA-D，RAVDESS など）で，主要な比較の UAR を性別ごとに示す．
- 著者のデータが必要か: 要（性別ごとの UAR）．記述は不要．

### 4.3 Optional

### [Optional-1] 外部の表情データで学習した顔の教師を，Albanie et al.（2018）に当たる比較として加える
- 該当箇所: sections/5-part2-mechanism-to-method.tex:83 ／ supplementary.tex:30
- 現状: 教師の候補はすべて本論文のソースのコーパスで学習した分類器である．
- 問題: 顔のエンコーダは AffectNet で表情に調整されている．その分類の頭が配布物に含まれているなら，AffectNet の 4 感情の出力をそのままラベル無しのプールに付ける教師は，追加の学習なしに作れる．これは Albanie et al.（2018）の「外部のコーパスで学習した顔の教師から音声の生徒へ」という手順に当たり，ソースのコーパスで学習した教師を使う利点を直接示す比較になる．
- 求める対応: 可能なら，この教師を候補または比較に加える．
- 著者のデータが必要か: 要

### [Optional-2] q，e，J を疑似ターゲットで見積もる「次の段階」を 1 つの構成で示す
- 該当箇所: sections/9-conclusion.tex:7
- 現状: 結論は "Because q, e, and the teacher's UAR advantage J were measured with target labels, estimating them on pseudo-targets ... is the next step" と書く．
- 問題: 実務家にとって最も役立つのは，ラベルを付ける前に「ラベルを足す，ソースを足す，別のモダリティを記録する」のどれが得かを決められることである．いまの数値はターゲットのラベルで測ったものなので，その判断に使えない．
- 求める対応: 1 つの構成（例：A）で，疑似ターゲットで見積もった q と J と，ターゲットのラベルで測った値の対応を示すと，7 節の実務の助言の価値が上がる．
- 著者のデータが必要か: 要

### [Optional-3] 言語，感情の種類，ラベルの形式への一般化を 1 つでも確かめる
- 該当箇所: sections/8-limitations-ethics.tex:10
- 現状: "We study four emotions in seven groups from six English corpora" と書く．
- 問題: コーパスをまたぐ音声感情認識の研究の多くは言語をまたぐ設定（例：Latif et al., 2023）を含み，TAC の読者は英語以外での再現を気にする．次元のラベル（覚醒度や快不快）や 4 感情以外のクラスで閾値と方向の分解が同じ傾向を示すかも分からない．
- 求める対応: 音声だけでよいので英語以外のコーパスを 1 つターゲットに加えるか，MSP-Podcast（Lotfian と Busso, 2019）のような大きな自然な発話のコーパスで Part I の主要な量を出す．
- 著者のデータが必要か: 要

### [Optional-4] 「学習時には複数のモダリティがあり，配備時には 1 つだけ」という設定の具体的な場面を書く
- 該当箇所: sections/7-discussion.tex:5
- 現状: "if other modalities are recorded with the target utterances, even only at labeling time, let a teacher ... label the whole pool, while the deployed classifier remains single-modality" と書く．
- 問題: 読者はこの条件がどの応用で成り立つのか（例：開発時に映像つきで収録し，配備は電話の音声だけで行うコールセンター）を自分で補う必要がある．
- 求める対応: 具体的な場面を 1 文で書く．
- 著者のデータが必要か: 不要

### [Optional-5] 列の正規化とプールのクラスの偏りの関係を MELD で調べる
- 該当箇所: sections/8-limitations-ethics.tex:16 ／ Table 1
- 現状: "we did not analyze how its gain depends on the class ratios" と書く．MELD のプールは Table 1 で neutral が 11,343 件のうち 6,430 件（約 57%）である．
- 問題: 感情ごとに重みを揃える処理は，プールのクラスの比が偏ったターゲットでは誤った事前確率を教えうる（Veilleux et al., 2021）．偏りが最も大きい MELD での結果が，その危険の大きさを示す．
- 求める対応: MELD で，均等の重みと不均等の重みの差（拡張版はこの 2 つを選べる）を示す．
- 著者のデータが必要か: 要

## 5. 比べるべき既存の文献

「確かめた水準」は，査読のために Web で何を確かめたかを示す．どの文献も全文は精読していない．

### 5.1 比較の実験に加えるべきもの（Major-3，Major-4）

| 文献 | 本論文との関係 | 確かめた水準 |
|---|---|---|
| B. Sun, J. Feng, K. Saenko, "Return of frustratingly easy domain adaptation," *AAAI*, 2016, pp. 2058–2065. | ソースとターゲットの共分散を揃えるラベル無しの適応（CORAL）．凍結した特徴の上で閉じた形で回り，方向を変えうる．開始点の前処理と競合の両方に使える | AAAI の論文ページと mlanthology で書誌と要旨を確認 |
| C. Busso, S. Mariooryad, A. Metallinou, S. Narayanan, "Iterative feature normalization scheme for automatic emotion detection from speech," *IEEE Trans. Affective Computing*, vol. 4, no. 4, pp. 386–397, 2013. | 話者ごとにラベル無しで特徴を正規化する手順．本論文の中心化（コーパスの平均を引く）を話者の単位に細かくしたものに当たる | USC SAIL の書誌ページと検索結果の要旨で確認 |
| Y. Ganin, E. Ustinova, et al., "Domain-adversarial training of neural networks," *JMLR*, vol. 17, no. 59, pp. 1–35, 2016. | 敵対的なドメイン適応の基本形．凍結した特徴の上の小さな網で競合に加えられる | JMLR のページで書誌を確認 |
| M. Abdelwahab, C. Busso, "Domain adversarial for acoustic emotion recognition," *IEEE/ACM Trans. Audio, Speech, and Language Processing*, vol. 26, no. 12, pp. 2423–2435, 2018. | 音声感情認識で，ラベル無しのターゲットのデータを使う敵対的な適応 | NSF PAR の書誌と arXiv:1804.07690 の要旨で確認 |
| Y. Zong, W. Zheng, T. Zhang, X. Huang, "Cross-corpus speech emotion recognition based on domain-adaptive least-squares regression," *IEEE Signal Processing Letters*, vol. 23, no. 5, pp. 585–589, 2016. | ラベル無しのターゲットを使う線形のコーパスをまたぐ手法．線形の分類器という本論文の設定に近い | 北京理工大学の機関リポジトリで書誌と要旨を確認 |
| S. Latif, R. Rana, S. Khalifa, R. Jurdak, B. Schuller, "Self supervised adversarial domain adaptation for cross-corpus and cross-language speech emotion recognition," *IEEE Trans. Affective Computing*, vol. 14, no. 3, pp. 1912–1926, 2023. | TAC に載った，コーパスと言語をまたぐ音声感情認識のラベル無しの適応 | USQ と Augsburg 大学の機関リポジトリで書誌を，arXiv:2204.08625 で要旨を確認 |
| M. Chen, Z. Yu, "Pre-finetuning for few-shot emotional speech recognition," *Proc. Interspeech*, 2023, pp. 3602–3606.（本文の [13]） | ソースの感情コーパスで先に微調整し，少数のラベルで調整する手順．Major-4 の微調整の比較で従うべき手順 | ISCA Archive と arXiv:2302.12921 で確認 |
| T. Feng, S. Narayanan, "PEFT-SER: On the use of parameter efficient transfer learning approaches for speech emotion recognition using pre-trained speech models," *Proc. ACII*, 2023. | LoRA などによる音声感情認識の微調整の比較．S7 節の LoRA の手順の参照になる | arXiv:2306.05350，PyPI のページ，ISI の書誌で会議を確認．頁は未確認 |
| S.-w. Yang et al., "SUPERB: Speech processing universal performance benchmark," *Proc. Interspeech*, 2021, pp. 1194–1198. | 凍結した音声模型の評価で層の重みつき和を使う標準の手順 | ISCA Archive で書誌を確認．重みつき和の手順は記憶と，後続の雑誌版（Yang et al., IEEE/ACM TASLP, 2024，arXiv:2404.09385）の紹介文による．本文では未確認 |
| L. Pepino, P. Riera, L. Ferrer, "Emotion recognition from speech using wav2vec 2.0 embeddings," *Proc. Interspeech*, 2021. | 複数の層の出力を学習した重みで合わせると感情認識に有利な表現になるという報告 | ISCA Archive の書誌と arXiv:2104.03502 の要旨で確認．層ごとの数値と頁は未確認 |

### 5.2 関連研究で位置づけるべきもの（Minor-13，Optional-1，Optional-3）

| 文献 | 本論文との関係 | 確かめた水準 |
|---|---|---|
| S. Albanie, A. Nagrani, A. Vedaldi, A. Zisserman, "Emotion recognition in speech using cross-modal transfer in the wild," *Proc. ACM Multimedia*, 2018, pp. 292–301.（本文の [34]） | VGGFace2 で事前学習し FERPlus で調整した顔の教師で，ラベル無しの動画から音声だけの生徒を学習する．本論文の基本版との差を正確に書く必要がある | arXiv:1808.05561 の内容の要約で教師と生徒の構成を確認．頁は本文の文献表による |
| X. Zhang, J. Sun, S. Hong, T. Li, "Amanda: Adaptively modality-balanced domain adaptation for multimodal emotion recognition," *Findings of ACL*, 2024, pp. 14448–14458. | 多モダリティの感情認識のラベル無しのドメイン適応 | ACL Anthology で書誌と要旨を確認．使ったデータセットとソース・ターゲットが別のコーパスかどうかは未確認 |
| J. Zhao, R. Li, Q. Jin, "Missing modality imagination network for emotion recognition with uncertain missing modalities," *Proc. ACL-IJCNLP*, 2021, pp. 2608–2618. | テスト時にモダリティが欠けた感情認識．「学習時は複数，配備時は 1 つ」の設定の近く | ACL Anthology で確認 |
| J. Wagner, A. Triantafyllopoulos, et al., "Dawn of the transformer era in speech emotion recognition: Closing the valence gap," *IEEE Trans. Pattern Analysis and Machine Intelligence*, vol. 45, no. 9, pp. 10745–10759, 2023. | wav2vec 2.0 と HuBERT を MSP-Podcast で微調整し，IEMOCAP と MOSI でコーパスをまたいで評価した．凍結した設定を選ぶ理由を述べるときの比較相手 | 検索結果の書誌と arXiv:2203.07378 の要旨で確認．コーパスをまたぐ数値は未確認 |
| C. Busso, S. Parthasarathy, A. Burmania, M. AbdelWahab, N. Sadoughi, E. Mower Provost, "MSP-IMPROV: An acted corpus of dyadic interactions to study emotion perception," *IEEE Trans. Affective Computing*, vol. 8, no. 1, pp. 119–130, 2017. | 本論文と同じ 4 感情と音声・映像をもつコーパス．設計に使っていないターゲットの候補（Major-1） | MSP 研究室の書誌で確認 |
| R. Lotfian, C. Busso, "Building naturalistic emotionally balanced speech corpus by retrieving emotional speech from existing podcast recordings," *IEEE Trans. Affective Computing*, vol. 10, no. 4, pp. 471–483, 2019. | MSP-Podcast．近年のコーパスをまたぐ音声感情認識で広く使われる大きな自然な発話のコーパス | UTD MSP 研究室のページで書誌を確認 |
| B. Schuller et al., "Cross-corpus acoustic emotion recognition: Variances and strategies," *IEEE Trans. Affective Computing*, vol. 1, no. 2, pp. 119–131, 2010.（本文の [1]） | 話者ごと・コーパスごとの正規化の効果を示した．Part I の閾値の結果を位置づける基準 | 書誌は本文の文献表による．内容は記憶による．査読のための追加の確認はしていない |

コーパスをまたぐ音声感情認識の部分空間の学習による転移（"Transfer linear subspace learning for cross-corpus speech emotion recognition"，IEEE Xplore の文書番号 7931593）も関係するが，著者と誌名を確かめられなかったので表に入れていない（未確認）．
