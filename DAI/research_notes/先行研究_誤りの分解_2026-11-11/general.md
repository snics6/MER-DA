**検証の結果（2026-11-12，`verify_general.md`）**：数値の食い違いは 0．Johnson 2018 の他病院のモデルは「45 病院」ではなく，46 病院を病院単位で半分に分けた学習用の側で学んだもの．他病院のモデルの当て直しにもその病院のラベルを使い，「その病院のラベルが較正を直す」は設計からの解釈．RLSbench の 2〜10 pt は学習中の再標本化と事後の重み直しを組んだ手順全体の得で，切片の補正だけの得ではない．Debray 2013 の事例 2 は「一般に 1 に近くない」で，事例 3（効果のばらつきが弱く，発生率から求めた切片で全研究の E/O が 1 に近い）を書き落としていた．Vergouwe 2017 は本文で確かめ ◎．見落としの部分一致に Godau ほか 2023（MICCAI，arXiv 2303.12540）．

# 誤りの切片側と向き側への分解：一般の機械学習の先行研究（2026-11-11）

担当：モダリティを問わない一般の機械学習（画像分類のドメイン適応・テスト時適応・少数ショット・線形プローブ，ラベルシフトの理論，基盤モデルの較正，クロスモーダル蒸留，医療の予測モデルの更新）．顔・言葉・声と融合は別の 3 体が調べている．

## 要約（判定）

1. 完全一致（実データで，複数の情報源を同じ切片側／向き側の分け方で比べた研究）は，今回の検索（65 回）と本文の確認の範囲では見つからなかった．
2. 最も近いのは医療の予測モデルの Johnson, Pollard, Naumann 2018（eICU，46 病院）である．他病院のデータで学んだモデルに直線の当て直しをかけたものと，その病院の 200〜2,400 人のラベルだけで学んだモデルを，判別（AUROC）と較正（SMR）に分けて比べた．報告は「その病院のラベルは較正を直すが，判別では 2,400 人でも他病院のモデルに届かない」である．情報源①と②を切片側／向き側に近い分け方で比べた例で，二値の 1 課題・1 病院に限られる．
3. 分け方そのものは既知である．医療統計では「切片だけの当て直し → 切片と傾き → 係数の推定し直し」の段階を，新しい施設のラベルの数に応じて選ぶ研究が標準である（Steyerberg 2004，Vergouwe 2017，Davis 2019）．論文で「切片と向きに分ける見方を導入した」とは書けない．
4. 情報源ごとの部分一致は次のとおりである．②は Joshi ほか 2012（多ドメイン学習の得の約半分を，ドメインごとの切片だけで再現できた）．③は RLSbench 2023 と DART 2024（ラベル無しの適応の上に，出力の切片を当て直した分の得を測った）．④は理論の予言だけがあり（Blum & Mansour 2017，Menon 2015，Elkan & Noto 2008），実データで分けて測った研究は見つからなかった．
5. 注意が要る理論として，Chen ほか 2020 はラベル無しの自己学習が向き（偽の特徴への重み）まで直しうることを示した．「③は切片しか直さない」を一般則として書くことはできない．
6. 見つからなかったのは次の研究である．③④を①②と同じ分け方で測った研究，多クラスで「切片の当て直しで取り戻せる成績」を情報源ごとに測った研究，その内訳をラベル数 K の関数として測った研究．

---

## 0. この報告で使う語

- **切片側の誤り（a）**：重みを固定したまま，切片（バイアス項・閾値・クラスの割合）を当て直せば取り戻せる誤り．
- **向き側の誤り（b）**：切片をどう当て直しても残る誤り．判別の向きや特徴の重みのずれによる．
- **判別（discrimination）と較正（calibration）**：医療統計の 2 つの評価軸．判別は「陽性の人に陰性の人より高い点をつけられるか」で，AUROC（陽性と陰性を 1 人ずつ選んだとき陽性の点が高い確率）で測る．較正は「予測した確率が実際の割合と合うか」で，切片（calibration-in-the-large，全体の平均が合うか）と傾き（calibration slope）で測る．二値では，切片と傾きの当て直しは点の順位を変えないので AUROC を変えない．このため判別は（b），切片の当て直しで直る較正の部分は（a）にほぼ対応する．
- **当て直し（recalibration）**：学んだモデルの出力（対数オッズ）に，新しいデータで切片だけ，または切片と傾きだけを付け直すこと．重みの向きは変えない．
- **事後の重み直し（re-weighting，RW）**：ターゲットのクラスの割合を推定し，出力の確率に「ターゲットの割合 ÷ ソースの割合」をかけること．対数の上ではクラスごとの切片を足す操作と同じである．
- **情報源**：①ターゲットの少数のラベル，②ほかのドメインのラベル付きデータ，③ターゲットのラベル無しデータ，④ほかのモダリティ（または別の見方）の予測．
- **水準**：◎ 原文で該当箇所を確かめた．○ 抄録か信頼できる二次資料．△ 検索の断片だけ．「（過去）」は手元の過去の調査で ◎ を付けたもの．

---

## 1. 判定の詳細

### 1.1 完全一致

見つからなかった．実データで，①〜④のうち 2 つ以上を，「切片の当て直しで取り戻せる分」と「残る分」の同じ分け方で比べた研究は無かった．①②の 2 つを判別と較正に分けて比べた Johnson ほか 2018 が最も近いが，次の 3 点で完全一致にあたらない．

- 分けた量が違う．「切片の当て直しで取り戻せる成績」と「残りの成績」ではなく，AUROC と SMR（観測した死亡数 ÷ 予測した死亡数．1 なら全体の平均が合う）を別々に報告している．
- 二値の 1 課題（院内死亡），評価は 1 病院（hospital 73）である．
- ③は付録で 1 つの操作（病院ごとの平均と標準偏差で特徴を標準化）を試しただけで，「目立った差は無かった」とだけ書いている．④は扱っていない．

### 1.2 部分一致（情報源が 1 つ，または分け方が似た量）

| 情報源 | 研究 | 何を分けたか |
|---|---|---|
| ① | Steyerberg ほか 2004，Vergouwe ほか 2017，Davis ほか 2019（医療の予測モデルの更新） | 新しい施設のラベル数に応じて，切片だけ／切片と傾き／係数の推定し直しのどれを選ぶか |
| ① | Lee ほか 2023（surgical fine-tuning） | 少数のターゲットのラベルでどの層を学び直すと最もよいか．出力側のずれでは最終層．ただし最終層の重みと切片を分けていない |
| ① | Zhu ほか 2023（GLA） | CLIP のゼロショットの出力から，事前学習のクラスの偏り（16-shot のラベルで推定）を引いた分の得 |
| ①② | Johnson ほか 2018 | 1.1 節のとおり |
| ② | Joshi ほか 2012 | 多ドメイン学習の得のうち，ドメインごとの切片だけで再現できる分 |
| ③ | Garg ほか 2023（RLSbench） | ドメイン適応の各手法の上に，推定したクラスの割合で事後の重み直しを足した分の得 |
| ③ | DART（arXiv 2411.15204） | テスト時の BN の統計の入れ替えの上に，出力のロジットへ「行列 W」と「ベクトル b（切片）」を足す補正を分けて足した分の得 |
| ③ | Cai ほか 2023（DISDE），Zhang ほか 2023 | 成績の低下を，データの側のずれ（X のずれと Y|X のずれ，または P(Y) などの分布）に割り振る．分類器の切片と重みには割り振らない |
| ③ | Zhao ほか 2021（Calibrate Before Use） | 中身の無い入力（"N/A"）への出力から求めたクラスごとの偏りを引く補正（対数の上でクラスごとの切片を足すのと同じ）の得を，例示の数 0〜8 で比べた．大規模言語モデルの文脈内学習で，分類器を学ぶ設定ではない |

### 1.3 関係する理論

Okamoto 1963 と Raudys 1998（線形判別の誤り率の標本数による展開．係数から切片の項と向きの項が数値で分かれる．解釈は手元の計算），Saerens ほか 2002・Lipton ほか 2018・Garg ほか 2020（ラベルシフト．ラベル無しで事前確率を推定し出力を直す），Alexandari ほか 2020（較正にクラスごとの切片を入れないとラベルシフトの補正が効かない），Maity ほか 2024（ターゲットの対数オッズを「ソースの対数オッズ＋低次元の調整」と置く．切片だけの調整をラベルシフトの特別な場合として挙げる），Natarajan ほか 2013・Menon ほか 2015・Elkan & Noto 2008（雑音の入ったラベルで学んだ分類器は順位が保たれ閾値だけがずれる），Blum & Mansour 2017（共訓練で別の見方の予測は向きを決め，閾値は別に直す），Chen ほか 2020・Oymak & Gulcu 2021（ラベル無しの自己学習が向きを直す条件），Castelli & Cover 1996（ラベル無しは混合の形を，ラベルはどれがどのクラスかを決める），Kull & Flach 2015・Berta ほか 2025（損失を「当て直しで消える部分」と「残る部分」に分ける尺度）．

### 1.4 該当しない（正解率だけ，または層・表現の単位）

Kirichenko ほか 2023（DFR），Izmailov ほか 2022，LaBonte ほか 2023，Rosenfeld ほか 2022（DARE），Kang ほか 2020，Burns & Steinhardt 2021，LAME（Boudiaf ほか 2022），ZERO（Farina ほか 2024），DN（Zhou ほか 2023），Frolic（2024），DebiasPL（Wang ほか 2022），AMU-Tuning（Tang ほか 2024），MFH（Xue ほか 2023），FPS（arXiv 2508.18693），WhyShift（Liu ほか 2023）．表現と最終層を分けたもの，正解率だけを比べたもの，切片の補正だけを提案したものである．AMU-Tuning の "logit bias" は入力ごとに変わる追加のロジットで，切片ではない（取り違えに注意）．

---

## 2. 研究ごとの表

| 研究 | 設定 | 測った量 | (a)(b) を分けたか | 情報源 | 水準 | URL |
|---|---|---|---|---|---|---|
| Johnson, Pollard, Naumann 2018（arXiv．掲載先は未確認） | eICU 2.0，46 病院 50,106 件の ICU 滞在，院内死亡，L2 罰則つきロジスティック回帰（82 特徴）．病院 73 で 500 人を評価に残し，200 人刻みの局所データで 50 回繰り返す | 他病院のモデル＋1 変数のロジスティック回帰による当て直し（ずらしと伸縮）と，局所データだけのモデルの AUROC と SMR．付録 C で病院ごとの平均と標準偏差による標準化 | 近い．当て直しは単調な変換なので AUROC を変えないと明記．他病院のモデルの AUROC は 0.849 で一定，局所のモデルは 0.614 から 2,400 人で 0.796．局所のモデルの SMR は 1 を大きく超えた値から 1 に近づく．付録 C の標準化は「目立った差は無い」 | ①②（③は付録の 1 操作） | ◎（HTML 本文） | https://arxiv.org/abs/1812.02275 |
| Joshi, Dredze, Cohen, Rosé 2012（EMNLP） | Amazon の商品レビュー，議会の法案・投票など 3 データ．SVM・ロジスティック回帰と多ドメイン学習（FEDA，MTRL など）．ドメインごとのクラスの偏りを再標本化で人工的に作った 4 版（cb1〜cb4） | 多ドメイン学習の得と，ドメイン ID を特徴に足しただけの基準（DOM-ID．ドメインごとの切片と同じ）の得 | 近い．"about half of the improvement seen in MDL is accounted for by a simple baseline of using the domain identifier as a feature"．残り半分はドメインごとの特徴の重み．ラベル無しで切片を当て直せるかは今後の問いとして提示 | ② | ◎（PDF 本文 4.2 節と結び） | https://aclanthology.org/D12-1119/ |
| Steyerberg ほか 2004（Statistics in Medicine 23） | 心筋梗塞後 30 日死亡．TIMI-II（3,339 人）で作ったモデルを GUSTO-I の 200 人（死亡 14）〜10,000 人（死亡 700）で更新し，別の患者で評価 | 切片や傾きの当て直しと，係数の推定し直し（縮小つき）の比較 | 段階として分けた．結論は「少ない標本では簡素な更新がよく，大きな改訂は大きな標本と縮小を組み合わせたときだけ」 | ①（元のモデルは別集団のデータ） | ○（抄録） | https://doi.org/10.1002/sim.1844 |
| Vergouwe ほか 2017（Statistics in Medicine 36．オンライン版 2016） | 前立腺がん・外傷性脳損傷・小児の発熱の 3 例 | 切片だけ／切片と傾き／係数の改訂を順に検定して選ぶ手続き | 段階として分けた．前立腺がんの例では更新の必要は「切片の違いだけ」から来ていた | ① | ○（抄録） | https://doi.org/10.1002/sim.7179 |
| Davis ほか 2019（JAMIA） | シミュレーションと退役軍人病院の 2 モデル（30 日死亡，急性腎障害） | 切片の補正／直線の当て直し／柔軟な当て直し／学び直しのうち，ブートストラップで選ばれる割合 | 段階として分けた．発生率が上がる場合は切片の補正が選ばれ（局所 1,000 人で 48%，5,000 人で 99.2%），予測因子と結果の関係が変わる場合は学び直しが選ばれた（96.4〜100%）．最小の標本では学び直しは判別を上げたが較正は上げなかった | ① | ◎（PMC 本文） | https://pmc.ncbi.nlm.nih.gov/articles/PMC6857513/ |
| Lee ほか 2023（ICLR．surgical fine-tuning） | 7 課題（CIFAR-C，ImageNet-C，Living-17，Entity-30，CIFAR-Flip，Waterbirds，CelebA），ResNet・CLIP ViT-B/16，少数のターゲットのラベル | どのブロックだけを学び直すとよいか | 分けていない．単位は層．出力側のずれ（ラベルの入れ替え，偽の相関）では最終層が最良．最終層は重みと切片をまとめて学ぶ | ① | ◎（PDF 本文 1・3 節） | https://arxiv.org/abs/2210.11466 |
| Zhu ほか 2023（NeurIPS．GLA） | CLIP，ImageNet と 11 データセット，少数ショット | ゼロショットの出力から，推定したクラスの偏りを引いた分の得．微調整したモデルの偏り | (a) だけ．16-shot のラベルで推定した偏りを引くと 11 データセットの平均で 59.1 → 60.7（+1.6）．微調整したモデルにも偏りが残る（ImageNet で末尾のクラスが先頭のクラスより 6.3 pt 低い） | ① | ◎（HTML 本文の表 5） | https://arxiv.org/abs/2310.08106 |
| Garg ほか 2023（ICML．RLSbench） | 14 データセット（画像・表・言葉），500 を超えるソースとターゲットの組，13 前後のドメイン適応の手法 | 各手法の正解率と，再標本化（RS）と事後の重み直し（RW）を足したときの差 | (a) に当たる操作の得だけ．クラスの割合のずれが大きいと RW などで 2〜10 pt 上がり，ずれが無いと 0.5 pt 未満．残りの誤りの内訳は測っていない．オラクルは早期終了の基準だけで，真の割合による補正は報告していない | ③ | ◎（PDF 本文 4 節） | https://arxiv.org/abs/2302.03020 |
| DART（arXiv 2411.15204．著者は未確認） | CIFAR-10C-LT など，テスト時の BN の統計の入れ替え（BNAdapt）の上に出力の補正を学ぶ | ロジットに行列 W をかける補正と，ベクトル b を足す補正（切片）の分解の得 | 近い量．BNAdapt の 79.0／67.0（不均衡比 10／100）を，W と b の両方で 84.7／85.1 に上げた．W だけと b だけの値は表の抽出が崩れていたため読み取りの推定（b だけで約 80.7，W だけで約 83.0．不均衡比 100）で，要確認 | ③ | ○（HTML 本文の抽出，表は崩れていた） | https://arxiv.org/abs/2411.15204 |
| Cai, Namkoong, Yadlowsky 2023（DISDE） | Adult（州の間），FMoW-WILDS（年代の間），ERM と DANN | 損失の変化を「X のずれ」「Y|X のずれ」「まれな X」の 3 項に分ける | 分けていない（データ側の分解）．DANN は表現の X のずれの項を減らしたが Y|表現 の項を増やし，全体では ERM と同じだった | ③（DANN） | ◎（HTML 本文） | https://arxiv.org/abs/2303.02011 |
| Zhang, Singh, Ghassemi, Joshi 2023（ICML） | 合成，半合成，eICU，Camelyon17，CelebA | 成績の変化を P(Y) などの分布ごとに Shapley 値で割り振る | 分けていない（データ側の分解） | なし（診断だけ） | ◎（冒頭と 5 節の見出し） | https://proceedings.mlr.press/v202/zhang23ai.html |
| Zhao ほか 2021（ICML．Calibrate Before Use） | GPT-2・GPT-3 の文脈内学習，AGNews など | 中身の無い入力への出力で求めた偏りを引く補正の得 | (a) だけ．補正は W = diag(p_cf)⁻¹，b = 0（対数の上でクラスごとの切片）．GPT-3 2.7B の AGNews で差は 0-shot 18.5，1-shot 26.6，4-shot 27.8，8-shot 21.9 pt で，例示を増やしても単調には縮まない | ③に近い（ラベル無し） | ◎（本文 5 節と表 1） | https://arxiv.org/abs/2102.09690 |
| Alexandari, Kundaje, Shrikumar 2020（ICML） | MNIST，CIFAR-10/100，糖尿病網膜症 | EM によるラベルシフトの補正と較正の方式 | (a) の内部．較正にクラスごとの切片を入れた方式が最もよい | ③ | ○（抄録） | https://arxiv.org/abs/1901.06852 |
| Šipka, Šulc, Matas 2022（WACV） | 細粒度の画像分類 | クラスの割合の推定と出力の補正の方式 | (a) だけ | ③ | ○（抄録） | https://arxiv.org/abs/2106.11695 |
| Chen, Wei, Kumar, Ma 2020（NeurIPS） | 線形分類，偽の特徴がガウスで本来の特徴が対数凹の混合．半合成の CelebA・MNIST | ラベル無しのエントロピー最小化が偽の特徴への重みを消すか | 理論で (b) を直す条件を示した（初期の分類器がある程度正しいこと） | ③ | ○（抄録） | https://arxiv.org/abs/2006.10032 |
| Oymak & Gulcu 2021（AISTATS） | 2 成分ガウス混合での自己学習 | 自己学習の反復の有限標本の振る舞い | 理論．向きが混合の中心に揃う条件と正則化の役割 | ③ | ○（抄録） | https://arxiv.org/abs/2006.11006 |
| Elkan & Noto 2008（KDD） | 正例とラベル無しから学ぶ分類 | 学んだ確率と真の確率の関係 | 理論．両者は定数倍しか違わない（順位は同じで閾値だけずれる） | ④に近い（弱いラベル） | ○（抄録） | https://cseweb.ucsd.edu/~elkan/posonly.pdf |
| Kull & Flach 2015（ECML PKDD） | 適切なスコア規則 | 損失を「調整（adjustment）の損失」「調整後の較正の損失」「まとめ方の損失」「消せない損失」に分ける | 尺度だけ．調整は出力に定数を足して平均を合わせる操作 | なし | ○（抄録） | https://doi.org/10.1007/978-3-319-23528-8_5 |
| Berta, Holzmüller, Jordan, Bach 2025（arXiv） | 画像・表など | 当て直しの後に残る損失（refinement）と当て直しで減る損失（calibration）の推定 | 尺度だけ．当て直しは refinement を変えない | なし | ○（抄録と講演資料） | https://arxiv.org/abs/2501.19195 |
| Kirichenko ほか 2023（ICLR．DFR），Izmailov ほか 2022（NeurIPS） | Waterbirds，CelebA，FMoW など | 表現を固定し最終層だけを学び直したときの最悪群の正解率 | 分けていない（表現と最終層） | ① | ○（抄録と二次資料） | https://arxiv.org/abs/2204.02937 ， https://arxiv.org/abs/2210.11369 |
| LaBonte ほか 2023（NeurIPS） | 同上 4 データ | 最終層の学び直しでのクラスの釣り合わせの寄与 | 分けていない．クラスの釣り合わせだけで DFR の最悪群の正解率の 94%（無しで 76%） | ① | △（検索の断片） | https://proceedings.neurips.cc/paper_files/paper/2023/file/265bee74aee86df77e8e36d25e786ab5-Paper-Conference.pdf |
| Rosenfeld, Ravikumar, Risteski 2022（DARE） | DomainBed など | 凍結特徴の上の線形の頭．ドメインごとの白色化 | 分けていない．超過リスクを平均の誤差と残差に分ける証明 | ③（ドメインごとの統計） | ◎（過去） | https://arxiv.org/abs/2202.06856 |
| Kang ほか 2020（ICLR．decoupling） | 長い裾のクラス分布の画像分類 | 分類器の学び直し（cRT），重みのノルムの正規化（τ-norm） | 分けていない．重みのノルムの偏りを直す | なし（訓練データの割合） | ○（抄録と二次資料） | https://arxiv.org/abs/1910.09217 |
| Burns & Steinhardt 2021（CVPR） | 画像の破損など | BN の統計の合わせ込みが効くずれの範囲 | 分けていない | ③ | ○（抄録） | https://arxiv.org/abs/2103.05898 |
| FPS（arXiv 2508.18693） | 教師なしドメイン適応 | 凍結した特徴の上の線形の頭を動かす | 分けていない．ΔW と Δb を同時に学ぶ．「ずれによる低下は特徴の劣化ではなく判別の境界のずれから来る」という主張は両方のラベルで学んだ線形プローブを上限に使った確認 | ③ | ◎（HTML 本文） | https://arxiv.org/abs/2508.18693 |
| WhyShift（Liu, Wang, Cui, Namkoong 2023，NeurIPS） | 表データ 5 種，86,000 の設定 | どの種類のずれが多いか | 分けていない．Y|X のずれが最も多い | なし | ○（抄録） | https://arxiv.org/abs/2307.05284 |
| MFH（Xue ほか 2023，ICLR） | 6 つのマルチモーダルデータ | クロスモーダル蒸留の成否と，モダリティに共通の決め手の特徴の割合 | 分けていない | ④ | ○（抄録） | https://arxiv.org/abs/2206.06487 |
| DebiasPL（Wang ほか 2022，CVPR），Frolic（2024，NeurIPS） | CLIP の疑似ラベルとゼロショット | 疑似ラベルとゼロショットの出力のクラスの偏りを消す | (a) の補正の提案だけ | ④に近い（CLIP の予測）／③ | ○（抄録） | https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Debiased_Learning_From_Naturally_Imbalanced_Pseudo-Labels_CVPR_2022_paper.html ， https://arxiv.org/abs/2410.19294 |
| MASH-Bench（arXiv 2608.22460） | 銃撃事件の危険度の分類 | 真の割合（オラクル）による事前確率の補正 | (a) を試し，取り戻せなかったと報告 | ③のオラクル | △（検索の断片） | https://arxiv.org/abs/2608.22460 |

---

## 3. 最も近い 4 本との比較

### 3.1 Johnson, Pollard, Naumann 2018

- 同じ点：ターゲット（1 病院）の少数のラベル（①）と，ほかのドメイン（45 病院）のラベル付きデータ（②）を，同じ評価の患者で比べた．局所のラベル数を 200〜2,400 人で振った．凍結に近い線形モデル（ロジスティック回帰）である．当て直しは向きを変えないことを明記した．「局所のラベルは較正（切片側）を直し，向き側（判別）は他病院のデータに及ばない」という分業の観察は，MER-DA の問いと同じ形である．
- 違う点：二値で，指標は AUROC と SMR である．多クラスの UAR で「切片の当て直しで取り戻せる分」を測ってはいない．局所のモデルが向きも 0.614 → 0.796 と直していく様子は報告しているが，切片側と向き側の割合には分けていない．③は標準化の 1 操作だけ，④は無い．評価は 1 病院だけである．

### 3.2 Joshi, Dredze, Cohen, Rosé 2012

- 同じ点：ほかのドメインのデータを混ぜる（②）得を，ドメインごとの切片（DOM-ID）で再現できる分とそれ以外に分けた．実データで，線形分類器である．
- 違う点：クラスの偏りは再標本化で人工的に作った．ターゲットのドメインにもラベルが多くある多ドメイン学習で，ラベル数を振っていない．①③④と比べていない．得の分解で，誤りの分解ではない．

### 3.3 Garg ほか 2023（RLSbench）と DART

- 同じ点：ラベル無しのターゲット（③）を使う手法の上に，出力の切片（クラスごとの重み直し，またはロジットに足すベクトル b）を足した分の得を測った．DART は b と W を分けて足した．
- 違う点：③の手法そのものが切片側と向き側のどちらを直したかは測っていない．切片の補正は「③で推定した割合」で行い，ターゲットのラベルによるオラクルの切片で取り戻せる上限とは比べていない．①②④との比較は無い．

### 3.4 医療の予測モデルの更新（Steyerberg 2004，Vergouwe 2017，Davis 2019）

- 同じ点：新しい施設の少数のラベル（①）で，切片だけ／切片と傾き／係数の推定し直しを比べ，ラベル数が少ないと切片側の更新が選ばれることを示した．Davis は「発生率の変化には切片，予測因子と結果の関係の変化には学び直し」とずれの種類と更新の段階を対応させた．
- 違う点：情報源は①だけである（元のモデルは別集団で作ったものなので②を前提にはしている）．ラベル無しや別の見方の予測は扱わない．二値の臨床課題である．

---

## 4. 論文の書き方への含意（判定の材料として）

ここに書くのは，見つかった先行研究からそのまま言えることだけである．新しい案ではない．

1. 切片と向きの分け方は，医療統計の「判別と較正」「切片だけの当て直し → 改訂」の段階として既知である．論文では Steyerberg 2004 と Vergouwe 2017 を引き，「この分け方を情報源の比較に使う」と書く必要がある．「分け方を新しく導入した」とは書けない．
2. 「少数のラベルは切片側を直し，他のドメインのデータは向き側を支える」という観察は，二値の医療データで Johnson 2018 に先例がある．MER-DA でこの点を書くときは，Johnson 2018 を先例として引き，多クラス・UAR・クラスあたり K 本・③④を含めた点を差として書く必要がある．
3. 「他ドメインを混ぜる得の一部はドメインごとの切片で説明できる」は Joshi 2012 に先例がある（約半分）．
4. 「ラベル無しで切片を当て直す」得の大きさは RLSbench・DART・GLA・Calibrate Before Use に先例がある．「ラベル無しは向きを直さない」を一般則として書くと，Chen 2020 の理論と食い違う．MER-DA の実測として，条件（凍結した特徴，線形の頭，感情の重なり）を付けて書く必要がある．
5. 次は計算による注釈で，文献にある主張ではない．二値の AUROC が切片に依らないのと同じ理由で，多クラスでも 2 つのクラスのロジットの差の順位（その 2 クラスに限った AUROC）はクラスごとの切片を変えても変わらない．この量を使えば，医療統計の判別の尺度と MER-DA の向き側の誤りを同じ言葉でつなげられる可能性がある．

---

## 5. 検索語の記録

ウェブ検索は 65 回行った．66 回目で，このセッションの全エージェントで共有する検索の上限（200 回）に達し，それ以降は検索できなかった．上限に達した後は，すでに見つけた原文の WebFetch による確認だけを行った．

| # | 検索語（英語） | 見つかったもの | 見つからなかったもの |
|---|---|---|---|
| 1 | decompose accuracy drop under distribution shift into label shift and conditional shift contributions empirical | DISDE，Zhang 2023，SEES（Chen 2022） | 分類器の切片と重みへの分解 |
| 2 | model updating small validation sample intercept recalibration logistic recalibration model revision sample size comparison Steyerberg | Steyerberg 2004，Vergouwe 2017，van Calster 2020 | ― |
| 3 | surgical fine-tuning which layer to tune depends on type of distribution shift label shift last layer | Lee 2023 | 切片だけの学び直し |
| 4 | retrain only the bias of the last layer under distribution shift recovers accuracy "bias-only" classifier recalibration versus retraining weights | DFR，LaBonte 2023 | 切片だけと重みの学び直しの比較 |
| 5 | what does batch normalization adaptation fix under label shift class prior analysis test-time adaptation non-i.i.d. class imbalance batch statistics bias | 2402.04958（チャネル選択の正規化），ROID，DART | BN の適応が切片側か向き側かを測った研究 |
| 6 | RLSbench relaxed label shift oracle label proportion correction re-weighting gains decomposition domain adaptation methods | RLSbench | 真の割合による補正 |
| 7 | generalized logit adjustment foundation model label bias zero-shot CLIP correction compared to few-shot fine-tuning | GLA，AMU-Tuning | ショット数をそろえた比較 |
| 8 | CLIP zero-shot class prior bias correction unlabeled data label bias estimation improves accuracy comparable to few-shot linear probe | Frolic | 線形プローブとの比較 |
| 9 | contextual calibration in-context learning majority label bias correction compared with increasing number of shots gain diminishes | Calibrate Before Use，Batch Calibration，2410.02210 | ― |
| 10 | "oracle" class prior correction how much of the domain gap is closed by adjusting class priors remaining gap feature shift image classification | MASH-Bench，2609.25647 | 画像分類での割合 |
| 11 | decision threshold versus decision direction error decomposition linear classifier transfer few-shot "threshold" "direction" | Luthra 2026（方向の分散），2607.06936，2607.08843 | 閾値と方向への誤りの分解の実測 |
| 12 | Burns Steinhardt limitations of post-hoc feature alignment for robustness CVPR 2021 findings | Burns & Steinhardt 2021 | ― |
| 13 | Steyerberg 2004 "Validation and updating of predictive logistic regression models" recalibration in the large intercept results | 同論文の書誌 | 本文の数値 |
| 14 | calibration drift over time discrimination stable recalibration intercept sufficient clinical prediction model dataset shift | 較正の時間変化の研究群 | ― |
| 15 | "label shift" "covariate shift" test-time adaptation decouple distribution tracker ODS open-world data shift Zhou 2023 ablation | ODS，Label Shift Adapter | 成分ごとのアブレーションの数値 |
| 16 | post-hoc logit adjustment versus classifier retraining comparison long-tailed decoupling representation classifier tau-normalization bias analysis | Kang 2020，2403.00250，Prior2Posterior | 同じ特徴での直接比較 |
| 17 | "discrimination" versus "calibration" external validation multicenter heterogeneity intercept versus predictor effects individual participant data meta-analysis random intercept Debray | Snell 2016 | 切片と予測因子の効果の異質性の分解の本文 |
| 18 | self-training avoids using spurious features under domain shift linear classifier theory Chen Wei Kumar Ma direction | Chen 2020 | ― |
| 19 | test-time adaptation gains explained by output adjustment only LAME parameter-free online test-time adaptation Boudiaf comparison feature adaptation | LAME | 出力だけの補正と特徴の適応の得の分解 |
| 20 | entropy minimization test-time adaptation analysis what changes classifier bias marginal prediction distribution "ZERO" frustratingly easy test-time adaptation vision-language models | ZERO | ― |
| 21 | test-time distribution normalization contrastive vision-language subtract mean embedding unlabeled zero-shot improvement Zhou NeurIPS 2023 | DN | ― |
| 22 | Janssen Moons Kalkman Grobbee Vergouwe 2008 updating methods improved performance clinical prediction model new patients intercept adjustment results | Janssen 2008 の書誌 | 本文の数値 |
| 23 | Kull Flach novel decompositions of proper scoring rules adjustment loss calibration loss refinement loss | Kull & Flach 2015 | 本文（出版社で認証が要る） |
| 24 | refinement error calibration error decomposition distribution shift post-hoc recalibration recovers how much "refine, then calibrate" | Berta 2025 | ずれのもとでの適用 |
| 25 | "label shift" versus "concept shift" which dominates real-world distribution shifts empirical study tabular WhyShift Y\|X shift | WhyShift，TableShift | ― |
| 26 | few-shot linear probe logistic regression bias term ablation class prior estimation error dominates few-shot classification | Firth の偏り補正（Ghaffari ほか） | 切片の項のアブレーション |
| 27 | Joshi Dredze Cohen Rosé "Multi-Domain Learning: When Do Domains Matter?" class bias label distribution domain-specific | Joshi 2012 | ― |
| 28 | Alexandari Kundaje Shrikumar maximum likelihood bias-corrected calibration hard to beat label shift adaptation class-specific bias terms vector scaling | Alexandari 2020 | ― |
| 29 | Vergouwe Nieboer Oostenbrink closed testing procedure updating prediction models arXiv full text recalibration in the large | Vergouwe 2017，Davis 2019 | 本文（PDF のリンクが切れていた） |
| 30 | bias-only fine-tuning BitFit why does tuning only bias terms work analysis domain shift class prior | BitFit | ずれやクラスの割合との関係の分析 |
| 31 | Izmailov feature learning in the presence of spurious correlations decompose worst-group error feature quality versus last layer classifier | Izmailov 2022 | ― |
| 32 | domain adaptation error decomposition representation shift versus classifier shift oracle linear probe target labels "feature" "classifier" contribution analysis | Wu 2020（2004.10390），Tachet des Combes 2020 | 線形プローブのオラクルによる分解 |
| 33 | Blum Mansour 2017 efficient co-training linear separators weak dependence empirical evaluation threshold adjustment | Blum & Mansour 2017，Diakonikolas 2024 | 実験 |
| 34 | cross-modal knowledge distillation analysis what transfers modality focusing hypothesis modality-general decisive features empirical | MFH | ― |
| 35 | pseudo-labels from teacher class-conditional noise threshold shift linear student classifier direction preserved empirical study | 2311.14237，2405.14088 | 生徒の向きが保たれ閾値だけずれることの実測 |
| 36 | test-time adaptation analysis prediction bias toward certain classes corrected accuracy gain attributable to class marginal correction rather than features | DART，DELTA，WACV 2026 の logit 調整，ADTE | 割合の補正と特徴の変化の得の分解 |
| 37 | arXiv 2405.14088 high-dimensional linear classifier class-conditional label noise Gaussian mixture optimal classifier fails intercept | El Firdoussi & Seddik（AISTATS 2026） | 切片についての記述 |
| 38 | Oymak Gulcu statistical algorithmic insights semi-supervised learning self-training Gaussian mixture pseudo-label direction bias | Oymak & Gulcu，Frei 2022 | ― |
| 39 | few labeled target examples estimate only class priors versus fine-tune classifier comparison label shift "few labels" prior adaptation sample efficiency | 2507.14661，RLLS，OTTER | ラベル数を振った直接比較 |
| 40 | Maity Dutta Terhorst Sun Banerjee linear adjustment posterior drift transfer learning few target samples intercept only adjustment experiments | Maity 2024 | 切片だけの版の実験 |
| 41 | "calibration-in-the-large" machine learning domain shift transfer hospitals deep learning recalibration intercept slope versus fine-tuning comparison | Johnson 2018 の手がかり，Mohta & Raffel 2021，Wald 2021 | ― |
| 42 | sample linear discriminant function expected error contribution from estimating the constant term versus the coefficient vector asymptotic expansion | Wyman 1990，広島大の技報 | 定数項と係数ベクトルの寄与を分けた原著の記述 |
| 43 | "label shift" "conditional shift" contribution decomposition target error ImageNet domain adaptation benchmark analysis Office-Home class imbalance how much of the gap | Tachet des Combes 2020，2507.20191 | Office-Home での数値の分解 |
| 44 | zero-shot CLIP versus few-shot linear probe "logit bias" analysis: few-shot methods mainly learn class bias AMU-Tuning findings | AMU-Tuning | ― |
| 45 | knowledge distillation student inherits teacher class prior bias logit adjustment distillation long-tailed teacher bias correction analysis | Niu 2022（IPWD），LTKD（Kim 2026） | 生徒の誤りの切片と向きへの分解 |
| 46 | weighted distillation with unlabeled examples teacher bias correction Iliopoulos 2022 class-conditional noise teacher pseudo-labels | Iliopoulos 2022 | ― |
| 47 | Elkan Noto learning classifiers from only positive and unlabeled data constant factor classifier differs only by scaling threshold empirical | Elkan & Noto 2008 | ― |
| 48 | batch norm statistics re-estimation equivalent to changing classifier bias logit shift analysis corruption robustness "mean" shift last layer | Benz 2021，DART | BN の統計の入れ替えと切片の補正の等価性の分析 |
| 49 | Castelli Cover relative value of labeled and unlabeled samples pattern recognition unknown mixing parameter labeled samples determine decision | Castelli & Cover 1996 | ― |
| 50 | mean centering features per domain linear classifier transfer removes bias analysis "feature centering" domain shift frozen embeddings few-shot cross-domain | FPS，STR，Tao の分布の補正 | ドメインごとの中心化が何を直すかの分析 |
| 51 | "prior shift" "decision boundary" rotation versus translation classifier adaptation analysis domain shift linear head offset rotation decomposition | Šipka 2022，BAIT | 境界の回転と平行移動の分解 |
| 52 | Kirichenko deep feature reweighting ablation retrain bias only class balancing last layer worst group recovered by class reweighting logit adjustment | LaBonte 2023，Hill 2025 | ― |
| 53 | "only the bias" tuned on target domain labels classifier weights frozen domain generalization oracle analysis accuracy recovered | 2507.17001 | ターゲットのラベルで切片だけを学ぶオラクルの分析 |
| 54 | Rosenfeld Ravikumar Risteski domain-adjusted regression ERM may already learn features sufficient out-of-distribution last layer oracle target retraining | DARE | ― |
| 55 | Debiased learning from naturally imbalanced pseudo-labels CLIP zero-shot pseudo-labels imbalance adaptive debiasing logit adjustment Wang CVPR 2022 | DebiasPL | ― |
| 56 | co-training empirical test conditional independence views pseudo-labels classifier bias class proportion drift threshold calibration multi-view | Pan & Woodard 2025（2510.07509，理論だけ） | 共訓練で閾値のずれを実測した研究 |
| 57 | cross-modal distillation student calibration class bias analysis teacher other modality pseudo-labels "label distribution" mismatch student threshold | MKE（Xue 2021），2403.11675 | 生徒の切片と向きを分けた研究 |
| 58 | dissecting test-time adaptation gains analysis 2025 which component matters normalization statistics versus entropy minimization versus prior correction controlled study | ReTTA など | 成分を分けた統制された研究 |
| 59 | "label shift" OR "prior shift" fraction of performance gap explained cross-dataset transfer frozen features linear probe study 2024 2025 "class prior" recalibration recovers | Heim 2023，OTTER，MASH-Bench | 凍結特徴での割合 |
| 60 | Heim "Towards Better Understanding of Domain Shift on Linear-Probed Visual Foundation Models" findings | Heim 2023（抄録だけ） | 本文 |
| 61 | Šipka Šulc Matas hitchhiker's guide prior-shift adaptation oracle known priors versus estimated priors results fine-grained classification | Šipka 2022 | 真の割合と推定した割合の数値 |
| 62 | Zhang Singh "Why did the model fail" attributing performance changes distribution shifts Shapley causal mechanisms P(Y) P(X\|Y) results real data | Zhang 2023 | ― |
| 63 | semi-supervised domain adaptation few target labels what do they fix analysis class prototypes alignment versus class prior empirical analysis SSDA | MME，DFA，ProMM，ECACL | 少数のターゲットのラベルが何を直すかの分析 |
| 64 | "distribution shift" "intercept" "slope" logit recalibration few labels transfer machine learning image text benchmark comparison full retraining sample size curve | 最小リスクの当て直し（NeurIPS 2023），2112.06674 | 画像・言葉での当て直しと学び直しのラベル数による比較 |
| 65 | multi-site prediction model pooled data versus local data local recalibration intercept discrimination improves with pooling calibration needs local data comparison study | Johnson 2018（eICU），Statistics in Medicine 2021 の合併データの論文 | ― |
| 66 | "internal-external cross-validation" heterogeneity calibration-in-the-large across clusters baseline risk intercept versus predictor effects Debray Riley 2013 framework | （上限のため実行されず） | ― |

---

## 6. 見つからなかったことと調査の限界

### 6.1 今回の検索の範囲で見つからなかったこと

- 実データで，①〜④のうち 2 つ以上を「切片の当て直しで取り戻せる分」と「残る分」の同じ分け方で比べた研究．
- ③（BN の統計の入れ替え，平均を引く，テスト時適応，ドメイン適応，自己学習）が切片側と向き側のどちらを減らしたかを，ターゲットのラベルによる切片の当て直しを基準にして測った研究．RLSbench と DART は③の上に切片の補正を足した得だけを測った．
- ④（クロスモーダル蒸留，共訓練，疑似ラベル）で学んだ分類器の誤りを切片側と向き側に分けた実データの研究．理論（Blum & Mansour 2017，Menon 2015，Elkan & Noto 2008）は「向きは保たれ閾値がずれる」を予言するが，それを実データで確かめた研究は見つからなかった．
- 多クラスで，切片の当て直しが取り戻す UAR（均衡正解率）を測った研究．医療の研究は二値で AUROC と較正を使う．
- 切片側と向き側の誤りを，ターゲットのラベル数 K の関数として分けて測った研究．Johnson 2018 と Steyerberg 2004 はラベル数を振ったが，誤りを割合に分けてはいない．
- 線形判別の誤り率の展開で，定数項（切片）と係数ベクトル（向き）の寄与を分けて書いた原著の記述．過去の調査の計算どおり「係数から分けられる」と書くのが安全である．

### 6.2 調査の限界

- 検索の上限に達したため，次の検索は行えなかった．医療の多施設のデータでの切片と予測因子の効果の異質性（Debray 2013，Riley 2016 の内部外部交差検証），時間によるずれでの判別と較正の変化（Guo ほか 2022，Nestor ほか 2019），LP-FT（Kumar ほか 2022），共訓練の実証（Nigam & Ghani 2000），BitFit の分析，ラベル無しの中心化の分析（SimpleShot），Lee ほか 2021 の理論．医療の多施設の研究に，①②を判別と較正に分けたものがさらにある可能性が高い．
- Steyerberg 2004，Vergouwe 2017，Janssen 2008 は出版社の本文を読めず，抄録だけで判定した．Kull & Flach 2015 も抄録だけである．
- DART の表 3 は抽出が崩れていたため，b だけと W だけの値は読み取りの推定である．引用する前に原文の表で確かめる必要がある．著者名も未確認である．
- Johnson 2018 の掲載先（NeurIPS の ML4H ワークショップかどうか）は確かめていない．arXiv の版（1812.02275v1）で読んだ．
- 取得した PDF は調査のセッションの一時領域（scratchpad）にあり，セッションが終わると消える．
