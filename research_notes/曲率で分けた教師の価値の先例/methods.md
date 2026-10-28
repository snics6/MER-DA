# 曲率で分けた教師の価値：方法の先例と M1〜M4 の近さの判定（2026-10-28）

この記録は，提案中の手法案 M1〜M4 がすでに発表されているかを調べるために，曲率や部分空間で重みを変える正則化・射影，ラベル無しデータ上の一致から精度を推定する方法，半教師あり DA の方法を原典で確かめたものである．論文（`paper/`）と `docs/` は編集していない．

**比べる相手（本研究の手法）．**凍結した特徴の上の線形 softmax の頭 θ を学ぶ．ソースで学んだ錨 a0 がある．教師（別のモダリティか，音声・映像・言葉を融合したモデル）がラベル無しのターゲットの候補に軟らかい擬似ラベルを付ける．ターゲットのラベルはクラスあたり K 本（K=0〜25）．手順は 2 段である．(1) 擬似ラベルに a0 への L2 罰則をつけて θ_Q を解く．(2) K 本のラベルに θ_Q への L2 罰則をつけて解き直す．解き直しは，教師が動かした量 θ_Q − a0 のうち，一般化固有方向 j（K 本のラベルの損失のヘッセ行列 H_K と罰則の精度行列 P の対）に沿った成分の 1/(1+λ_j) を残す（研究者が検証済み）．λ_j が小さい方向は K 本のラベルで確かめられない方向で，K > 0 での教師の価値はこの方向にある．

**手法案．**(M1) 擬似ラベルの重み（教師への信頼）を，パラメータ空間の方向ごとに λ_j の関数として決める．(M2) ラベルで確かめられない方向での教師の信頼度を，複数の教師・視点・モダリティ（音声，映像，言葉，融合）のラベル無しデータ上の食い違いから推定し，方向ごとの信頼に使う．(M3) ラベルで確かめられる方向（λ が大きい方向）で測った教師の誤差を，確かめられない方向へ移す（方向をまたいだ経験ベイズ・ランダム効果の仮定）．(M4) 残しておいたソースのコーパスを仮のターゲット（ラベルあり）にして，方向ごとの信頼をターゲットのラベルなしで選ぶ．

**調べ方．**書誌は OpenAlex の API と Crossref の API（DOI から巻・頁・著者），arXiv の API（抄録・コメント欄）で確かめた．Ildiz ほか 2024，PPI++，PSPA，Mobahi ほか 2020，EATA，Pre-Train Your Loss，MetaReg は PDF を curl で取って `pdftotext` で文字にし，該当箇所を grep して読んだ．Kakade & Foster 2007 は本文の PDF を取れず，共著者 Foster の講演スライドの文字で中身を確かめた．dblp と OpenReview は bot 対策の頁が返り，使えなかった．IEEE Xplore・ICASSP・Interspeech の論文集は直接は検索していない（ウェブ検索のみ）．

**確かめた水準の記号．**◎＝本文の該当箇所を読んだ，○＝抄録か一次の頁（著者のスライドを含む）で確かめた，△＝書誌のみ確かめた（中身は記憶か題から），✕＝書誌も一次の頁で確かめていない．

**重み付けの単位の区別．**以下では，重みが何に付くかを次の 4 つに分けて書く．「標本ごと」（発話 1 本ずつの確信度など），「分類器・教師ごと」（教師 1 つに 1 つの数），「座標ごと」（パラメータの成分 1 つずつ．対角の重み），「方向ごと」（パラメータ空間の一般の方向・部分空間ごと．固有ベクトルの基底で重みを変える）．M1〜M4 が要るのは最後の「方向ごと」で，ラベルの曲率から方向を決めるものである．

---

## Q1. 曲率や部分空間で重みを変える正則化・射影の方法は，ラベル付きデータの曲率から教師・擬似ラベルへの方向ごとの信頼を決めているか

### Takeaway
調べた範囲では，K 本のラベルの曲率（ヘッセ行列やフィッシャー情報行列）の固有方向に応じて，教師や擬似ラベルへの信頼を方向ごとに決める方法は見つからなかった．方向ごと・座標ごとに罰則の重みを変える仕組みそのものは，Fisher merging，RegMean，重みつきリッジ（Wu & Xu 2020），EWC，零空間への射影（OGD，GPM，Adam-NSCL，AlphaEdit）に既にある．ただしこれらで重みの出所になる曲率は，錨や古い課題の側の曲率か，新しい課題のデータの共分散であり，教師の擬似ラベルへの信頼を決めるためには使われていない．研究者が検証した「解き直しは 1/(1+λ_j) を残す」という関係は，2 つのガウス分布を精度行列で重みづけて平均する式（Fisher merging やベイズの事後平均と同じ式）であり，式としては新しくない（本ノートの導出．下の Inferences）．

### Cited Findings

**表 1．曲率・部分空間で重みを変える方法と M1〜M4 との関係**

| 研究 | 方向・座標の重みの出所 | 何を正則化・射影するか | 教師・擬似ラベル | 少数のターゲットラベル | 重みの単位 | 判定 |
|---|---|---|---|---|---|---|
| EWC（Kirkpatrick ほか 2017）○ | 古い課題のフィッシャー情報（対角） | 古い課題の解への 2 次の罰則 | なし | なし | 座標ごと | M1：部分的に重なる（曲率で罰則の重みを座標ごとに変える．曲率は錨の側） |
| L2-SP（Li, Grandvalet, Davoine 2018）○ | なし（等方） | 事前学習の重みへの L2 罰則 | なし | 少数のラベルで微調整 | 一様 | M1：違う．本研究の 2 段の各段の罰則そのもの |
| DELTA（Li ほか 2019）○ | 教師ありで学んだ注意（特徴マップの選択） | 元のネットワークとの外側の層の出力の差 | なし（元のネットワークの出力をそろえる） | ターゲットのラベルで注意を学ぶ | 特徴マップ（チャネル）ごと | M1：部分的に重なる（ターゲットのラベルで，錨へそろえる強さをチャネルごとに決める） |
| Fisher merging（Matena & Raffel 2022）○ | 各モデルのフィッシャー情報 | 複数モデルのパラメータの重みつき平均 | なし | なし | 座標ごと（対角近似） | M1：部分的に重なる（曲率で方向ごとの信頼を決める規則そのもの．擬似ラベルと少数ラベルは無い） |
| RegMean（Jin ほか 2023）○ | 各モデルの入力のグラム行列 | 線形層の重みの閉じた形の統合 | なし | なし | 方向ごと（グラム行列の行列） | M1：部分的に重なる（線形層では 2 乗損失の曲率で方向ごとに重みづける） |
| OGD（Farajtabar ほか 2020）○ | 古い課題の出力の勾配が張る部分空間 | 新しい課題の勾配を直交補空間へ射影 | なし | なし | 部分空間ごと（0 か 1） | M1：部分的に重なる（データで決まる部分空間で更新を分ける） |
| GPM（Saha ほか 2021）○ | 古い課題の活性の SVD | 勾配を重要な部分空間の直交方向へ射影 | なし | なし | 部分空間ごと（0 か 1） | 同上 |
| Adam-NSCL（Wang ほか 2021）○ | 古い課題の入力特徴の非中心化共分散の SVD | 更新を近似の零空間へ射影 | なし | なし | 部分空間ごと（0 か 1） | 同上 |
| AlphaEdit（Fang ほか 2025）○ | 保持する知識の零空間 | 編集の摂動を零空間へ射影 | なし | なし | 部分空間ごと | 同上 |
| Bulk-SGD（Zakarin & Singh 2025）○ | ヘッセ行列の上位の固有空間（Dominant） | 更新を Dominant の直交補空間（平らな方向）に限る | なし | なし | 部分空間ごと | M1：部分的に重なる（曲率の小さい方向だけを動かす．目的は学習の速さ） |
| EATA（Niu ほか 2022）◎ | 擬似ラベルを付けた試験標本でのフィッシャー情報（対角） | 元のモデルの重みへの罰則 | 元のモデルの硬い擬似ラベル | なし | 座標ごと | M1：部分的に重なる（擬似ラベルから曲率を作り，座標ごとの罰則に使う．罰則は錨へ向き，教師への信頼ではない） |
| Pre-Train Your Loss（Shwartz-Ziv ほか 2022）◎ | ソースの事後分布の共分散 | ソースの事後分布を事前分布にした罰則 | なし | 下流の課題のラベル | 方向ごと | M1：部分的に重なる（錨への罰則を方向ごとに変える．重みの出所はソースの側） |
| Wu & Xu 2020 ○ | データの共分散 Σ_x と係数の事前共分散 Σ_β | 重みつきリッジ (XᵀX + λΣ_w)⁻¹ | なし | なし | 方向ごと（行列 Σ_w） | M1：部分的に重なる（最適な重みの行列を両方の共分散から決める．錨は 0） |
| Hoerl & Kennard 1970 △ | （記憶：正準形の各方向に別のリッジ定数） | リッジ推定 | なし | なし | 方向ごと（記憶） | M1：部分的に重なる（記憶．未確認） |
| Ildiz ほか 2024 ◎ | 生徒の特徴の共分散の固有値と標本数 | 最適な教師（代理モデル）の係数 | 代理モデルのラベルで学ぶ | なし | 方向ごと（固有方向） | M1：部分的に重なる（教師の価値を固有方向ごとに式で示す．信頼の重みではなく，最適な教師の形） |
| Mobahi ほか 2020 ◎ | カーネル（グラム行列）の固有値 | 自己蒸留を繰り返したときの解 | 自分の予測 | なし | 方向ごと（固有方向） | 仕組みの式が部分的に重なる．M1〜M4 は違う |
| Dong ほか 2025 ○ | 弱い教師と強い生徒の特徴の部分空間 | （方法ではなく解析） | 弱い教師の擬似ラベル | なし | 部分空間ごと | 仕組みの解析が部分的に重なる．M1〜M4 は違う |
| LP-FT（Kumar ほか 2022）○ | （解析） | 頭だけを学んでから全体を微調整 | なし | ラベルあり | — | 仕組みの解析が部分的に重なる（下の注）．M1〜M4 は違う |

各行の根拠は次のとおりである．

- EWC は，古い課題にとって重要な重みの学習を選択的に遅くすることで古い課題を覚えておく（"selectively slowing down learning on the weights important for those tasks"）．PNAS 114(13):3521–3526，2017 — [arXiv 1612.00796](https://arxiv.org/abs/1612.00796)，[Crossref の DOI](https://doi.org/10.1073/pnas.1611835114)
- L2-SP は，微調整で初期のモデルに近づける正則化を比べ，事前学習のモデルを基準にした単純な L2 罰則を推す（"recommend a simple L^2 penalty with the pre-trained model being a reference"）．ICML 2018 — [arXiv 1802.01483](https://arxiv.org/abs/1802.01483)
- DELTA は，重みではなく外側の層の出力をそろえ，そろえる特徴マップを教師ありで学んだ注意で選ぶ（"constraining a subset of feature maps that are precisely selected by attention that has been learned in a supervised learning manner"）．ICLR 2019 — [arXiv 1901.09229](https://arxiv.org/abs/1901.09229)
- Fisher merging は，各モデルの事後分布をフィッシャー情報を精度行列とするガウス分布で近似し，同時の尤度を最大にするパラメータを選ぶ（"approximate each model's posterior as a Gaussian distribution whose precision matrix corresponds to its Fisher information"）．NeurIPS 35，17703–17716 頁 — [arXiv 2111.09832](https://arxiv.org/abs/2111.09832)，[Crossref](https://doi.org/10.52202/068431-1287)
- RegMean は，統合したモデルと個々のモデルの予測の差を最小にする重みでパラメータ空間で統合する．ICLR 2023 — [arXiv 2212.09849](https://arxiv.org/abs/2212.09849)（グラム行列を使う閉じた形は本文の内容で，記憶．抄録には「予測の差を最小にする重み」までしか書かれていない）
- OGD は，新しい課題の勾配を，古い課題での出力が変わらない部分空間へ射影する — [arXiv 1910.07104](https://arxiv.org/abs/1910.07104)（会議は AISTATS 2020 と記憶．未確認）
- GPM は，各課題の後に活性を SVD して重要な勾配の部分空間の基底を求め，その直交方向へ勾配を進める．ICLR 2021（arXiv の journal-ref 欄） — [arXiv 2103.09762](https://arxiv.org/abs/2103.09762)
- Adam-NSCL は，Adam が作る更新の候補を，それまでの全課題の近似の零空間へ射影する．零空間は入力特徴の非中心化共分散の SVD で求める．CVPR 2021，184–193 頁 — [arXiv 2103.07113](https://arxiv.org/abs/2103.07113)，[Crossref](https://doi.org/10.1109/CVPR46437.2021.00025)
- AlphaEdit は，編集の摂動を保持する知識の零空間へ射影してから加える．ICLR 2025（arXiv の journal-ref 欄） — [arXiv 2410.02355](https://arxiv.org/abs/2410.02355)
- Bulk-SGD は，ヘッセ行列の上位の固有空間（Dominant）の直交補空間に更新を限る．平らな方向への更新は収束を速めうるが安定性を損ないうると報告する — [arXiv 2505.11972](https://arxiv.org/abs/2505.11972)
- EATA は，試験時の適応で忘却を抑えるため，元のモデルが付けた硬い擬似ラベルで「擬似ラベル付きの ID 試験集合」を作り，その上で対角のフィッシャー情報を計算して，元の重みへの罰則の座標ごとの重みにする（本文 4.2 節．"we calculate the fisher importance of model weights" を pseudo-labeled ID test set の上で行う）．ICML 2022 — [arXiv 2204.02610](https://arxiv.org/abs/2204.02610)
- Pre-Train Your Loss は，ソースの課題の事後分布を下流の課題の事前分布にし，その事前分布の共分散が「パラメータ空間のどの方向が」重要かを示すことが大事だと述べる（本文 1 節．"represented with an informative covariance matrix, indicating which directions in parameter space"）．NeurIPS 35，27706–27715 頁 — [arXiv 2205.10279](https://arxiv.org/abs/2205.10279)，[Crossref](https://doi.org/10.52202/068431-2009)（共分散の作り方の詳細は読んでいない）
- Wu & Xu は，一般化（重みつき）リッジ (XᵀX + λΣ_w)† Xᵀy を，データの共分散 Σ_x と係数の異方的な事前共分散 Σ_β の下で解析し，最適な重みの行列 Σ_w をリッジなしと最適な λ の両方で決める（"we determine the optimal weighting matrix Σ_w"）．NeurIPS 2020（arXiv のコメント欄） — [arXiv 2006.05800](https://arxiv.org/abs/2006.05800)
- Hoerl & Kennard は，Technometrics 12(1):55–67，1970 — [Crossref](https://doi.org/10.1080/00401706.1970.10488634)．正準形の各方向に別のリッジ定数 k_i を置く「一般化リッジ」の記述はこの論文にあると記憶するが，本文で確かめていない
- Ildiz ほか は，代理モデル（教師）の出力をラベルにして目的のモデルを学ぶ蒸留をリッジなしの高次元回帰で解析し，最適な代理モデルの形を示す．系 1 では，最適な代理モデルの係数は目的の分布の共分散の固有値 λ_i と標本数で決まる ζ_i = τ/(λ_i+τ) だけで決まり，上位の固有方向では真の係数より大きく（増幅），裾の固有方向では小さく（縮小）なり，その切り替わりに閾値がある．共分散が cI のときに限り最適な代理モデルは真の係数に一致し，標本数が次元より多い領域（n > p）でも真の係数に一致する（命題 2）．W2S の文脈では，同じデータ量で弱い教師のラベルで学ぶほうが強いラベルで学ぶより良くなりうるが，データのスケーリング則は改善できないと述べる — [arXiv 2410.18837](https://arxiv.org/abs/2410.18837)（v2，2025-02-27．journal-ref 欄は空．ICLR 2025 と記憶するが未確認）
- Mobahi ほか は，自己蒸留を繰り返すと解を表す基底関数の数が段々減ることを示す．グラム行列を固有値分解し，各段で D(c_{t−1}I + D)⁻¹ という対角の因子が掛かって小さい固有値の成分が速く縮むと書く（本文 727 行目付近の文字）．NeurIPS 2020（OpenAlex の所在） — [arXiv 2002.05715](https://arxiv.org/abs/2002.05715)
- Dong ほか は，弱い教師の擬似ラベルで強い生徒を学ぶ W2S をリッジなし回帰で解析し，弱い教師の分散は生徒と教師の特徴の部分空間の共通部分 V_s ∩ V_w ではそのまま受け継がれ，食い違う部分空間 V_w \ V_s では dim(V_s)/N 倍に減ると示す（N は擬似ラベルの数）．ICML 2025（arXiv のコメント欄） — [arXiv 2502.05075](https://arxiv.org/abs/2502.05075)
- LP-FT は，全体の微調整が事前学習の特徴をゆがめて分布外で線形プローブより悪くなりうることを 2 層の線形ネットワークで示し，頭を先に学ぶ 2 段の手順を推す．ICLR 2022（arXiv のコメント欄） — [arXiv 2202.10054](https://arxiv.org/abs/2202.10054)（「学習データが張る部分空間の外では特徴が変わらない」という本文の議論は記憶で，未確認）
- Thompson ほか 2019 は，NMT の DA の微調整で EWC を使って忘却を抑える（題による．本文は未読）．NAACL 2019，2062–2068 頁 — [Crossref](https://doi.org/10.18653/v1/N19-1209)
- ウェブ検索で「ヘッセ行列の固有方向ごとの擬似ラベルの信頼」「ラベル付きデータの勾配の零空間に限った蒸留」「平らな方向だけを正則化する蒸留」を 3 通りの語で探したが，該当する論文は出なかった．出たのは標本ごとに擬似ラベルの信頼を決める ICI（[arXiv 2007.08461](https://arxiv.org/pdf/2007.08461)．検索結果の要約のみ），線形の蒸留で生徒の重みが教師の重みを学習データの張る空間へ射影したものになるという Phuong & Lampert 2019（[PMLR](http://proceedings.mlr.press/v97/phuong19a/phuong19a.pdf)．検索結果の要約のみ）などである — [ウェブ検索の結果（ICI，FTSP など）](https://arxiv.org/pdf/2007.08461)，[Subspace-Guided KD（WACV 2026．特徴の部分空間をそろえる別の意味）](https://openaccess.thecvf.com/content/WACV2026/papers/Hayder_Subspace-Guided_Knowledge_Distillation_for_Efficient_Model_Transfer_WACV_2026_paper.pdf)
- Sculpting Subspaces（Nayak ほか 2025）は，ヘッセ行列の固有ベクトルの代わりに各重み行列の SVD を使い，小さい特異値の部分空間に更新を限る継続学習の方法である（検索結果の要約のみ．一次の頁では未確認） — [arXiv 2504.07097](https://arxiv.org/abs/2504.07097)

### Inferences
- **1/(1+λ_j) の関係は既知の式と同じである．**解き直しの目的関数を 2 次で近似すると，K 本のラベルの損失のヘッセ行列を H，罰則の精度行列を P，罰則なしのラベルだけの解を θ_K として，θ_R ≈ (H + P)⁻¹(H θ_K + P θ_Q) になる．これは精度 H のガウス分布と精度 P のガウス分布の積の平均で，Fisher merging が 2 つのモデルを統合するときの式（[Matena & Raffel](https://arxiv.org/abs/2111.09832)）と同じ形である．θ_Q に対する θ_R の微分は (H+P)⁻¹P で，H v = λ P v の固有方向では 1/(1+λ) になる．したがって論文では，この縮小の因子を新しい結果として書かず，「解き直しは教師の解とラベルの解の Fisher merging（ベイズの事後平均）にあたる」と既知の式に結びつけて書くのが安全である．新しさは，この分解で教師の価値がどの方向にあるかを実測したことと，M1〜M4 の使い方の側に置く必要がある．
- **M1 の形（方向ごとの罰則の行列）だけでは新しさにならない．**P を H の固有値の関数にすることは，残差 θ − θ_Q に一般化リッジ（重みつきリッジ）をかけることと同じである．線形ガウスの模型では，教師の誤差 θ* − θ_Q の共分散 Σ_e が分かれば最適な P は σ²Σ_e⁻¹（ベイズの答え）になり，Wu & Xu の Σ_β を Σ_e に置き換えた問題になる（[Wu & Xu](https://arxiv.org/abs/2006.05800)．この置き換えは本ノートの推論）．したがって M1 が新しいかどうかは，「確かめられない方向での Σ_e をどう推定するか」（M2・M3・M4）にかかる．
- **零空間への射影の方法は，役割を入れ替えると M1 の硬い版になる．**Adam-NSCL・GPM・OGD は「古い課題のデータが張る方向を守り，新しい課題は零空間だけで動かす」．K 本のラベルを「守る課題」，教師の擬似ラベルを「新しい課題」と見なせば，「擬似ラベルによる更新を H_K の零空間（λ ≈ 0）だけに許す」手法になる．この組み合わせ（ラベルの曲率の零空間に教師の更新を限る）は見つからなかった．
- **EATA と DELTA は「何で重みを決めるか」が M1 と逆向きである．**EATA は擬似ラベルから曲率を作って錨（元のモデル）を守る．DELTA はターゲットのラベルで重要と分かったチャネルほど元のネットワークにそろえる（抄録の範囲での読み）．M1 は，ラベルで確かめられない方向ほど教師を信じる．
- **W2S の理論（Ildiz ほか，Dong ほか）は，教師の価値を固有方向・部分空間ごとに分ける解析としては本研究の解析に最も近い．**ただしどちらも少数のターゲットラベルによる解き直しを扱わず，方向ごとの信頼を選ぶ手順も示さない．本研究の「K 本のラベルで確かめられない方向に教師の価値がある」という実測は，Dong ほかの「食い違う部分空間で分散が減る」，Ildiz ほかの「裾の固有方向で教師を縮める」と比べて書くとよい．

### Gaps
- RegMean のグラム行列による式，OGD の会議名，Hoerl & Kennard の一般化リッジの記述，LP-FT の部分空間の議論は，本文で確かめていない．
- 2022〜2026 年の「低曲率の部分空間での微調整・蒸留」の論文は，ウェブ検索で 3 回探した範囲で，教師の信頼に曲率を使うものは無かった．OpenReview の全文検索は bot 対策で使えず，NeurIPS・ICLR の論文集を網羅的には見ていない．
- 「ヘッセ行列で重みづけた蒸留」「曲率を考えた擬似ラベル」という語の論文が 2025〜2026 年に出ている可能性は残る（見つからなかったことしか言えない）．

---

## Q2. ラベル無しデータ上の一致・食い違いから分類器や教師の精度を推定する方法は，パラメータの方向ごとの信頼に使われているか

### Takeaway
精度の推定（Platanios ほか，Jiang ほか，Chen ほか，Baek ほか，Parisi ほか）と共同学習の理論（Blum & Mitchell，Dasgupta ほか，Balcan ほか）は，分類器ごとか標本ごとの精度・誤りを推定するもので，パラメータの方向ごとの信頼には使っていない．マルチモーダルの感情認識でも，モダリティ間の一致は標本ごとの重みに使われている（TiCAL）．方向ごとに近いのは Kakade & Foster 2007 の多視点回帰だけで，2 つの視点のラベル無しデータの正準相関 λ_i から，相関の低い正準方向ほど強く縮めるリッジの罰則 Σ(1−λ_i)/λ_i β_i² を作る．これは「ラベル無しデータ上の視点間の一致から，方向ごとの正則化の強さを決める」点で M2 と重なる．ただし縮める先は 0 で教師ではなく，方向はラベルの曲率ではなく正準相関の方向である．

### Cited Findings

**表 2．一致・食い違いを使う方法の重みの単位**

| 研究 | 何を推定・使うか | 重みの単位 | 教師・擬似ラベル | 少数のラベル | 判定 |
|---|---|---|---|---|---|
| Kakade & Foster 2007 ○ | 2 視点の正準相関 λ_i から作るリッジの罰則 | 方向ごと（正準方向） | なし | ラベル付きでリッジ回帰 | M2：部分的に重なる（ラベル無しの視点間の一致で方向ごとの罰則を決める） |
| Sindhwani, Niyogi, Belkin 2005 △ | 視点間の予測の食い違いを罰する共正則化 | （線形なら 2 次形式．記憶） | なし | あり | M2：部分的に重なる（記憶．未確認） |
| Blum & Mitchell 1998 △ | 2 視点で互いに擬似ラベルを付ける共同学習 | 標本ごと | 互いの予測 | あり | 違う |
| Dasgupta, Littman, McAllester 2002 △ | 2 つの分類器の食い違いで誤りを上から抑える | 分類器ごと | 互いの予測 | — | 違う |
| Balcan, Blum, Yang △ | 拡張（expansion）の条件での共同学習の理論 | 分類器ごと | 互いの予測 | — | 違う |
| Platanios, Blum, Mitchell △ | 複数の関数の一致率から各関数の精度を推定 | 分類器ごと | — | なし | 違う |
| Platanios ほか 2016（ベイズ），2017（確率論理）△ | 同上の拡張 | 分類器ごと | — | なし | 違う |
| Parisi ほか 2014 △ | 分類器の予測の共分散行列（ランク 1 の構造）から順位付けと統合 | 分類器ごと（固有ベクトルの成分） | — | なし | 違う（スペクトルを使うが，分類器の空間） |
| Zhang ほか 2014 △ | クラウドソーシングの作業者の混同行列をスペクトル法＋EM で推定 | 作業者ごと | — | なし | 違う |
| Ratner ほか 2019 △ | 弱いラベル源の精度を一致の構造から推定 | ラベル源ごと | 弱いラベル | なし | 違う |
| Jiang ほか ○ | 別の SGD の実行との食い違い率で試験誤差を推定 | モデルごと | — | なし | 違う |
| Chen ほか 2021 ○ | 自己学習のアンサンブルで誤りの検出と精度の推定 | 標本ごと・モデルごと | 擬似ラベル | なし | 違う |
| Baek ほか 2022 △ | 一致率の直線（agreement-on-the-line）で分布外の精度を予測 | モデルごと | — | なし | 違う |
| TiCAL（Yin ほか 2026）○ | 単モーダルの擬似ラベルと典型度で標本ごとのモダリティ間の一致を推定 | 標本ごと | 単モーダルの擬似ラベル | 全ラベル | 違う |
| Albanie ほか 2018 △ | 顔の教師の確率を音声の生徒に蒸留 | 標本ごと（重みなし） | 顔の教師 | 評価のコーパスでは全ラベル | 違う |
| Dong ほか 2025 ○ | 教師と生徒の特徴の部分空間の食い違い | 部分空間ごと（解析） | 弱い教師 | なし | M2：仕組みの解析が部分的に重なる（複数の教師の食い違いではない） |
| PSPA（Miao ほか 2023）◎ | 予測と少数のラベルの共分散から座標ごとの信頼 ω_j | 座標ごと | 機械学習の予測 | あり（ラベルで信頼を決める） | M2：違う（ラベル無しの食い違いは使わない）．M1・M3 は Q4 |

- Kakade & Foster は，まずラベル無しデータでノルム（等価にカーネル）を学び，次にラベル付きデータでそのノルムのリッジ回帰を解く半教師ありの手法である．ノルムは CCA から導く．リッジ問題の内在的な次元は正準相関係数で書ける — [mlanthology の書誌と抄録](https://mlanthology.org/colt/2007/kakade2007colt-multi)，[Crossref（LNCS，82–96 頁）](https://doi.org/10.1007/978-3-540-72927-3_8)
- 同じ手法を共著者 Foster の講演スライドは「CCA-ridge regression is to minimize least squares plus a penalty of Σ (1 − λi)/λi βi²」「Large penalties in the less correlated directions」「λi's are the correlations」と書く．リスクの上界は (5α + Σλ_i²/n)σ² の形で，α は多視点の仮定の破れ（偏り），Σλ_i²/n は分散である — [Foster の講演スライド（PDF）](https://stat.ufl.edu/wp-content/uploads/sites/120/fosterTalk.pdf)（論文本文の PDF は取れず，罰則の式は本文では確かめていない）
- Foster, Kakade, Zhang の技術報告（TTI-C 2008）は，同じ冗長性の仮定の下で，縮小による方法と次元削減による方法が同程度に良いことを示す（検索結果の要約） — [ウェブ検索の結果（TTI-C TR-2008-4 の頁）](https://sham.seas.harvard.edu/sites/projects.iq.harvard.edu/files/kakade/files/ttic-tr-2008-4-2008.pdf)
- Sindhwani, Niyogi, Belkin "A Co-Regularization Approach to Semi-supervised Learning with Multiple Views"（2005．会議名は OpenAlex に無い．ICML 2005 の多視点学習のワークショップと記憶） — [OpenAlex の検索](https://api.openalex.org/works?search=A%20co-regularization%20approach%20to%20semi-supervised%20learning%20with%20multiple%20views)
- Blum & Mitchell，COLT 1998，92–100 頁 — [Crossref](https://doi.org/10.1145/279943.279962)
- Dasgupta, Littman, McAllester，Advances in NIPS 14（2002 刊），375–382 頁 — [Crossref](https://doi.org/10.7551/mitpress/1120.003.0053)
- Balcan, Blum, Yang "Co-Training and Expansion: Towards Bridging Theory and Practice"．OpenAlex の所在に NeurIPS がある（NIPS 17．2004 年の会議で 2005 刊と記憶） — [KiltHub の DOI](https://doi.org/10.1184/r1/6604544)
- Platanios, Blum, Mitchell "Estimating Accuracy from Unlabeled Data"．OpenAlex の所在に Uncertainty in Artificial Intelligence がある（UAI 2014 と記憶） — [KiltHub の DOI](https://doi.org/10.1184/r1/6605273)．続編は ICML 2016 のベイズ版と NeurIPS 2017 の確率論理版 — [arXiv 1705.07086](https://arxiv.org/abs/1705.07086)
- Parisi, Strino, Nadler, Kluger，PNAS 111(4):1253–1258，2014 — [Crossref](https://doi.org/10.1073/pnas.1219097111)（共分散行列のランク 1 の構造を使う中身は記憶）
- Zhang, Chen, Zhou, Jordan "Spectral Methods meet EM" — [arXiv 1406.3824](https://arxiv.org/abs/1406.3824)
- Ratner ほか "Training Complex Models with Multi-Task Weak Supervision"，AAAI 2019 — [Crossref](https://doi.org/10.1609/aaai.v33i01.33014763)
- Jiang, Nagarajan, Baek, Kolter は，同じ学習集合で SGD を別に走らせた 2 つのネットワークのラベル無し試験データでの食い違い率で試験誤差を推定できることを示す — [arXiv 2106.13799](https://arxiv.org/abs/2106.13799)（ICLR 2022 と記憶．OpenReview は bot 対策で確かめられなかった）
- Chen ほか は，ラベル無しの試験データで，誤分類された標本の検出と精度の推定を，自己学習のアンサンブルを繰り返し学んで同時に行う．NeurIPS 2021（arXiv のコメント欄） — [arXiv 2106.15728](https://arxiv.org/abs/2106.15728)
- Baek ほか "Agreement-on-the-Line"，NeurIPS 2022 — [Crossref](https://doi.org/10.52202/068431-1401)
- TiCAL は，単モーダルの擬似ラベルと典型度の推定で各学習標本のモダリティ間の一貫性を評価し，学習に組み込む．CMU-MOSEI と MER2023 で評価する．AAAI 2026，40(21):17948–17956 — [arXiv 2511.15085](https://arxiv.org/abs/2511.15085)，[Crossref](https://doi.org/10.1609/aaai.v40i21.38854)
- Albanie, Nagrani, Vedaldi, Zisserman，ACM MM 2018，292–301 頁 — [Crossref](https://doi.org/10.1145/3240508.3240578)（中身の要点は `research_notes/TAC 論文 先行研究の精読と穴/g2_crossmodal_distillation_er.md` にある）
- ウェブ検索では，半教師ありのマルチモーダル感情・感情極性の分析で，見方ごとの確信度を重みづけて擬似ラベルを較正する A2SI（ACL），支配的なモダリティとの一致で擬似ラベルを選ぶ SCDR（CVPR 2025），視覚の知識で音声の擬似ラベルを増やす Knowledge-Based Systems の論文が出た．いずれも標本ごとの選択・重みである（検索結果の要約のみ．一次の頁は未確認） — [ウェブ検索の結果（SCDR の頁）](https://cvpr2023.thecvf.com/virtual/2025/poster/34877)，[Knowledge-Based Systems の論文の頁](https://www.datalearner.com/academic/journal-papers/0950-7051/volumes-and-issues/346/paper-detail/10089)

### Inferences
- **M2 の骨組み（ラベル無しの視点間の一致 → 方向ごとの正則化の強さ）は Kakade & Foster 2007 にある．**本研究が M2 を出すときは，Kakade & Foster を最も近い先例として引き，違いを 3 点書く必要がある．(1) 縮める先が 0 ではなく教師の解 θ_Q である．(2) 方向がラベルの曲率 H_K と罰則 P の一般化固有方向であり，正準相関の方向ではない．(3) 一致を測る相手が「入力の 2 視点」ではなく「複数の教師（音声・映像・言葉・融合）の擬似ラベルで解いた生徒の解」である．
- **複数の教師から方向ごとの誤差を推定する具体的な形は，事前分布の共分散を解の散らばりから作る方法に近い．**たとえば教師 m ごとに擬似ラベルで θ_Q^(m) を解き，その散らばりの共分散を教師の誤差の共分散 Σ_e の推定に使えば，M1 の P = σ²Σ_e⁻¹ に入れられる．解の散らばりを事前分布の共分散にする発想は Pre-Train Your Loss（ソースの事後分布の共分散を下流の事前分布にする）にもある（[Shwartz-Ziv ほか](https://arxiv.org/abs/2205.10279)）．ただしこの推定は，教師どうしの誤差が独立で偏りを共有しないことを仮定する．音声・映像・言葉の教師が同じ方向に偏る（たとえば全員が中立に偏る）と，散らばりは誤差を小さく見積もる．この仮定は共同学習の理論の条件付き独立や拡張の条件と同じ種類のもので，Dasgupta ほか・Balcan ほかを引いて限界として書く必要がある（推論）．
- **精度推定の文献（Platanios，Jiang，Chen，Baek，Parisi）は，M2 の「ラベル無しで教師の信頼度を推定する」部分の先例として引ける．**ただしどれも分類器ごとの 1 つの数か標本ごとの数を出すので，パラメータの方向ごとの信頼に写す手順は本研究が新しく示す部分になる．

### Gaps
- Kakade & Foster の論文本文（LNCS 版）は読めず，罰則の式は共著者の講演スライドの文字で確かめた．論文の式番号と定数は未確認である．
- Sindhwani ほか 2005 の会議名，Balcan ほかと Platanios ほか 2014 の年，Jiang ほかの会議名は一次の頁で確かめていない．
- Ando & Zhang の構造学習（JMLR 2005，ICML 2007 の 2 視点の特徴生成）は，補助問題から学んだ部分空間の内と外で罰則を変えるという記憶があり M1・M2 に近い可能性があるが，本文を読んでいないので判定していない（書誌は [OpenAlex（JMLR 2005）](https://doi.org/10.5555/1046920.1194905)，[Crossref（ICML 2007，25–32 頁）](https://doi.org/10.1145/1273496.1273500)）．
- ICASSP・Interspeech の論文集で，モダリティ間の一致を方向ごとの信頼に使う感情認識の論文は，ウェブ検索 1 回の範囲では見つからなかった．IEEE Xplore を直接は検索していない．

---

## Q3. 半教師あり DA（2019〜2026）で，少数のターゲットラベルを使って方向ごと・スペクトルで擬似ラベルを重みづける方法はあるか

### Takeaway
見つからなかった．半教師あり DA の代表（MME，APE，CDAC，DECOTA）と，ウェブ検索で出た擬似ラベルの注入（Rawat ほか）・Pred&Guide は，標本ごと・クラスごとに擬似ラベルを選ぶか重みづけるもので，パラメータの方向ごとの重みは使わない．少数のラベルで教師への信頼を調整する方法としては，Meta Pseudo Labels（教師を生徒のラベル付きデータでの成績で更新する）と，予測を使った推測の PSPA・PPI++（ラベルで予測への信頼の重みを決める）があり，PSPA は座標ごとの重みを使う．

### Cited Findings
- DECOTA は，半教師あり DA を UDA の課題と半教師あり学習の課題に分け，2 つの分類器が互いに確信の高い擬似ラベルを渡す共同学習である（標本ごとの確信度の閾値）．ICCV 2021，8886–8896 頁 — [Crossref](https://doi.org/10.1109/ICCV48922.2021.00878)．MME・APE・CDAC・DECOTA の本文の精読と書誌は `research_notes/半教師あり DA の先例/ssda.md` にある（本調査では読み直していない）
- 擬似ラベルの注入（Rawat ほか，ECCV ワークショップ 2022）は，ラベル付きの標本との特徴の類似で擬似ラベルを付け，確信度の閾値で注入・除去する．Pred&Guide は少数のラベル付きターゲットの予測の正しさでソースの標本をクラスごとに重みづける（検索結果の要約のみ） — [arXiv 2209.01881](https://arxiv.org/pdf/2209.01881)，[arXiv 2211.11975](https://www.arxiv.org/pdf/2211.11975)
- 「半教師あり DA」「部分空間」「スペクトル」「パラメータ空間の正則化」を組み合わせた検索では，該当する論文は出なかった — [ウェブ検索の結果（SDASL，CVPR 2015 など）](https://microsoft.com/en-us/research/wp-content/uploads/2016/08/CVPR15.1159.pdf)
- 少数ショットの自己学習で，教師のパラメータの情報利得（ベイズのニューラルネットの不確かさ）で擬似ラベルの標本を選ぶ方法（arXiv 2310.13022）はあるが，標本の選択に使い，生徒の罰則には使わない（検索結果の要約のみ） — [arXiv 2310.13022](https://arxiv.org/pdf/2310.13022)
- Meta Pseudo Labels は，教師を固定せず，生徒のラベル付きデータでの成績を手がかりに教師を更新し続ける．CVPR 2021，11552–11563 頁 — [arXiv 2003.10580](https://arxiv.org/abs/2003.10580)，[Crossref](https://doi.org/10.1109/CVPR46437.2021.01139)
- PSPA は，少数のラベル付きデータの推定方程式に，平均が 0 の補正項を「重みのベクトル ω = [ω_1, …, ω_q]」の対角行列 diag(ω) を掛けて足し，ラベルと機械学習の予測の寄与を座標ごとに決める．平均の推定の例では最適な重みは ω_opt = Cov[Y, f̂]/{Var[f̂] + n Var[f̂]/N} で，予測がでたらめなら ω_opt = 0 になる．古典的な推定量に比べて座標ごとに分散を減らすと述べ，PPI++ は座標ごとの分散の減少を保証できないと書く（本文の該当箇所） — [arXiv 2311.14220](https://arxiv.org/abs/2311.14220)（v4．journal-ref 欄は空）
- PPI++ は，少数のラベルと大量の予測を組み合わせ，予測の質に合わせる調整のつまみ λ をスカラー（λ ∈ ℝ）で選ぶ．漸近共分散 Σ_λ の別のスカラー化（最大の対角要素など）を選べば別の λ になると書き，1 つの座標 θ_j だけを推測する場合の手順も示す（本文の該当箇所）．The Annals of Applied Statistics 20(3):2235–2252，2026 — [arXiv 2311.01453](https://arxiv.org/abs/2311.01453)，[Crossref](https://doi.org/10.1214/26-AOAS2215)

### Inferences
- **PSPA は「少数のラベルで，予測（教師）への信頼を座標ごとに決める」点で M1 に部分的に重なる．**ただし信頼の重みは予測とラベルの共分散から決まるので，ラベルで確かめられる座標ほど予測を信じ，確かめられない座標では重みを決められない．M1 は確かめられない方向ほど教師を信じる手法で，重みを決める根拠が逆である．また PSPA は推測（パラメータの推定と信頼区間）の方法で，予測の偏りを補正項で消すので，教師の偏りがあっても推定量は一致性を保つ．M1 の教師は偏りを持ち込みうる．
- **半教師あり DA の文献に対しては，「擬似ラベルの重みを標本ごとではなくパラメータの方向ごとに決める」ことが差になりうる．**論文では，代表の方法（MME，APE，CDAC，DECOTA）が標本ごとの確信度や閾値を使うことを本文で確かめた記録（`半教師あり DA の先例/ssda.md`）を根拠にして，この差を書ける．

### Gaps
- 2023〜2026 年の半教師あり DA の論文を網羅的には見ていない（ウェブ検索 2 回と兄弟の調査記録のみ）．CVPR・ICCV・ECCV の論文集の全文検索はしていない．
- PSPA の会議・雑誌の採録先は確かめていない．

---

## Q4. M1〜M4 ごとの判定（まとめ）

### Takeaway
M1〜M4 のどれについても，「同じ」と判定できる先例は見つからなかった．各案に部分的に重なる先例は次のとおりである．M1 には重みつきリッジ（Wu & Xu），Fisher merging，零空間への射影（Adam-NSCL など），PSPA，W2S の理論（Ildiz ほか）が重なる．M2 には Kakade & Foster 2007 の CCA リッジが最も近い．M3 にはランダム効果の模型でのリッジの解析（Dobriban & Wager，Wu & Xu）が重なる．M4 には MetaReg（座標ごとの正則化の重みを，残したソースのドメインで学ぶ）が最も近い．4 つに共通して先例に無いのは，「K 本のラベルの曲率の一般化固有方向で，教師の解への罰則を方向ごとに変える」組み合わせである．

### Cited Findings

**表 3．手法案ごとの判定**

| 案 | 判定 | 最も近い先例 | 重なる部分 | 違う部分 |
|---|---|---|---|---|
| M1（λ_j の関数で方向ごとに教師を信頼） | 同じものは無い．部分的に重なるものが複数 | Wu & Xu 2020，Fisher merging，Adam-NSCL・GPM・OGD，PSPA，Ildiz ほか 2024，EATA | 方向・座標ごとに罰則の重みを変える形（Wu & Xu，Fisher merging），データの曲率で部分空間を分けて更新を限る形（Adam-NSCL など），少数のラベルで予測への信頼を座標ごとに決める形（PSPA），教師の最適な形が固有方向ごとに違うという解析（Ildiz ほか） | 罰則の行き先が教師の解であること，方向を K 本のラベルの曲率で決めること，確かめられない方向ほど教師を信じること |
| M2（複数の教師の食い違いで確かめられない方向の信頼度を推定） | 同じものは無い．部分的に重なる | Kakade & Foster 2007 | ラベル無しの視点間の一致（正準相関）から方向ごとの罰則を作る | 縮める先が 0，方向が正準方向，一致を測る相手が入力の 2 視点 |
| M3（確かめられる方向の誤差を確かめられない方向へ移す） | 同じものは無い．部分的に重なる | Dobriban & Wager 2018，Wu & Xu 2020，PPI++ | 係数をランダム効果と見なして最適なリッジの強さを決める（Dobriban & Wager），異方的な事前共分散の下での最適な重み（Wu & Xu），ラベルで決めたスカラーの信頼を全座標に使う（PPI++） | 教師の誤差を対象にすること，λ の大小で方向を 2 群に分けて一方の推定を他方へ移すこと |
| M4（残したソースを仮のターゲットにして方向ごとの信頼を選ぶ） | 同じものは無い．部分的に重なる | MetaReg（Balaji ほか 2018） | 座標ごとの正則化の重みを，メタ学習用とメタ試験用に分けたソースのドメインで学ぶ．特徴の網を凍結して上の層だけで学ぶ | 罰則の行き先が 0，重みが座標ごとでラベルの曲率の関数ではない，教師が無い，DG でターゲットのラベルを使わない |

- MetaReg は，正則化関数 R_φ(θ) を課題の網（task network）の重み θ にだけかけ，その形を重みつき L1，R_φ(θ) = Σ_i φ_i|θ_i| とする（重みつき L2 と 2 層のネットワークも試し，重みつき L1 が最良）．各エピソードでソースの 2 つのドメイン a ≠ b を選び，a のラベル付きデータで数歩の勾配降下をした後の b での損失を最小にするよう φ を更新する．規模を大きくするため特徴の網を凍結し，課題の網だけでメタ学習する（本文 3.2 節と 4 節の文字）．NeurIPS 2018，998–1008 頁（頁は mlanthology） — [NeurIPS の論文集の頁](https://proceedings.neurips.cc/paper/2018/hash/647bba344396e7c8170902bcf2e15551-Abstract.html)，[ウェブ検索の結果（mlanthology の頁）](https://mlanthology.org/neurips/2018/balaji2018neurips-metareg)
- Tseng ほか は，特徴ごとの変換層の超パラメータを，学習用の仮の既知ドメインと仮の未知ドメインに分けた learning-to-learn で探す．ICLR 2020（arXiv のコメント欄） — [arXiv 2001.08735](https://arxiv.org/abs/2001.08735)
- MLDG（Li, Yang, Song, Hospedales）は，ソースのドメインをメタ学習用とメタ試験用に分けるメタ学習で DG を行う．AAAI 2018 — [Crossref](https://doi.org/10.1609/aaai.v32i1.11596)（中身は題と記憶）
- Gulrajani & Lopez-Paz "In Search of Lost Domain Generalization" は，DG のモデル選択の方法（学習ドメインの検証，1 ドメインを残す検証など）を比べる — [arXiv 2007.01434](https://arxiv.org/abs/2007.01434)（中身と ICLR 2021 は記憶）
- Meta-Curvature は，MAML の内側の最適化で勾配を変換する曲率の行列を課題をまたいで学ぶ．NeurIPS 2019（arXiv のコメント欄） — [arXiv 1902.03356](https://arxiv.org/abs/1902.03356)
- Denevi ほか は，偏りベクトルへの 2 乗距離で正則化した SGD を考え，偏りベクトル（錨）を課題の列から推定するメタアルゴリズムを示す — [arXiv 1903.10399](https://arxiv.org/abs/1903.10399)（会議名は未確認）
- Dobriban & Wager は，高次元の比例の極限でリッジ回帰と正則化判別分析の予測の誤差を解析する．The Annals of Statistics 46(1):247–279，2018 — [Crossref](https://doi.org/10.1214/17-AOS1549)（係数を等方のランダム効果と見なしたときの最適なリッジの強さの式は記憶．兄弟の調査記録 `research_notes/ラベル換算の価値で重みを決める先例/report.md` に本文の要点がある）
- Wu & Xu は，係数の事前共分散 Σ_β を異方的にした下で最適な重みの行列 Σ_w を決める（Q1 の項） — [arXiv 2006.05800](https://arxiv.org/abs/2006.05800)

### Inferences
- **M3 の「確かめられる方向の誤差を確かめられない方向へ移す」ことは，等方のリッジの強さを周辺尤度や一般化交差検証で選ぶ操作に暗に含まれている．**θ − θ_Q に等方のリッジをかけ，強さを K 本のラベルで選ぶと，ラベルが情報を持つのは λ の大きい方向だけなので，そこでの教師の誤差の大きさから決めた 1 つの強さを全方向に使うことになる．これは「教師の誤差が方向によらず同じ分散を持つ」というランダム効果の仮定と同じである（本ノートの推論．Dobriban & Wager の等方のランダム効果の模型に相当）．したがって M3 を新しい手法として出すには，(a) λ で方向を分けて推定と適用を明示すること，(b) 移す仮定（誤差の分散が λ の大小で変わらないこと）を確かめる実験を付けることが要る．研究者の E68・E69 の結果（λ ≤ 1 の方向が得の 85〜108% を持つ）は，(b) の検証の材料になる．
- **M4 は MetaReg を最も近い先例として引き，違いを書く必要がある．**MetaReg は座標ごとの重み φ_i を直接学ぶので，ターゲットが変わって H_K の固有方向が変わると使い回せない．本研究の M4 は「λ の関数 f(λ)」を残したソースで学べば，固有方向がコーパスごとに違ってもつまみの数が少なく保てる．この「方向ではなく曲率の値の関数を学ぶ」形は見つからなかった．
- **論文の書き方の注意．**(1) 1/(1+λ) の縮小は既知の式（Fisher merging・ベイズの事後平均・一般化リッジ）として引く．(2) 「方向ごとの罰則」そのものは新しいと書かない（Wu & Xu，EWC，Fisher merging，Pre-Train Your Loss がある）．(3) 新しいと書けそうなのは，「少数のターゲットラベルの曲率で確かめられない方向に教師の価値があることの実測」と，「その方向での教師の信頼を，ラベル無しの複数教師の食い違い（M2）か，残したソース（M4）で決める手順」である．ただしこの判定は本調査の検索範囲（下の Gaps）に限られる．

### Gaps
- 経験ベイズ・自動関連度決定（ARD．MacKay の evidence 近似，Tipping 2001 の関連ベクトル機械）は，座標ごとの事前精度を周辺尤度で決める方法として M3 に近い可能性があるが，OpenAlex の検索で書誌を確かめられなかったので判定していない．
- 転移学習のリッジ（ソースの推定量へのリッジ）を固有方向ごとに解析した統計の文献（Dar & Baraniuk など）は今回調べていない．兄弟の調査 `research_notes/MER DA 未定式化理論の実現可能性/` と `research_notes/TAC 論文 先行研究の精読と穴/g4_theory_intercept_direction.md` に一部があるかもしれない．
- 書誌の確かめが △ の文献（Hoerl & Kennard，Sindhwani ほか，Balcan ほか，Platanios ほか，Parisi ほか，MLDG，Gulrajani & Lopez-Paz）は，中身を記憶か題で書いた．論文で引く前に本文を確かめる必要がある．

### BibTeX（部分的に重なる以上と判定した文献と，本研究の罰則の基盤の L2-SP）

```bibtex
@inproceedings{matena2022fisher,
  author    = {Matena, Michael S. and Raffel, Colin},
  title     = {Merging Models with {Fisher}-Weighted Averaging},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {17703--17716},
  year      = {2022},
  doi       = {10.52202/068431-1287}
}

@inproceedings{jin2023regmean,
  author    = {Jin, Xisen and Ren, Xiang and Preotiuc-Pietro, Daniel and Cheng, Pengxiang},
  title     = {Dataless Knowledge Fusion by Merging Weights of Language Models},
  booktitle = {International Conference on Learning Representations},
  year      = {2023},
  note      = {arXiv:2212.09849}
}

@inproceedings{wu2020optimal,
  author    = {Wu, Denny and Xu, Ji},
  title     = {On the Optimal Weighted $\ell_2$ Regularization in Overparameterized Linear Regression},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  year      = {2020},
  note      = {arXiv:2006.05800}
}

@article{hoerl1970ridge,
  author  = {Hoerl, Arthur E. and Kennard, Robert W.},
  title   = {Ridge Regression: Biased Estimation for Nonorthogonal Problems},
  journal = {Technometrics},
  volume  = {12},
  number  = {1},
  pages   = {55--67},
  year    = {1970},
  doi     = {10.1080/00401706.1970.10488634}
}

@article{kirkpatrick2017ewc,
  author  = {Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A. and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and Hassabis, Demis and Clopath, Claudia and Kumaran, Dharshan and Hadsell, Raia},
  title   = {Overcoming Catastrophic Forgetting in Neural Networks},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {114},
  number  = {13},
  pages   = {3521--3526},
  year    = {2017},
  doi     = {10.1073/pnas.1611835114}
}

@inproceedings{li2018l2sp,
  author    = {Li, Xuhong and Grandvalet, Yves and Davoine, Franck},
  title     = {Explicit Inductive Bias for Transfer Learning with Convolutional Networks},
  booktitle = {International Conference on Machine Learning},
  year      = {2018},
  note      = {arXiv:1802.01483}
}

@inproceedings{li2019delta,
  author    = {Li, Xingjian and Xiong, Haoyi and Wang, Hanchao and Rao, Yuxuan and Liu, Liping and Chen, Zeyu and Huan, Jun},
  title     = {{DELTA}: {DE}ep Learning Transfer using Feature Map with Attention for Convolutional Networks},
  booktitle = {International Conference on Learning Representations},
  year      = {2019},
  note      = {arXiv:1901.09229}
}

@misc{farajtabar2019ogd,
  author = {Farajtabar, Mehrdad and Azizan, Navid and Mott, Alex and Li, Ang},
  title  = {Orthogonal Gradient Descent for Continual Learning},
  year   = {2019},
  eprint = {1910.07104},
  archivePrefix = {arXiv},
  note   = {AISTATS 2020 版は未確認}
}

@inproceedings{saha2021gpm,
  author    = {Saha, Gobinda and Garg, Isha and Roy, Kaushik},
  title     = {Gradient Projection Memory for Continual Learning},
  booktitle = {International Conference on Learning Representations},
  year      = {2021},
  note      = {arXiv:2103.09762}
}

@inproceedings{wang2021adamnscl,
  author    = {Wang, Shipeng and Li, Xiaorong and Sun, Jian and Xu, Zongben},
  title     = {Training Networks in Null Space of Feature Covariance for Continual Learning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {184--193},
  year      = {2021},
  doi       = {10.1109/CVPR46437.2021.00025}
}

@inproceedings{fang2025alphaedit,
  author    = {Fang, Junfeng and Jiang, Houcheng and Wang, Kun and Ma, Yunshan and Jie, Shi and Wang, Xiang and He, Xiangnan and Chua, Tat-Seng},
  title     = {{AlphaEdit}: Null-Space Constrained Knowledge Editing for Language Models},
  booktitle = {International Conference on Learning Representations},
  year      = {2025},
  note      = {arXiv:2410.02355}
}

@misc{zakarin2025bulk,
  author = {Zakarin, Daniyar and Singh, Sidak Pal},
  title  = {Accelerating Neural Network Training Along Sharp and Flat Directions},
  year   = {2025},
  eprint = {2505.11972},
  archivePrefix = {arXiv}
}

@inproceedings{niu2022eata,
  author    = {Niu, Shuaicheng and Wu, Jiaxiang and Zhang, Yifan and Chen, Yaofo and Zheng, Shijian and Zhao, Peilin and Tan, Mingkui},
  title     = {Efficient Test-Time Model Adaptation without Forgetting},
  booktitle = {International Conference on Machine Learning},
  year      = {2022},
  note      = {arXiv:2204.02610}
}

@inproceedings{shwartzziv2022pretrain,
  author    = {Shwartz-Ziv, Ravid and Goldblum, Micah and Souri, Hossein and Kapoor, Sanyam and Zhu, Chen and LeCun, Yann and Wilson, Andrew Gordon},
  title     = {Pre-Train Your Loss: Easy {B}ayesian Transfer Learning with Informative Priors},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {27706--27715},
  year      = {2022},
  doi       = {10.52202/068431-2009}
}

@inproceedings{thompson2019ewcnmt,
  author    = {Thompson, Brian and Gwinnup, Jeremy and Khayrallah, Huda and Duh, Kevin and Koehn, Philipp},
  title     = {Overcoming Catastrophic Forgetting During Domain Adaptation of Neural Machine Translation},
  booktitle = {Proceedings of NAACL-HLT 2019},
  pages     = {2062--2068},
  year      = {2019},
  doi       = {10.18653/v1/N19-1209}
}

@misc{ildiz2024kd,
  author = {Ildiz, M. Emrullah and Gozeten, Halil Alperen and Taga, Ege Onur and Mondelli, Marco and Oymak, Samet},
  title  = {High-dimensional Analysis of Knowledge Distillation: Weak-to-Strong Generalization and Scaling Laws},
  year   = {2024},
  eprint = {2410.18837},
  archivePrefix = {arXiv},
  note   = {v2, 2025-02-27. 採録先は未確認}
}

@inproceedings{mobahi2020selfdistill,
  author    = {Mobahi, Hossein and Farajtabar, Mehrdad and Bartlett, Peter L.},
  title     = {Self-Distillation Amplifies Regularization in {H}ilbert Space},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  year      = {2020},
  note      = {arXiv:2002.05715}
}

@inproceedings{dong2025discrepancies,
  author    = {Dong, Yijun and Li, Yicheng and Li, Yunai and Lee, Jason D. and Lei, Qi},
  title     = {Discrepancies are Virtue: Weak-to-Strong Generalization through Lens of Intrinsic Dimension},
  booktitle = {International Conference on Machine Learning},
  year      = {2025},
  note      = {arXiv:2502.05075}
}

@inproceedings{kumar2022lpft,
  author    = {Kumar, Ananya and Raghunathan, Aditi and Jones, Robbie and Ma, Tengyu and Liang, Percy},
  title     = {Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution},
  booktitle = {International Conference on Learning Representations},
  year      = {2022},
  note      = {arXiv:2202.10054}
}

@misc{miao2023pspa,
  author = {Miao, Jiacheng and Miao, Xinran and Wu, Yixuan and Zhao, Jiwei and Lu, Qiongshi},
  title  = {Assumption-Lean and Data-Adaptive Post-Prediction Inference},
  year   = {2023},
  eprint = {2311.14220},
  archivePrefix = {arXiv}
}

@article{angelopoulos2026ppipp,
  author  = {Angelopoulos, Anastasios N. and Duchi, John C. and Zrnic, Tijana},
  title   = {{PPI++}: Efficient Prediction-Powered Inference},
  journal = {The Annals of Applied Statistics},
  volume  = {20},
  number  = {3},
  pages   = {2235--2252},
  year    = {2026},
  doi     = {10.1214/26-AOAS2215}
}

@inproceedings{kakade2007multiview,
  author    = {Kakade, Sham M. and Foster, Dean P.},
  title     = {Multi-view Regression Via Canonical Correlation Analysis},
  booktitle = {Learning Theory (COLT 2007)},
  series    = {Lecture Notes in Computer Science},
  pages     = {82--96},
  year      = {2007},
  doi       = {10.1007/978-3-540-72927-3_8}
}

@inproceedings{sindhwani2005coreg,
  author    = {Sindhwani, Vikas and Niyogi, Partha and Belkin, Mikhail},
  title     = {A Co-Regularization Approach to Semi-supervised Learning with Multiple Views},
  booktitle = {Workshop on Learning with Multiple Views (ICML 2005)},
  year      = {2005},
  note      = {会議名は記憶．未確認}
}

@article{dobriban2018ridge,
  author  = {Dobriban, Edgar and Wager, Stefan},
  title   = {High-dimensional Asymptotics of Prediction: Ridge Regression and Classification},
  journal = {The Annals of Statistics},
  volume  = {46},
  number  = {1},
  pages   = {247--279},
  year    = {2018},
  doi     = {10.1214/17-AOS1549}
}

@inproceedings{balaji2018metareg,
  author    = {Balaji, Yogesh and Sankaranarayanan, Swami and Chellappa, Rama},
  title     = {{MetaReg}: Towards Domain Generalization using Meta-Regularization},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {31},
  pages     = {998--1008},
  year      = {2018}
}

@inproceedings{tseng2020fwt,
  author    = {Tseng, Hung-Yu and Lee, Hsin-Ying and Huang, Jia-Bin and Yang, Ming-Hsuan},
  title     = {Cross-Domain Few-Shot Classification via Learned Feature-Wise Transformation},
  booktitle = {International Conference on Learning Representations},
  year      = {2020},
  note      = {arXiv:2001.08735}
}

@inproceedings{li2018mldg,
  author    = {Li, Da and Yang, Yongxin and Song, Yi-Zhe and Hospedales, Timothy},
  title     = {Learning to Generalize: Meta-Learning for Domain Generalization},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {32},
  number    = {1},
  year      = {2018},
  doi       = {10.1609/aaai.v32i1.11596}
}

@misc{gulrajani2020lost,
  author = {Gulrajani, Ishaan and Lopez-Paz, David},
  title  = {In Search of Lost Domain Generalization},
  year   = {2020},
  eprint = {2007.01434},
  archivePrefix = {arXiv},
  note   = {ICLR 2021 版は未確認}
}

@inproceedings{park2019metacurvature,
  author    = {Park, Eunbyung and Oliva, Junier B.},
  title     = {Meta-Curvature},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {32},
  year      = {2019},
  note      = {arXiv:1902.03356}
}

@misc{denevi2019biased,
  author = {Denevi, Giulia and Ciliberto, Carlo and Grazzi, Riccardo and Pontil, Massimiliano},
  title  = {Learning-to-Learn Stochastic Gradient Descent with Biased Regularization},
  year   = {2019},
  eprint = {1903.10399},
  archivePrefix = {arXiv}
}

@inproceedings{pham2021mpl,
  author    = {Pham, Hieu and Dai, Zihang and Xie, Qizhe and Le, Quoc V.},
  title     = {Meta Pseudo Labels},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {11552--11563},
  year      = {2021},
  doi       = {10.1109/CVPR46437.2021.01139}
}
```
