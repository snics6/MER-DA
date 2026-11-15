# G4：少数ラベルが直すのは切片か向きか，ソースの価値をラベル何本ぶんで測るか — 理論の論文 10 本の精読

調べた日：2026-10-27．前回の調べ（`../TAC 論文の先行研究と書き方/theory_preemption.md`）の URL と要点を出発点にし，本文を読み切って記録した．
確かめた水準：◎ 本文（該当の節・定理を読んだ），○ 抄録か予稿集・検索結果の頁，△ 検索結果の抜き書きだけ．
書誌は本文の冒頭・脚注・arXiv の注記で確かめた範囲だけを書き，確かめられないものは「要確認」と書く．原文の引用は英語のまま短く引く．

本文の取り方：PDF を curl で `scratchpad/g4_pdfs/` に落とし pdftotext で文字にした（Okamoto 1963 の原著だけは Project Euclid の bot 対策（Incapsula）で curl・WebFetch・Internet Archive のどれでも取れなかった．Shutoh の技術報告の再掲で代える）．数式は pdftotext で崩れるので，形が要る箇所は前後から復元し，1 箇所（Okamoto の係数の分解）は Python で数値的に確かめた（`scratchpad/check_okamoto.py`）．6 本（Kini，Kumar，De Silva，Hernandez，Dobriban & Wager，Luthra）は同じ指示書（`scratchpad/g4_brief.md`）を渡した下請けの精読記録を，引用の有無を grep で抜き取り検査して載せた．残り（Okamoto／Shutoh，Raudys，DARE，Ha & Chen，Zhang，Le Scao & Rush）は本記録の筆者が読んだ．

## 本研究の理論の主張（記録の対応づけに使う）

- 主張 A：凍結した特徴の上の L2 罰則つきロジスティック回帰で，クラスごとに K 本のラベルで学ぶとき，切片（クラスごとの閾値）の誤差は 1/(2K)，向き（重みベクトル）の誤差は q/K（q は向きの実効次元．実測 16〜22）で減る．ラベル無しの中心化（ドメインごとに平均を引く）で切片のずれの大部分が消える．2 クラスの正規の雲の模型で正解率 = ½[Φ(d/2 + o) + Φ(d/2 − o)]（d はクラス間のマハラノビス距離，o は閾値のずれ）．
- 主張 B：ソースの価値をターゲットのラベルの本数で測る．n = q e²/(1 − e²)（e はソースの向きの効率．推定した向きと真の向きの余弦）．

## 軸（表の列）

A1 学習器（凍結特徴の上の線形の頭）／A2 誤差を切片と向きに分けるか，K で減る式，p（実効次元）の入り方／A3 ソースの価値のラベル本数への換算，余弦² と標本数／A4 合併で切片が直る割合・暗黙のドメイン別切片／A5 固定の少数ショット（有限か比例の極限か）／A6 ドメインのずれの型／A7 マルチモーダル／A8 弱いラベル・疑似ラベル／A9 出発点への L2 罰則／A10 実データの検証／A11 予言と実測の照合．

## 先に結論

- 主張 A の「切片の誤差 1/(2K)，向きの誤差 q/K」は，線形判別関数の誤判別率の古典的な展開（Okamoto 1963）の 1/N の項を，閾値の項（次元に依らない Δ/8）と向きの項（(p−1)/(2Δ)）に分けたものと同じ形である．この分け方は展開の係数から 1 段の計算で出る（本記録で確かめた）が，Shutoh の再掲にはその解釈の文は無く，Okamoto の原著と McLachlan 1992 の言葉では確かめられていない．Raudys の式（EDC の T_μ = 1 + 2p*/(δ*²N)）は向きの項だけを残した形で，p* は「実効次元」として定義されている（Raudys 1998 式 (1)）．
- 正解率の式 ½[Φ(d/2 + o) + Φ(d/2 − o)] は，De Silva ほか 2023 の式 (7)，Dobriban & Wager の式 (16)（"the well-known finite-sample formula"），Zhang ほか 2026 の式 (1)，Kini ほか 2021 の付録 F.1 に同じ形がある．本研究の式はこれらの特別な場合であり，新しい式ではない．
- 主張 B の n = q e²/(1 − e²) を明示した文献は無い．余弦² = n/(n + p/α²) の形は Dobriban & Wager の系 3.3（Σ = I）に含まれ，Raudys の T_μ と同じ関係である．ソースの価値をターゲットのデータ量に換算する発想は Hernandez ほか 2021（文字数で実測），Le Scao & Rush 2021（訓練例の数で実測），Ha & Chen 2026 の OLS-Tar-MoreData（d/n_new = r/n(0)）に先例がある．
- 10 本のどれも，「凍結した深層特徴の上の罰則つきロジスティック回帰」「クラスあたり数本の固定の少数ショット」「ドメインのずれ」「切片と向きの分離」の 4 つを同時に扱っていない（末尾の表と「穴」）．

---

## 1. Okamoto 1963（Shutoh の技術報告の再掲で読んだ）

### 1. 書誌と読んだ範囲
- 原著：Okamoto, M., "An asymptotic expansion for the distribution of the linear discriminant function," Ann. Math. Statist. 34(4), 1963．DOI 10.1214/aoms/1177703864（Semantic Scholar の API の記録 [DOI:10.1214/AOMS/1177703864](https://api.semanticscholar.org/graph/v1/paper/DOI:10.1214/aoms/1177703864) と Project Euclid の検索結果 [projecteuclid.org/euclid.aoms/1177703864](https://projecteuclid.org/euclid.aoms/1177703864) で確認）．頁は Euclid の検索結果と Barra 1981 の引用では 1286–1301，Shutoh の参考文献では 1268–1301 と食い違う（Shutoh の誤植と読める．**要確認**）．訂正：Okamoto 1968, Ann. Math. Statist. 39, 1358–1359（Shutoh の参考文献）．**原著の本文は取れなかった（△ 書誌のみ）**．Project Euclid は open access だが bot 対策で PDF が返らない．
- 読んだもの：Shutoh, N., "Asymptotic expansions relating to discrimination based on 2-step monotone missing samples," Tokyo University of Science（広島大学の統計研究グループの技術報告 TR10-04 として公開）[PDF](https://www.math.sci.hiroshima-u.ac.jp/stat/TR/TR10/TR10-04.pdf)．1 節（設定，Okamoto の 1 次の展開の式 (3) と係数 a1〜a3，McLachlan 1973 の式 (4)，type I／type II の区別），2 節（MLE と LDF）を読んだ（◎）．3 節以降（欠測のある場合の導出，補題，シミュレーション）は飛ばした．
- McLachlan 1992 の教科書（Discriminant Analysis and Statistical Pattern Recognition, Wiley）は入手できず未読．検索では Hiroshima の技術報告が "asymptotic results under a large-sample framework were reviewed by Siotani (1982) and by McLachlan (1992)" と述べる頁だけ見つかった（△）．

### 2. 設定
- 2 群 Π(g) ~ N_p(μ(g), Σ)，共通の Σ．標本の線形判別関数 W = (x̄(1) − x̄(2))′ S⁻¹ [x − ½(x̄(1) + x̄(2))]．x は W > c なら Π(1) へ．S はプールした標本共分散（自由度 n1 = N1 + N2 − 2）．罰則は無い．
- 標本数の体制：p を固定して N1, N2 → ∞，N2/N1 → 定数（Shutoh の言う type I）．展開は (1/N1, 1/N2, 1/n1) の 1 次（Okamoto は 2 次まで）．p と n を同時に大きくする type II（Fujikoshi & Seo 1998）とは別．
- ずれ：学習と評価は同じ分布．ドメインのずれは無い．

### 3. 示したこと
Shutoh の式 (3)（Okamoto の 1 次の項）：
P(2|1)(v) ≡ Pr[(W − ½Δ²)/Δ ≤ v | x ∈ Π(1)] = Φ(v) − ½ [a1/(N1Δ²) + a2/(N2Δ²) + a3/(2n1)] φ(v) + O(N1⁻², N2⁻², n1⁻²)，
a1 = v³ + (p−3)v − pΔ，a2 = v³ + 2Δv² + (p−3+Δ²)v + (p−2)Δ，a3 = 4v³ + 4Δv² + (6p−6+Δ²)v + 2(p−1)Δ．
Δ² = (μ(1) − μ(2))′Σ⁻¹(μ(1) − μ(2)) はマハラノビス距離の 2 乗（本研究の d²）．

**本記録の計算（原文に無い）．**cut-off を c = 0（v = −Δ/2）とし N1 = N2 = N と置くと，Π(1) の期待誤判別率は
E[誤判別率] ≈ Φ(−Δ/2) + φ(Δ/2) { [Δ/8 + (p−1)/(2Δ)] / N + (p−1)Δ/(4n1) }．
p = 2, 5, 20, 100，Δ = 1, 2, 3 で係数を数値的に照合した（`scratchpad/check_okamoto.py`．誤差 0）．各項の由来は次のように読める（本記録の推論）：
- Δ/(8N)：次元に依らない項．閾値（2 つの標本平均の中点）の判別の向きに沿った位置の誤差に当たる．中点の向きに沿った分散は，級内の標準偏差を 1 として ¼(1/N + 1/N) = 1/(2N)．閾値のずれを o とすると誤判別率は ½[Φ(−Δ/2 + o) + Φ(−Δ/2 − o)] ≈ Φ(−Δ/2) + (Δ/4)φ(Δ/2)·E[o²] で，E[o²] = 1/(2N) を入れると Δφ(Δ/2)/(8N) に一致する．**本研究の「切片の誤差 1/(2K)」は，この E[o²] = 1/(2N) そのものである（K = N）．**
- (p−1)/(2ΔN)：次元に比例する項．推定した向きと真の向きの角 θ について tan²θ ≈ 2(p−1)/(NΔ²)（向きに直交する p−1 次元の雑音），実効の距離 Δcosθ ≈ Δ(1 − tan²θ/2) から，誤判別率の増分は φ(Δ/2)·(p−1)/(2ΔN) に一致する．**本研究の「向きの誤差 q/K」は tan²θ = q/K と置いたもので，等方な雑音の線形判別では q = 2(p−1)/Δ² に当たる．**
- (p−1)Δ/(4n1)：共分散の推定の項．p−1 だけに比例し，次元に依らない部分を持たない．共分散の推定は向きだけを乱す（閾値は乱さない）と読める．
McLachlan 1973 の式 (4)（Shutoh の再掲）は，誤判別率の推定量 Φ(−D/2) の期待値の展開で，係数 b1 にも (p−1) を含む項と含まない項が並ぶ．これは「誤判別率の推定の偏り」の式であり，本研究の主張とは対象が違う．
実験：Okamoto の原著の数値例は未読．Shutoh は欠測のある拡張についてシミュレーションを載せる（未読）．

### 4. 軸との対応
| 軸 | 判定 | 根拠 |
|---|---|---|
| A1 | △ | 生の特徴の上の標本 LDF（平均と共分散の差し込み）．凍結した深層特徴でもロジスティック回帰でもない |
| A2 | ○ | 1/N の項が次元に依らない Δ/8 と (p−1)/(2Δ) に分かれる（式 (3) から 1 段の計算）．ただし「閾値」「向き」という解釈は再掲の文に無く，本記録の推論 |
| A3 | × | ソースが無い．余弦と標本数の関係は tan²θ ≈ 2(p−1)/(NΔ²) の形で暗黙に含まれる（本記録の推論） |
| A4 | × | 合併もドメイン別切片も無い |
| A5 | △ | N は有限だが，展開は p を固定し N を大きくする枠組み．p ≫ N の少数ショットでは使えない（Shutoh の type I／type II の区別） |
| A6 | × | 同じ分布 |
| A7 | × | — |
| A8 | × | — |
| A9 | × | 罰則なし |
| A10 | × | 実データ無し（Shutoh もシミュレーションのみ） |
| A11 | △ | Shutoh は展開とモンテカルロを比べる（未読）．Okamoto 原著は不明 |

### 5. 穴
- 扱っていないこと：p が N と同程度の体制（Shutoh："Type I approximations are ones under the framework such that N1 → ∞, N2 → ∞ ... type II approximations are ones under the framework such that ... p → ∞, n1 − p → ∞"．type II は Fujikoshi & Seo 1998），罰則つきの推定，ロジスティック回帰，ドメインのずれ，閾値と向きの誤差を分けて述べる文．
- Shutoh が限界として書くこと：展開は 1 次まで（"we extend the result of Okamoto (1963) and McLachlan (1973) up to the terms of the first order"）．

### 6. 本研究との差の書き方
「標本から作った線形判別関数の誤判別率は，Okamoto (1963) の展開では標本数の逆数の項が次元に依らない部分 Δ/8 と次元に比例する部分 (p−1)/(2Δ) に分かれ，前者は閾値の推定，後者は向きの推定の誤差に当たる．本研究の切片の誤差 1/(2K) と向きの誤差 q/K はこの分け方を凍結した特徴の上の L2 罰則つきロジスティック回帰に移したもので，p−1 の代わりに実測した実効次元 q（16〜22）を置く点と，ドメインのずれで生じる切片のずれをラベル無しの中心化で消す点が異なる．」
式の対応：本研究の 1/(2K) は Okamoto の Δ/(8N) の項（E[o²] = 1/(2N)）を，q/K は (p−1)/(2ΔN) の項（tan²θ）を，それぞれ書き直したものに当たる．

### 7. 注意
- 閾値と向きへの分解は本記録の計算と推論であり，Okamoto の原著・McLachlan 1992 の言葉では確かめていない．論文に「Okamoto が閾値と向きに分けた」と書いてはいけない．「Okamoto の展開の係数から分けられる」と書く．
- Okamoto 1963 の頁番号（1286 か 1268 か）は要確認．原著と McLachlan 1992 は未読．

---

## 2. Raudys & Jain 1991 と Raudys 1998（Raudys 1998 Kybernetika と Jain & Raudys 1992 Informatica で読んだ）

### 1. 書誌と読んだ範囲
- Raudys, Š. and Jain, A. K., "Small sample size effects in statistical pattern recognition: Recommendations for practitioners," IEEE Trans. Pattern Anal. Mach. Intell. PAMI-13, 252–264, 1991（Raudys 1998 の参考文献 [5] から確認．号は 3 と記憶するが本文で確かめていない．**原著は取れず未読（○ 書誌と要旨の抜き書き）**．IEEE Xplore の頁は読み取れなかった）．
- Raudys, Š., "Intrinsic dimensionality and small sample properties of classifiers," Kybernetika 34(4), 461–466, 1998 [dml.cz](https://dml.cz/handle/10338.dmlcz/135232)．**全 6 頁を読んだ（◎）．式 (1) は OCR が崩れたので頁の画像で確かめた．**前回の調べの URL（`34_4_332_full.pdf`，「34(4):332」）は誤りで，同じ号に 332 頁から始まる論文は無い（kybernetika.cz の号の目次で確認）．
- Jain, A. and Raudys, Š., "On training sample size and complexity of artificial neural net classifier," Informatica 3(3), 301–337, 1992 [PDF](https://informatica.vu.lt/journal/INFORMATICA/article/86/file/pdf)．307〜311 頁（Fisher LDF の期待誤判別率，Deev の展開の式 (11)，Table 1）を読んだ（◎ 一部）．

### 2. 設定
- Raudys 1998 式 (1) の設定：2 つのガウス N(μ_i, Σ)（共通の共分散．GCCM），分類器は最近傍平均の線形判別（Euclidean distance classifier, EDC：Σ = Iσ² を仮定して作る）．N1 = N2 = N，q1 = q2．p と N がともに大きい枠組み（"when the dimensionality p and learning set sizes N1, N2 are large and are increasing simultaneously"．Raudys 1967 の結果）．
- Fisher LDF は同じ式に共分散の推定の因子を掛ける．罰則は無い（正則化判別は SLP の文脈で名前だけ出る）．ずれ：学習と評価は同じ分布．

### 3. 示したこと
**Raudys 1998 式 (1)（頁 462 の画像で確認）：**
EP_N^(E) ≈ Φ{ −(δ*/2) · 1/√T*_μ }，
δ* = μ′μ/√(μ′Σμ)（μ = μ1 − μ2），T*_μ = 1 + 2p*/(δ*²N)，p* = (μ′μ)²(tr Σ²)/(μ′Σμ)² "is an effective dimensionality"．
球形（Σ = Iσ²）なら p* = p，δ* = δ（δ² = μ′Σ⁻¹μ）．
**Fisher LDF（頁 463 の画像で確認）：**"Its generalization error can be expressed by (1) with T_μT_Σ instead of T*_μ [5]. The term T_μ = 1 + 2p/(δ²N) arises from the inexact sample estimation of the mean vectors of the classes and the term T_Σ = 1 + p/(2N − p) arises from the inexact sample estimation of the covariance matrix. For GCCM model, however, the generalization error depends on the true dimensionality p, and not on p*"．
- 導出は論文に無い（[2] Raudys 1967（露語）と [5] Raudys & Jain 1991 を引く）．**本記録の推論**：推定した向き μ̂ = μ + 雑音の，μ に直交する成分の 2 乗和は ≈ 2p/N（単位分散，N 本ずつ）なので tan²θ ≈ 2p/(δ²N)，cos²θ = 1/T_μ，実効の距離 δ/√T_μ．つまり式 (1) は Okamoto の展開の「向きの項」だけを閉じた形にしたもので，閾値の項 Δ/(8N)（p を含まない）は p, N → ∞ の枠組みでは高次として落ちている．
- **「本研究の q」に当たる量**：q/K = 2p*/(δ*²N) と置けば q = 2p*/δ*²（K = N）．球形なら q = 2p/δ²．Fisher LDF では全次元が効く（p* でなく p）ので q = 2p/δ² に共分散の因子 T_Σ が掛かる．Raudys の p* は「真の分布と分類器の型で決まる実効次元」で，本研究の q（罰則つきロジスティック回帰で実測）と役割が同じである．ただし p* は μ と Σ から定義される量で，罰則の効果は入っていない．
- 数値例（Raudys 1998）：p = 100，p* ≈ 10⁸，P∞ = 0.03，N = 200 → EP = 0.497（シミュレーションも一致）．p = 100，p* ≈ 1.05，δ = 3.76，P∞ = 0.03，N = 5 → EP = 0.0318（"only five vectors per class are sufficient"．シミュレーションで確認）．同じ 2 例を Fisher LDF で解くと EP = 0.0577．
- Jain & Raudys 1992：Fisher LDF の期待誤判別率は Deev 1970 の展開（式 (11)）が正確（Pikelis 1976，Wyman ほか 1990 の比較）．Table 1（P∞ と N と d の関数としての EP_N/P∞）の例：d = 50，P∞ = 0.03 で N = 30 → 8.40，N = 100 → 1.93，N = 250 → 1.31，N = 1000 → 1.07．N1 ≠ N2 のとき (1/N1 − 1/N2)(d−1) の項が誤差を増やす（クラスの不釣り合いによる閾値のずれ．本記録の読み）．
- 結論（Raudys 1998）："There is no unique definition of the 'intrinsic dimensionality' ... An exact definition depends both, on a true distribution of the pattern classes, and on the type of the classifier used."

### 4. 軸との対応
| 軸 | 判定 | 根拠 |
|---|---|---|
| A1 | △ | EDC・Fisher LDF（生の特徴）．線形の頭だが凍結特徴・ロジスティック回帰ではない |
| A2 | ○ | 向きの誤差を 2p*/(δ*²N) の因子で書き，p* を実効次元として定義．閾値の項は無い（p, N → ∞ で落ちる） |
| A3 | △ | cos²θ = 1/T_μ = N/(N + 2p*/δ*²) が式 (1) に含まれる（本記録の推論）．ソースの価値の換算は無い |
| A4 | × | — |
| A5 | △ | N = 5 の数値例があるが，式は p, N が大きい枠組み |
| A6 | × | 同じ分布 |
| A7 | × | — |
| A8 | × | — |
| A9 | × | 正則化判別は名前だけ |
| A10 | × | シミュレーションのみ |
| A11 | ○ | 式 (1) の値とシミュレーションの一致を 2 例で報告 |

### 5. 穴
- 扱っていないこと：閾値の誤差の項，罰則つきの推定で p* がどう縮むか，ドメインのずれ，実データ．
- 限界として書くこと：p* の定義が分類器に依存すること（上の引用）．Raudys & Jain 1991 の原著の限界の文は未読．

### 6. 本研究との差の書き方
「Raudys の小標本の理論（Raudys & Jain 1991；Raudys 1998）では，最近傍平均の線形判別の誤り率が Φ{−(δ*/2)/√(1 + 2p*/(δ*²N))} で近似され，p* = (μ′μ)²tr(Σ²)/(μ′Σμ)² が実効次元として入る．本研究の向きの誤差 q/K は 2p*/(δ*²N) と同じ量で，q を凍結特徴の上の L2 罰則つきロジスティック回帰で実測した（16〜22）点と，閾値の誤差 1/(2K) を別に数えた点が異なる．」
式の対応：主張 A の q/K = Raudys の 2p*/(δ*²N)．主張 B の e² = n/(n+q) = Raudys の 1/T_μ を n で書いたもの（本記録の推論）．

### 7. 注意
- 式 (1) の T_μ と T_Σ は頁の画像で確かめた．導出は未読（Raudys 1967 と Raudys & Jain 1991）．
- 「向きの項だけ」という読みは本記録の推論．Raudys & Jain 1991 の原著に閾値の項があるかは未確認．
- IEEE TPAMI 13 の号番号は要確認．

---

## 3. DARE（Rosenfeld, Ravikumar, Risteski 2022）

### 1. 書誌と読んだ範囲
- Rosenfeld, E., Ravikumar, P., Risteski, A., "Domain-Adjusted Regression or: ERM May Already Learn Features Sufficient for Out-of-Distribution Generalization," arXiv:2202.06856v2 [cs.LG]，2022-10-28（本文の注記で確認）[arXiv](https://arxiv.org/abs/2202.06856)．掲載先：NeurIPS 2022 Workshop on Distribution Shifts（DistShift）と NeurIPS の仮想会場の頁と ML Anthology が示す（○ 検索結果 [neurips.cc](https://neurips.cc/virtual/2022/60545)，[mlanthology](https://mlanthology.org/neuripsw/2022/rosenfeld2022neuripsw-domainadjusted)）．本会議・雑誌への掲載は見つからず（**要確認**）．
- 読んだ範囲（◎）：Abstract，§1，§2，§3（目的関数 (1)，実装の Lagrangian，JIT-UDA），§4（模型 (2)，Example 4.1），§5（Assumption 5.1，Theorem 5.2，5.1 節 Assumption 5.3・5.5，Theorem 5.6，5.2 節 Definition 5.7・5.8，Theorem 5.9，5.3 節 Theorem 5.11），§6（Table 1），§7，付録 B.6 の Theorem B.4（5.11 の完全な主張），付録 E.2（Table 3）．飛ばした：B.3〜B.5 の証明，C・D，E.1・E.3 の本体，A（回帰と anchor regression の関係）．

### 2. 設定
- 模型 (2)：潜在 ε = ε0 + b_e（ε0 は平均 0 の任意のドメイン別分布，b_e はドメインの平均），y = 1{β*ᵀε + η ≥ 0}（η はロジスティック雑音），観測 x = A_e ε（ドメイン別の線形変換）．β* は全ドメインで共通．回帰は y = β*ᵀε + η．x は「凍結した網の特徴」と解釈する（"we now consider x to represent the frozen features of the trained network"）．
- 分類器：各ドメインで白色化 Σ_e^{−1/2}(x − μ_e) した特徴の上の 1 つの線形予測器 β（多項ロジスティック損失）．制約：softmax(βᵀΣ_e^{−1/2}μ_e) = 1/k（各ドメインの平均の表現が出力に効かない）．実装は制約を λ の罰則にした Lagrangian．出発点への罰則は無い．
- 標本数の体制：主定理は母集団（"we assume full distribution access to the training domains"）．Theorem 5.11 だけ有限標本で，n_S, n_T = Ω(m(Σ)d²)．ターゲットのラベルは使わない（DG と UDA）．
- ずれの型：ドメインごとの平均 b_e と線形変換 A_e（共分散のずれ）．Assumption 5.1 で A_e の右特異ベクトルが共通．ラベルの比率のずれは扱わない．

### 3. 示したこと
- 図 1・§2（実験）：DomainBed（Office-Home，PACS，VLCS，DomainNet）で ERM で微調整した ResNet-50 の特徴を固定し，テストドメインのデータも使って線形の頭だけ学び直す（"cheating"）と，網全体を学び直す場合に迫る．"cheating only for the linear classifier rivals cheating for the whole network"．事前学習のみの特徴では成り立たない（微調整の効果）．これから "in settings where we have access to a small amount of data from the target distribution, simply retraining the last linear layer on this data will suffice for excellent performance" と推測する（**量的な主張ではなく推測**）．
- Theorem 5.2：母集団の DARE の解は Π̂β*（回帰），αΠ̂β*（ロジスティック，ε がガウス，α ∈ (0,1]）．Π̂ はドメインの平均 b_e の張る空間の直交補空間への射影．つまり制約は「平均が変わる方向を予測器から消す」ことに等しい．
- Theorem 5.6：Assumption 5.3（部分空間の近似回復，定数 B）・5.5・‖b_e0‖ ≤ ρ の下で sup R_e0(β̂) = (1 + ρ²)(‖β*‖² + 2B‖β*_Π̂‖‖β*_{I−Π̂}‖)，DARE の解は最小最大．証明の筋："We decompose the excess risk into error on the target mean β*ᵀb_e0 and error on the residuals β*ᵀε0, bounding the two separately."（**平均の誤差と残差の誤差の分解はここ**．ただし「切片」ではなく，テストドメインの平均が予測に及ぼす誤差）．
- Theorem 5.9：E 個のドメインを見たとき，E ≥ rank(Σ_b) なら最小最大の予測器を確実に回復．そうでなければ超過リスクは O(√(r(Σ_b)/E))（r は実効ランク tr(Σ)/‖Σ‖）．
- Theorem 5.11（JIT-UDA）：ラベル無しのターゲット標本 n_T で共分散を推定して調整すると，超過二乗リスクは O(d²‖μ_T‖²(m(Σ_S)/n_S + m(Σ_T)/n_T))（完全版 B.4 では (1 + ‖μ_T‖²)[E[η²]·d/n_S/(1 − O(√(d/n_S))) + d²(m(Σ_S)/n_S + m(Σ_T)/n_T)]）．m(Σ) = λ_max³/λ_min．**d² が入るので n ≫ d² が要る**．著者："Experimentally, we found that DARE does not outperform methods specifically intended for UDA, possibly due to the fact that n ≪ d²"．
- 実装上の知見（§3）：「テスト時の平均を推定して引く」案は試して捨てた："We also consider estimating the test-time mean rather than enforcing invariance, but this results in substantially worse accuracy—the environment feature means vary quite a bit, so the estimate is usually inaccurate."（DG の設定．テストドメインの標本は無いので，訓練ドメインの平均から推定したと読める．本記録の推論）．ラベル無しのターゲット標本があるときは制約を外して経験共分散を使うのが良い（JIT-UDA）．
- Table 1（固定特徴の上の線形予測器の正解率，3 回の平均，一部）：Office-Home A：ERM 61.4 → DARE 64.7，C 52.1 → 55.4，P 74.6 → 77.5，R 75.8 → 79.2．PACS A 84.3 → 85.6．VLCS L 66.1 → 62.5（劣る）．DomainNet s 48.6 → 51.4．IRM・GroupDRO・Reweighted ERM も固定特徴の上では ERM に勝つ．
- Table 3（付録 E.2）：ドメインごとに最適な線形分類器（ロジスティック回帰）どうしの平均余弦．調整前 → 調整後：Office-Home A 0.596 → 0.693；PACS C 0.776 → 0.896；VLCS C 0.200 → 0.598，L 0.436 → 0.723，S 0.231 → 0.674，V 0.210 → 0.591．**ドメインごとの向きの余弦を測り，白色化で上がることを示す（本研究の e に近い量）**．
- 記号の対応：b_e ↔ 本研究の切片のずれの源（ドメインの平均のずれ）．β* ↔ 真の向き．Table 3 の余弦 ↔ e．K・q・d に当たる量は無い．

### 4. 軸との対応
| 軸 | 判定 | 根拠 |
|---|---|---|
| A1 | ○ | 凍結した深層特徴の上の多項ロジスティック回帰（白色化つき） |
| A2 | △ | リスクを「平均の誤差」と「残差の誤差」に分けて抑える（Theorem 5.6 の証明の筋）．K で減る式は無い．Theorem 5.11 の率は d²/n |
| A3 | × | ソースの価値の換算は無い．Table 3 で向きの余弦は測る |
| A4 | △ | 制約でドメインの平均を予測から消す（暗黙のドメイン別切片を禁じる側）．合併で切片が直る割合は扱わない |
| A5 | × | DG と JIT-UDA．ターゲットのラベルは使わず，有限標本の定理は n ≳ d² |
| A6 | ○ | 平均 b_e と線形変換 A_e（共分散）のずれ．ラベル比率は無し |
| A7 | × | 画像のみ |
| A8 | × | — |
| A9 | × | 出発点への罰則は無い |
| A10 | ○ | DomainBed の 4 データ集合，3 回 |
| A11 | △ | 理論の率と実測の量的な照合は無い．Table 2 で共分散の推定の誤差を報告 |

### 5. 穴
- 扱っていないこと：ターゲットのラベル K 本で学び直すときの誤差（推測のみ），切片と向きを分けた誤差の式，実効次元，ソースの合併の得と損，複数モダリティ．
- 限界として自分で書いていること："Assumption 5.1 and the constraint in Equation (1) are not trivial. Domain generalization is an exceptionally difficult problem"；Lemma B.1（制約つきロジスティック回帰の閉じた解）は "we prove this lemma only for Gaussian z"；JIT-UDA は "n ≪ d²" で UDA 専用の方法に勝てない．
- 今後の課題："JIT-UDA presents a promising direction for future theoretical research: it is much weaker—and arguably more realistic—to assume access to unlabeled samples only at test-time."；"future work would benefit from modularity, working to improve representation learning and robust classification/regression separately"．

### 6. 本研究との差の書き方
「DARE（Rosenfeld ら 2022）は，凍結した特徴をドメインごとに白色化し，ドメインの平均が出力に効かない制約のもとで 1 つの線形の頭を学ぶ．その保証は超過リスクをドメインの平均の誤差と残差の誤差に分けて抑え（Theorem 5.6），ラベル無しのターゲット標本があれば学び直し無しで調整できる（Theorem 5.11．率は d²/n）．本研究のラベル無しの中心化は DARE の平均の調整と同じ操作であるが，本研究は少数のターゲットラベル K 本が直す量（切片 1/(2K)，向き q/K）と中心化が置き換える量を同じ分割で測る点，および K ≪ d² の体制を対象にする点が異なる．」
式の対応：本研究の式は DARE のどの式も書き直したものではない．DARE の「平均の誤差」が本研究の切片のずれ o に，Table 3 の余弦が e に当たる（本記録の対応づけ）．

### 7. 注意
- 「テスト時の平均の推定は劣った」という記述は DG の設定（テスト標本無し）で，本研究のようにターゲットのラベル無し標本から平均を引く場合とは条件が違う（本記録の推論．付録 E でどう推定したかは未読）．
- 掲載先は本文に無い．Theorem 5.11 の定数の形は pdftotext で崩れた箇所を B.4 と照らして復元した．

---

## 4. Kini, Paraskevas, Oymak, Thrampoulidis 2021（NeurIPS 2021，arXiv 2103.01550）

確かめた水準：◎ 本文（下請けの精読記録．引用の存在を grep で確認）．[arXiv](https://arxiv.org/abs/2103.01550)

### 1. 書誌と読んだ範囲

**書誌（◎ 本文で確認）**
- 題：Label-Imbalanced and Group-Sensitive Classification under Overparameterization
- 著者：Ganesh Ramachandra Kini（UC Santa Barbara），Orestis Paraskevas（UC Santa Barbara），Samet Oymak（UC Riverside），Christos Thrampoulidis（Univ. of British Columbia）
- arXiv の注記：arXiv:2103.01550v3 [cs.LG]，2021 年 11 月 8 日
- 掲載先：1 頁目の脚注に "35th Conference on Neural Information Processing Systems (NeurIPS 2021), Sydney, Australia." とある
- コード：github.com/orparask/VS-Loss（参考文献 [cod]）
- pdftotext の出力は 47 頁ぶん．本文 11 頁（1〜6 節）＋参考文献＋付録 A〜G

**読んだ範囲**
- 本文：1 節（動機・貢献・関連研究），2 節（設定，VS 損失 Eq. (1)(2)(3)，CS-SVM Eq. (4)），3 節（Theorem 1，Remark 1，Observation 1），4 節（ガウス混合の模型，Theorem 2，公平性のトレードオフ），5 節（実験 Table 1，Fig. 4，Table 2），6 節（結び）
- 付録：A（深層網の実装と Table 3，MNIST），B（Waterbirds の GS-SVM 実験），C.1〜C.5.1（ガウス混合の追加実験，**C.2 の L2 正則化つき勾配降下の実験**，理論の検証，アンダーサンプリング），D.1（Lemma 1，Theorem 3 の主張と証明，Theorem 1 の導出），D.2（Theorem 4 の主張），D.3（Theorem 5 の主張と証明の冒頭），D.4，E.1（Theorem 6，Eq. (31)〜(34)），E.1.1（δ⋆ の推定の処方 Eq. (36)），E.2（Lemma 2），E.3 の冒頭（Eq. (39)(40)），F.1，F.2，F.4，F.5（Proposition 1），G（Theorem 7 の主張），G.2（Proposition 2）
- 飛ばした範囲：D.1.3（Lemma 1 の証明），Theorem 4 の証明，Theorem 5 の証明の後半，E.3 の Eq. (40) 以降（Theorem 6 の証明），F.3（Theorem 2 の証明．CGMT の計算），G.1（Theorem 7 の証明）．図の中身は見えないので，図の情報は見出し文とキャプションだけから読んだ

### 2. 設定

**模型と分類器**
- 入力 x∈R^d（論文の d は次元．本研究の p に相当），ラベル y∈{−1,+1}（+1 が少数クラス），群 g∈[K]（論文の K は群の数で，本研究の K とは無関係）．多クラスは付録 D.2 と実験で扱う
- 分類器は固定の特徴写像 h(x)∈R^p の上の線形モデル f_w(x)=⟨w,h(x)⟩（2 節，Remark 3）．**Theorem 1/3/5 のモデルには切片が無い**．Theorem 2 の CS-SVM には罰則なしの切片 b がある
- 損失は VS 損失（Eq. (1)）：ℓ_VS = ω_y·log(1+e^{ι_y}·e^{−Δ_y·y·f_w(x)})．ω_y は重み，ι_y は加法的なロジット補正（LA 損失 [MJR+20] に相当），Δ_y は乗法的な補正（CDT 損失 [YCZC20] に相当）
- **罰則は無い．**理論は「学習誤差 0 を超えてなお損失を 0 に近づけ続ける」terminal phase of training（TPT）を対象にする．罰則つきは付録 C.2 の数値実験だけ
- 比較対象は CS-SVM（Eq. (4)）：min‖w‖₂ s.t. ⟨w,h(x_i)⟩≥δ（y=+1），≤−1（y=−1）．δ はクラス間のマージン比

**標本数の体制**
- Theorem 1/3/4/5：任意の線形分離可能な有限の学習集合（標本数は固定．分布の仮定なし）．主張はノルム制約 R→∞ または勾配流 t→∞ の極限
- Theorem 2/6/7：ガウス混合で **n,d→∞，d/n=γ>γ⋆ の比例極限**（4 節 "as n,d→∞ with d/n=γ>γ⋆"）．少数クラスの標本数は πn で，n と一緒に無限大になる．クラスごとの標本数が有限の少数ショットの理論は無い

**ずれの型**
- ラベルの比率の不均衡（学習の事前確率 π と評価指標の balanced error）
- 群の不均衡：各クラスの中に平均 µ_g と分散 σ_g が違う群が混ざる（4 節，付録 G の "x|(y,g)∼N(yµ_g,σ_g I_d)"）．群は学習時には既知，テスト時には未知（2 節）．**学習分布とテスト分布は同じ**で，ドメイン間の転移ではない

### 3. 示したこと

記号の対応：論文の b̂/‖ŵ‖ ↔ 本研究の o（閾値のずれ），論文の µ_±ᵀŵ/‖ŵ‖ ↔ 本研究の (d/2)·e（平均が対蹠的で ‖µ‖=d/2，e が ŵ と µ の余弦のとき），論文の γ=d/n ↔ 本研究の p/(4K) に相当する比（ただし論文には実効次元 q に当たるものは無い），論文の δ=Δ₋/Δ₊ ↔ 切片の補正の量（Lemma 2 で (δ−1)/(δ+1) に換算される）．

**(a) Theorem 1（本文）／Theorem 3（付録 D.1，一般形）：加法的な補正は分離可能なデータでは消える**
- 仮定：(i) 2 クラスとも 1 本以上ある有限の学習集合，(ii) 特徴写像 h の上で線形分離可能（∃w: y_i wᵀh(x_i)≥1 ∀i），(iii) 線形モデル f_w=⟨w,h(x)⟩（切片なし），(iv) VS 損失で Δ_±>0，ω_±≥0（Theorem 3 では ω_i>0），ι_± は任意
- 主張：ノルム制約つきの最小化 w_R=argmin_{‖w‖≤R} L_n(w) について，R→∞ で w_R/‖w_R‖ が CS-SVM の解 ŵ_δ/‖ŵ_δ‖（δ=Δ₋/Δ₊）に収束する（Eq. (12)）
- 原文："the theorem makes clear that ω± and ι± become ineffective in the TPT as they all result in the same SVM solutions. On the other hand, the multiplicative parameters ∆± lead to the same classifier as that of CS-SVM"（3.1 節）
- 仕組み（証明 D.1.2，Eq. (18)〜(20)）：最大マージン方向での損失の上界は ω_max·n·e^{ι_max−R̄}，それ以外の方向での下界は ω_min·log(1+e^{ι_min−R̄(1−ε)})．ι と ω は e^{ι_max−ι_min} という**有限の係数**にしか入らず，e^{R̄} がそれを超えた時点で最大マージン方向が勝つ．一方 Δ_i は制約 y_i wᵀx_i≥1/Δ_i（Eq. (9)）の右辺を変えるので極限の向きそのものを変える
- 多クラス版：Theorem 4（付録 D.2．交差エントロピー型 VS 損失 Eq. (21)，多クラス CS-SVM Eq. (22)）
- 最適化アルゴリズムを入れた版：Theorem 5（付録 D.3）．**指数損失** ℓ(t)=e^{−t} の VS 版への勾配流は w(t)=ŵ·log(t)+ρ_t（残差 ρ_t は有界）で，向きは CS-SVM に収束する．ロジスティック損失については "one would expect that insights also apply to logistic loss" と述べるにとどまる
- Remark 1：Theorem 1 は [RZH03]（Rosset, Zhu, Hastie 2003）の「正則化つき ERM で正則化パラメータが 0 に向かうと正規化した解が SVM に収束する」の VS 損失・CS-SVM への拡張だと位置づけている．**正則化が有限の場合の主張は無い**
- 乗法的な補正が初期に害になること（Observation 1，3.2 節）：f_w=0 の初期では勾配 ∇ℓ_VS=−ω_y Δ_y σ(−Δ_y y f+ι_y)·y·h(x) において Δ_y は ω_y と同じ役割になり，Theorem 1 が要求する Δ₊<Δ₋ は少数クラスの勾配を小さくする．ι₊>ι₋ はこれを打ち消して収束を速める．これが VS 損失（両方を入れる）の動機

**(b) Lemma 2（付録 E.2）：切片つきの CS-SVM は SVM に閾値の後処理の移動を足したもの**
- 硬マージン SVM の解 (ŵ₁,b̂₁) に対し，ŵ_δ=((δ+1)/2)ŵ₁，b̂_δ=((δ+1)/2)b̂₁+(δ−1)/2 が CS-SVM（Eq. (4)，切片つき）の最適解．予測は ŷ=sign((ŵ₁ᵀx+b̂₁)+(δ−1)/(δ+1))
- 原文："The term (δ−1)/(δ+1) can be seen as an additive form of post-hoc weight normalization to account for class imbalances. In the literature this post-hoc adjustment of the threshold b of standard SVM is often referred to as boundary-movement SVM (BM-SVM)"
- 本記録の推論：損失の中の加法的な補正 ι_y は発散するノルムに飲み込まれて効かないが，**学習後の正規化した分類器の閾値を動かす加法的な補正は効く**．つまり論文の「乗法的な補正」は，結果としては切片の補正と同じ分類器を与える．これは本研究の「切片（閾値）の補正が主要な手」と矛盾しない

**(c) Theorem 2（4 節）：ガウス混合での CS-SVM の balanced error の厳密な漸近**
- 仮定：x|y∼N(µ_y,I_d)（F.4 で共通の Σ≠I に拡張．白色化で同じ定理に帰着），π=P{y=+1}，h(x)=x，罰則なしの切片 b，δ>0 固定，n,d→∞ かつ d/n=γ>γ⋆（γ⋆≤1/2 は分離可能性の相転移の閾値．Proposition 1 Eq. (64)）
- 主張：(‖ŵ_δ‖, µ₊ᵀŵ_δ/‖ŵ_δ‖, µ₋ᵀŵ_δ/‖ŵ_δ‖, b̂_δ) が確率収束で (q_δ, e₁ᵀVSρ_δ, e₂ᵀVSρ_δ, b_δ) に収束し，R₊→Q(e₁ᵀVSρ_δ+b_δ/q_δ)，R₋→Q(−e₂ᵀVSρ_δ−b_δ/q_δ)．ここで (q_δ,ρ_δ,b_δ) は η_δ(q,ρ,b)=E[(G+E_YᵀVSρ+(bY−Δ_Y)/q)²₋]−(1−‖ρ‖²)γ=0 を満たす一意の 3 つ組（Eq. (42)．式の復元は pdftotext の崩れのため中程度の自信）
- 読み方（原文）："b_δ is the asymptotic the intercept, q_δ⁻¹ is the asymptotic classifier's margin 1/‖ŵ_δ‖₂ to the majority, and ρ_δ determines the asymptotic alignment of the classifier with the class mean"．本記録の推論：ρ_δ は正規化した分類器のうち平均の張る部分空間に乗る成分で，1−‖ρ_δ‖² が平均と無関係な方向に漏れた成分（本研究の「向きの誤差」に相当）．それは γ=d/n の関数として決まる
- 条件つき誤り率の導出は付録 F.1 の番号なしの式：R₊=Q(µ₊ᵀŵ/‖ŵ‖+b̂/‖ŵ‖)．**本研究の ½[Φ(d/2+o)+Φ(d/2−o)] はこの式を対蹠的な平均（µ₊=−µ₋，F.4）に書き直したもの**で，Q(t)=1−Φ(t) に注意すれば R_bal=½[Q(s·ρ̃+o)+Q(s·ρ̃−o)]（s=‖µ‖=d/2，ρ̃ は余弦，o=b̂/‖ŵ‖）
- Theorem 7（付録 G）：群つきガウス混合での GS-SVM の群別誤りと DEO の漸近．Proposition 2 は分離可能性の閾値
- Theorem 6（付録 E.1）：balanced error を最小にする δ⋆ の陽な式 Eq. (33)(34)：δ⋆=(ℓ₋−ℓ₊+2q₁⁻¹)/(ℓ₊−ℓ₋+2q₁⁻¹)₊（ℓ_± は δ=1 の SVM の量）．この δ⋆ で R₊=R₋=Q((ℓ₊+ℓ₋)/2) になる．Lemma 2 を使うと δ の最適化が閾値 1 次元の最適化 Eq. (32) に落ちる．E.1.1 には標本平均と SVM の解から δ⋆ を推定する処方 Eq. (36)
- C.5.1：多数クラスをアンダーサンプリングした SVM は γ を γ/(2π) に置き換えた Theorem 2 で予言できる（Eq. (7)）

**(d) 正則化・早期停止に関する記述（本研究の問いに直接関わる）**
- 付録 C.2（数値実験のみ，定理は無い）：p=d=50，n=30，π=0.1 のガウス混合で，重み減衰 β∈{0,10⁻³,10⁻²} の勾配降下 w_{t+1}=(1−β)w_t−η∇L．"The weight-decay factor was kept small enough to still achieve zero training error."
  - "The classifier norm plateaus when trained with regularization (while it increases logarithmically without regularization; see Theorem 5)."
  - "The final balanced error performance of such a classifier closely matches the balanced error produced by a classifier trained without regularization but with training stopped early at that iteration for which the classifier-norm is equal to ζ"
  - **"If early-stopped (appropriately) before entering TPT, LA-loss can give better balanced performance than CDT-loss. In view of the above mentioned mapping between weight-decay and training epoch, the use of weight decay results in same behavior."**
- 付録 C.1："the additive offsets ι_y's improve performance in the non-separable regime."
- 本記録の推論：論文は「罰則の強さでこの条件を避けられる」ことを，実験では認め，理論では扱っていない．Theorem 1 が適用されるのは正則化が 0 に向かう極限（Remark 1）であり，有限の λ の L2 罰則つきロジスティック回帰では解のノルムが有限にとどまるので，切片や加法的な補正は飲み込まれない．「どれだけの λ なら十分か」の式は無い

**(e) 実験の設定と主な数値**
- Fig. 2(a)（3.1 節）：ガウス混合，n=100，d=300，π₊=0.1（少数クラスは約 10 本），‖µ₊‖=2‖µ₋‖=4，使う特徴数 p を 5〜300 で変えて γ=p/n を動かす，25 回平均，テスト 10⁴．学習誤差は γ≳0.45 で 0．結論："CDT-loss has better balanced-error performance compared to the LA-loss when both trained in TPT ... (a) LA-loss performs the same as SVM, and, (b) CDT-loss performs the same as CS-SVM."．Theorem 2 の実線が実測に合う（C.1："sharply predicts ... despite the moderate dimension of d=300"）
- Fig. 2(b,c)（C.3）：d=50，n=30，π=0.1，GD 学習率 0.1 で 10⁶ 反復．VS は CDT より桁違いに速く低い R_bal に達する
- Table 1（5.1 節，CIFAR-10/100，不均衡比 N_max/N_min=100 の LT と STEP，ResNet-32，200 エポック（STEP は 300），重み減衰 2×10⁻⁴，5 回平均，± は 95% 信頼区間，**balanced なテスト集合を検証集合として調律に使う**）．balanced validation の top-1 正解率（%）：
  - CIFAR-10 LT-100：CE 71.94，wCE 72.6，LDAM-DRW 77.03，LA(τ⋆) 80.81，CDT(γ⋆) 79.55，VS 80.82
  - CIFAR-10 STEP-100：CE 62.69，LDAM-DRW 76.92，LA 78.23，CDT 73.26，VS 79.10
  - CIFAR-100 LT-100：CE 38.82，LDAM-DRW 42.04，LA 42.87，CDT 42.57，VS 43.52
  - CIFAR-100 STEP-100：CE 39.49，LDAM-DRW 45.36，LA 45.69，CDT 44.12，VS 46.53
  - 調律値（Table 3）：ι_y=τ·log(N_y/N_tot)，Δ_y=(N_y/N_max)^γ（Eq. (5)）．LA の τ⋆=2.25/2.25/1.375/0.875，CDT の γ⋆=0.4/0.3/0.1/0.1，VS の (τ⋆,γ⋆)=(1.25,0.15)/(1.5,0.2)/(0.75,0.05)/(0.5,0.05)
- Fig. 4（CIFAR-10 LT-100）：γ=0.15 の CDT は 200 エポックで学習正解率 ≈93%，τ=−0.5 を足した VS は ≈97%．300 エポックまで回すと両者の balanced 正解率は近づく（"in the interpolating regime what dominates the performance are the multiplicative adjustments"）
- Table 2（5.2 節，Waterbirds，ImageNet 事前学習の ResNet50，300 エポック，10 回平均．部分群の比率 0.22/0.012/0.038/0.73）：Symm. DEO／balanced 正解率／最悪部分群正解率：CE 25.3/84.9/68.1，Group LA 24.0/84.2/70.1，Group CDT 18.5/87.2/75.4，Group VS 18.1/88.1/76.7，CE+DRO 16.3/88.7/75.2，Group LA+DRO 16.3/88.7/74.3，Group CDT+DRO と Group VS+DRO は 11.7/90.3/79.9 と 11.8/90.2/78.9（pdftotext で 2 行の対応が崩れており，どちらがどちらかは要確認．本文は "Group-CDT/VS have comparable performances"）
- 付録 A.2（MNIST，7 対その他，d=784，π=0.1，200 回平均）：線形と ReLU ランダム特徴（n=300）で，δ⋆ の推定を使う CS-SVM が SVM と δ=((1−π)/π)^{1/4} の両方に勝つ．Fig. 6 で LA→SVM，VS→CS-SVM への収束を確認
- 付録 B.2（Waterbirds の事前学習 ResNet-18 特徴 d=512 の上のランダム特徴 N=500〜10000，n=4795，部分群 3498/184/56/1057）：GS-SVM は SVM より最悪部分群誤りが小さく，誤分類率の損は小さい．部分群 56 本ずつに間引いた SVM は最悪部分群誤りは最小だが誤分類率が 3 倍超
- 付録 C.5（Fig. 12：d=500，π=0.05，100 回）：Theorem 2 の予言が数百次元でも一致．δ=δ⋆ の CS-SVM は balanced と誤分類率の両方で SVM とリサンプリング SVM に勝つ

### 4. 軸 A1〜A11 との対応

| 軸 | 判定 | 根拠 |
|---|---|---|
| A1 凍結した特徴の上の線形の頭 | ○ | 理論は固定の特徴写像 h(x) の上の線形モデル（Remark 3，6 節 "limited to settings with fixed features"）．ただし学習器は罰則なしで TPT まで回した VS 損失＝CS-SVM で，L2 罰則つきロジスティック回帰ではない |
| A2 切片と向きの分離，K で減る誤差の式，次元の入り方 | △ | Theorem 2 は切片 b_δ と向きの整合 ρ_δ（と 1−‖ρ‖²）を分けて極限値を出す．γ=d/n の関数であり，1/K や q/K の形の有限標本の誤差の式は無い．実効次元の概念も無い |
| A3 ソースの価値をラベル本数に換算 | × | ソースドメインが無い．余弦と標本数の関係は ρ_δ(γ) の暗黙の形でしか現れない |
| A4 合併で切片が直る割合，ドメイン別切片 | × | ドメインの合併を扱わない．Lemma 2 の (δ−1)/(δ+1) は不均衡に対する閾値の移動で，ドメイン別ではない |
| A5 固定の少数ショット | × | 理論は比例極限（n,d→∞）．実験では少数クラス 3〜10 本（Fig. 2，C.2）でも理論が合うと報告するが，有限 K の定理は無い |
| A6 ドメインのずれ | △ | 群ごとに平均 µ_g と分散 σ_g が違う混合（平均のずれ型）を扱うが，学習とテストは同じ分布．ラベル比率のずれは「学習は π，評価は balanced」という形で暗黙に入る（本記録の推論） |
| A7 マルチモーダル | × | 単一の特徴 |
| A8 弱いラベル・疑似ラベル | × | 扱わない |
| A9 出発点への L2 罰則（リッジ，仮説転移） | × | 原点への重み減衰の実験（C.2）だけ．仮説転移は無い |
| A10 実データでの検証 | ○ | CIFAR-10/100（深層網，5 回），Waterbirds（ResNet50，10 回；ランダム特徴，10 回），MNIST（200 回） |
| A11 予言と実測の照合 | ○ | Theorem 2/6/7 の式と人工データの実測を Fig. 2(a)，9，11，12，13，15 で照合（人工データのみ）．実データでは収束先（SVM か CS-SVM か）の定性的な照合（Fig. 6） |

### 5. 穴

**扱っていないこと（本記録の整理）**
- 有限の標本数での誤差の縮み方（1/K，q/K の類）．すべての量は比例極限の値
- 有限の L2 罰則の下での理論．Theorem 1 は正則化が消える極限（Remark 1），C.2 は実験のみ
- Theorem 1/3/5 のモデルに切片が無い．罰則なしの切片を持つロジスティック回帰で加法的な補正がどうなるかは述べていない（Theorem 2 の CS-SVM には切片がある）
- Theorem 5（勾配流）は指数損失で，ロジスティック損失は予想にとどまる
- クラスごとに違う共分散（F.4 は共通の Σ のみ．Theorem 7 の群別分散はスカラー σ_g）
- ソース／ターゲットの転移，複数ドメインの合併，マルチモーダル，弱いラベル

**限界として自分で書いていること（原文）**
- "The current theory is limited to settings with fixed features. While this assumption is prevailing in most related theoretical works [...], it is still far from deep-net practice where (last-layer) features are learnt jointly with the classifier."（6 節）
- "Fisher consistency is only relevant in the large sample size limit. Instead, we focus on overparameterized models."（1.2 節）
- "we keep a balanced test set and in addition to evaluating our models on it, we treat it as our validation set and use it to tune our hyperparameters. More sophisticated tuning strategies (perhaps using bi-level optimization) are deferred to future work."（5.1 節）
- "For simplicity, we consider a VS-type adjusted exponential loss ℓ(t)=e^{−t}, rather than logistic loss ... one would expect that insights also apply to logistic loss."（D.3）
- "we caution that the GS-SVM hyper-parameters are in general harder to interpret as 'margin-ratios'."（2.1 節）

**今後の課題（原文）**
- "We suspect the VS-loss and our better understanding on the individual roles of different hyperparameters can benefit NLP and computer vision applications; we expect future work to undertake this opportunity with additional experiments."（6 節）
- "it is of interest to extend our theory to other fairness metrics of interest."（6 節）
- "We expect that, with more careful tuning of the hyper-parameters δ(y,g), GS-SVM can eventually achieve even lower sub-group errors ... We leave this to future work."（B.2）

### 6. 本研究との差の書き方（関連研究の案）

案 1（切片の補正の位置づけ）：
「Kini ら [NeurIPS 2021] は，線形分離可能な学習データで正則化なしに損失を 0 に近づけ続けると，損失に足したクラスごとの加法的なロジット補正は最大マージン解の向きに影響せず（Theorem 1），乗法的な補正だけがコスト感応 SVM の向きを変えることを示した．同時に，切片を持つコスト感応 SVM は通常の SVM の閾値を (δ−1)/(δ+1) だけ動かしたものと同じである（Lemma 2）．本研究は L2 罰則つきロジスティック回帰を扱うので解のノルムは有限にとどまり，切片の補正はこの意味で有効に働く体制にある．Kini らも重み減衰や早期停止のもとでは加法的な補正が有利になりうると実験で報告している（付録 C.2）が，有限の罰則の下での理論は与えていない．」

案 2（正解率の式の由来）：
「本研究の 2 クラスの正規の雲の模型の正解率 ½[Φ(d/2+o)+Φ(d/2−o)] は，線形分類器の条件つき誤り率 Q(µ_±ᵀŵ/‖ŵ‖ ± b̂/‖ŵ‖)（Kini ら付録 F.1，Theorem 2 の R_± の定義）を，平均が対蹠的で向きの余弦が e の場合に書き直したものである．Kini らはこの量の n,d→∞，d/n=γ の比例極限での値（切片 b_δ と向きの整合 ρ_δ）を導いたのに対し，本研究はクラスごとの標本数 K が数本の有限の体制で，切片の誤差が 1/(2K)，向きの誤差が q/K で縮むという形に書き，さらにソースの価値を K に換算する（n=qe²/(1−e²)）．」

本研究の式との対応（要点）：
- 本研究の式は Kini らのどの式も書き直したものではなく，両者が共通に使う初等的な恒等式（正規の雲＋線形分類器の誤り率）から出発している．Kini らの固有の貢献（Theorem 1 の収束先，Theorem 2 の比例極限の 3 つ組，Theorem 6 の δ⋆ の式）はいずれも本研究の式には入っていない
- 本研究の「切片の誤差 1/(2K)」「向きの誤差 q/K」に対応する式は Kini らに無い．Kini らの「向きの誤差」に当たる 1−‖ρ_δ‖² は γ=d/n の関数で，有限 K の展開ではない
- Kini らの「加法的な補正は効かない」は正則化が 0 の極限の主張で，本研究の設定（有限の λ）には適用されない．これを本研究の設定の違いとして明示するとよい

### 7. 注意

- **本記録の推論**である箇所：(i) Lemma 2 の読み方「損失内の加法補正は消えるが学習後の閾値の移動は効く」は論文が明示的に述べた対比ではなく，Theorem 1 と Lemma 2 を並べた本記録の整理．(ii) 1−‖ρ_δ‖² を「向きの誤差」と読んだこと．論文は "ρ_δ determines the asymptotic alignment" としか言わない．(iii) 「有限の λ の L2 罰則なら切片が有効」は Remark 1 と C.2 からの推論で，論文に定理は無い．(iv) 本研究の設定が Kini らの「分離可能な体制」（p≫4K なら γ=p/n>γ⋆≤1/2）にあるという見立ても本記録の推論．(v) balanced error を「ラベル比率のずれ」と読んだ A6 の判定
- 式の復元：Theorem 2 の η_δ の定義（Eq. (42) の中身）は pdftotext で崩れており，(bY−Δ_Y)/q の部分の復元は中程度の自信．Theorem 1 の ω_± の条件が本文では "ω± ≥ 0"，Theorem 3 では "ω_i > 0" と表記が揺れている
- 未読：Theorem 2・4・5（後半）・6・7 の証明の本体（F.3，G.1，E.3 の後半，D.1.3）．主定理の仮定と結論は主張文で確かめた．図の中身（数値の読み取り）は見ていない
- Table 2 の下 2 行（Group CDT+DRO と Group VS+DRO）の数値の対応は pdftotext で崩れているので，引用するなら原 PDF で要確認
- 書誌：NeurIPS 2021 の掲載は 1 頁目の脚注による．頁番号や DOI は本文に無く要確認．[KT21a] と [KT21b] が参考文献で同じ題の重複になっている（論文側の誤りとみられる）

---

## 5. Kumar, Raghunathan, Jones, Ma, Liang 2022 LP-FT（ICLR 2022，arXiv 2202.10054）

確かめた水準：◎ 本文（下請けの精読記録）．[arXiv](https://arxiv.org/abs/2202.10054)

### 1. 書誌と読んだ範囲

**書誌（◎ 本文で確かめた）**
- 題：Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution
- 著者：Ananya Kumar, Aditi Raghunathan, Robbie Jones, Tengyu Ma, Percy Liang（Stanford University, Department of Computer Science）
- 版：arXiv:2202.10054v1 [cs.LG]，2022-02-21（本文 1 頁の注記）
- 掲載先：本文に ICLR の記載は無い（「Published as a conference paper at ICLR 2022」のような見出しが無い arXiv 版）．**ICLR 2022 は要確認**
- 全 54 頁（本文 15 頁 + 参考文献 + 付録 A〜C）

**読んだ範囲**
- 本文：Abstract，1 Introduction，2 Setup，3.1〜3.4（Definition 3.1・3.2，Theorem 3.3，Assumption 3.4，Theorem 3.5，Proposition 3.6，Proposition 3.7），4.1〜4.3（Table 1・2），5 Related work，6 Conclusion（脚注 3・4・6・7・10 を含む）
- 付録：A.2 の冒頭（Theorem 3.3 の再掲 Eq. (A.14) と Lemma A.3 の主張），A.3（Theorem A.8 と証明），A.3.3（Lemma A.13・A.14 と証明の全部），A.4（Theorem A.15 と証明），A.6 の冒頭（Proposition 3.6 の証明），A.7（Proposition A.20，Proposition 3.7 の証明の末尾），B.1〜B.5（Table 3〜9），C
- 飛ばした範囲：A.1（主角の準備），A.2 の証明本体（Lemma A.4〜A.7，不変量の計算），A.3.1〜A.3.2（Lemma A.9〜A.12．主張だけ読んだ），A.5（Lemma A.16），A.6 の補題の証明（Lemma A.17〜A.19）

### 2. 設定

**模型**
- 予測器は 2 層の線形網 f(x) = vᵀBx．B ∈ R^{k×d} が特徴抽出器（事前学習で得た B₀ から出発），v ∈ R^k が線形の頭．**切片（バイアス）は無い**
- 回帰で，損失は二乗誤差（Section 3.1）．分類は理論では扱わない（実験のみ）
- 正解は雑音なしの線形関数 y = v*ᵀB*x（well-specified）．B*・B₀ は行が正規直交．事前学習の質は特徴抽出器の距離 d(B₀, B*) = min_U ‖B₀ − U B*‖₂ ≤ ε で測る（Definition 3.1，Eq. (3.1)．U は回転行列）
- 学習は勾配流（学習率を 0 に近づけた勾配降下）で，明示的な罰則は無い．LP は v だけ，FT は v と B の両方を動かす（Eq. (3.2)・(3.3)）．頭の初期値 v₀ は N(0, σ²I) か 0

**標本数の体制**
- 有限の n だが少数ではない．訓練データ X ∈ R^{n×d} の張る部分空間 S の次元を m とし，1 ≤ m < d − k（過剰パラメータ）．例として d = 10K，k = 100，m = 5K（Section 3.1）
- Assumption 3.4（ID 部分空間の仮定）：ID データは m 次元の部分空間 S 上に密度を持ち，k < m < d − k，**n ≥ m**．ガウスの場合はさらに n ≥ 5m，n ≥ 10 log(1/δ)（Lemma A.14・Theorem A.15）
- 雑音が無いので，標本数が誤差を減らす項は現れない．n は条件（n ≥ m）と，上界の log(n/δ) の因子にだけ入る（本記録の読み．根拠は 3 節の Lemma A.14）

**ずれの型**
- 訓練（ID）データは部分空間 S に乗り，OOD の二次モーメント Σ = E[xxᵀ] は可逆（S の外の方向を持つ）．つまり「テスト分布が訓練データの張らない方向を含む」型のずれ（共変量のずれの一種）．脚注 6 で，OOD 誤差はノルム有界な x の最悪誤差とも書けると述べる
- 平均のずれ・共分散のずれ・ラベル比率のずれ・概念のずれ（w* の変化）はどれも扱わない．w* は ID と OOD で共通

### 3. 示したこと

記号の対応：本論文の k（特徴の次元）↔ 本研究の p．本論文の n（ラベル付き標本の総数）↔ 本研究の K × クラス数（ただし本論文は n ≥ m で数千）．本論文の ε（特徴の誤差）と ϕ（頭の初期値の向きのずれ）には本研究の o・d・q に直接の対応は無い（4 節・6 節で述べる）．

**Theorem 3.3（Eq. (3.7)，付録 Eq. (A.14) に同じ）：FT の OOD 誤差の下界**
- 主張：cos θ_max(R₀, S^⊥) > 0 なら，すべての時刻 t で
  √L_ood(FT(t)) ≥ √σ_min(Σ) · [ cos θ_max(R₀, S^⊥) · min(ϕ, ϕ²/‖w*‖₂) / ( √k · (1 + ‖w*‖₂)² ) − ε ]
  ここで ϕ² = |(v₀ᵀv*)² − (v*ᵀv*)²| は頭の初期値の向きのずれ，R₀ = rowspace(B₀)，S^⊥ は訓練データの張る空間の直交補空間，θ_max は最大主角（2 つの部分空間がどれだけ離れているかを表す角）．分数の形は pdftotext の崩れを前後から復元したもので，分子・分母の配置に自信はある（本文と付録の 2 箇所で同じ崩れ方をしている）
- 日常語で：微調整は，訓練データの張らない方向の特徴を変えない（Lemma A.3，Eq. (3.6)）．そのため OOD で正解するには頭 v が v* に近づく必要があるが，頭と特徴の更新は連動しており（Du et al. 2018 の均衡の不変量），頭を大きく動かすと ID 方向の特徴も大きく歪む．どちらかの方向で誤差が残る
- この下界は n に依存しない．特徴の次元 k は 1/√k で入る（k が大きいほど下界は緩い）．頭の初期値が v* に一致すれば下界は 0 になる（Section 3.2.2 の解釈）
- 著者が認める弱点：ε が「−ε」で入る（特徴が悪いほど下界が緩くなる向き）．理由は「the errors of the pretrained feature extractor d(B₀, B*) and the fine-tuning step can potentially cancel out」（脚注 7 で「very 'unlikely'」と述べる）

**Lemma A.14（Eq. (A.132)・(A.133)）：LP の OOD 誤差の上界**
- 一般の場合：cos θ_max(S, R) > 0 なら確率 1 − δ 以上で √L_ood(LP) ≤ c_δ / cos θ_max(S, R) · d(B₀, B*) · ‖w*‖₂
- 等方ガウス（P_z = N(0, I_m)），n ≥ 5m，n ≥ 10 log(1/δ) の場合：√L_ood(LP) ≤ O( log(n/δ) / (cos θ_max(R, S))² · d(B₀, B*) · ‖w*‖₂ )．証明では σ_min(Z) ≥ O(√n)（Eq. (A.157)），σ_max(Z)² ≤ O(n log(n/δ))（Eq. (A.158)）の比から log(n/δ) が出る
- 日常語で：凍結した特徴の上の最小二乗の頭の OOD 誤差は，特徴の誤差 ε に比例する．雑音が無いので，ラベル数で減る項は無い．log(n/δ) は上界の技術的な因子で，n が増えると上界はわずかに緩くなる（本記録の推論：実際の誤差が n で増えるという主張ではない）
- 本研究との関係で重要：**これが本論文で唯一の「線形の頭の誤差の式」であり，少数ラベルでの誤差の式ではない**

**Theorem 3.5（Eq. (3.9)）／Theorem A.8（Eq. (A.97)）：LP と FT の比**
- cos θ_max(R*, S) ≠ 0 かつ cos θ_max(R*, S^⊥) ≠ 0 なら，B₀ → B* のとき L_ood(LP) / inf_t L_ood(FT(t)) → 0（確率収束）．特徴が良くなるほど LP が FT に対して相対的に有利になる

**Theorem A.15（Eq. (A.170)・(A.171)）：ガウスの場合の有限版**
- d(B₀, B*) < O( cos θ_max(R*, S^⊥) · (cos θ_max(R*, S))² · δ² / ( √k · log(n/δ) ) ) なら，確率 1 − δ 以上で L_ood(LP) < inf_t L_ood(FT(t))
- 特徴の次元 k が大きく，n が大きいほど，LP が勝つための ε の閾値は小さくなる（本記録の読み）．注意：Eq. (A.171) の右辺は pdftotext で「inf_t L_ood(v_lp^{∞i}, B₀^i)」と見え，FT のはずの箇所が LP になっている．原本の誤植か変換の崩れかは判別できない

**Proposition 3.6：ID では FT が勝つ**
- w* ∉ R₀ かつ cos θ_max(S, R_aug) ≠ 0 なら，FT は ID 損失 0，LP は ID 損失 > 0（訓練データが S を張るので，FT は S 全体で正解する．付録 A.6）

**Proposition 3.7（Eq. (3.10)・(3.11)）：LP-FT の根拠**
- 特徴が完全（B₀ = U B*）なら，ランダムな頭から始める FT は全時刻で L_ood > 0，LP の解 v_lp^∞ から始める FT（LP-FT）は L_ood = 0．証明は，LP の解では v と B の両方の勾配が 0 になり，パラメータが動かないことによる（Eq. (A.216)・(A.218)）
- 不完全な特徴での LP-FT の上界は示していない（Section 3.4）

**実験（Section 4，付録 B）**
- 10 のずれ：CIFAR-10→STL，CIFAR-10→CIFAR-10.1，Entity-30，Living-17，DomainNet（sketch→real/painting/clipart），FMoW（北米→アフリカ・欧州），ImageNet→{V2, R, Sketch, A}．模型は ResNet-50（MoCo-v2，CLIP，MoCo-TP）と CLIP ViT-B/16
- LP の実装：「we train an ℓ2-regularized logistic regression classifier on frozen features from the penultimate layer of the pretrained model, selecting the best ℓ2-regularization hyperparameter based on ID validation accuracy」（Section 4.1）．切片の有無は書かれていない．3 種（seed）の平均，90% 信頼区間．ImageNet は 1 回で ℓ2 の探索なし
- 訓練データの大きさは 20K〜100 万超（Section 4.1：「linear probing does not appear to perform better than fine-tuning simply because of a small training set」）．**少数ショットの実験は無い**
- Table 1（ID 正解率 %）：FT 85.1 / LP 82.9 / LP-FT 85.7（平均）．個別：CIFAR-10 97.3/91.8/97.5，Ent-30 93.6/90.6/93.7，Liv-17 97.1/96.5/97.8，DomainNet 84.5/89.4/91.6，FMoW 56.5/49.1/51.8，ImageNet 81.7/79.7/81.7
- Table 2（OOD 正解率 %）：FT 59.3 / LP 66.2 / LP-FT 68.9（平均）．個別：STL 82.4/85.1/90.7，CIFAR-10.1 92.3/82.7/93.5，Ent-30 60.7/63.2/62.3，Liv-17 77.8/82.2/82.6，DomainNet 55.5/79.7/80.7，FMoW 32.0/36.6/36.8，ImNetV2 71.5/69.7/71.6，ImNet-R 52.4/70.6/72.9，ImNet-Sk 40.5/46.4/48.4，ImNet-A 27.8/45.7/49.1．LP が FT に勝つのは 10 中 8（負けは CIFAR-10.1 と ImageNetV2，どちらも ID と近い）
- Table 3（DomainNet の OOD 内訳）：FT 55.29/50.26/60.93，LP 87.16/74.50/77.29，LP-FT 86.82/75.91/79.48（real/painting/clipart）
- Table 4（OOD で早期終了・選択した場合）：LP 67.1 vs FT 61.3（平均）．早期終了では歪みを直せない
- 特徴の移動量（Living-17，Section 4.3）：FT で ID 0.0188 ± 0.0001，OOD 0.0167 ± 0.0001．LP-FT では ID 0.0011，OOD 0.0009（20 倍小さい）．Table 5・6 で 10 例中 9 例が「ID の方が大きく動く」（例外は FMoW の FT）
- 条件の確認：特徴が悪い MoCo-v1（ImageNet で 10% 低い）では FT（ID 96%，OOD 71%）が LP（92%，66%）に勝つ．ID≈OOD の CIFAR-10.1 でも FT が LP に勝つ
- Table 9（Living-17，手の比較）：Scratch 92.4/58.2，LP 96.5/82.2，FT 97.1/77.7，FT(頭の学習率 10 倍) 97.2/80.4，FT(事前学習の重みへの L2 罰則 = L2-SP，λ ∈ {1, 0.1, 0.01, 0.001, 0.0001}) 97.1/80.0，Side-tuning 95.5/81.0，LP-FT 97.8/82.6（ID/OOD %）
- Table 7・8（Living-17，CLIP ResNet-50 と CLIP ViT-B/16）：LP 94.7/78.6，FT 94.7/67.3，LP-FT 95.6/77.0；LP 97.5/87.6，FT 97.8/81.5，LP-FT 98.0/86.1

### 4. 軸 A1〜A11 との対応

| 軸 | 判定 | 根拠 |
|---|---|---|
| A1 学習器 | ○ | LP は凍結した特徴の上の線形の頭．理論は罰則なしの最小二乗（勾配流），実験は ℓ2 罰則つきロジスティック回帰（Section 4.1） |
| A2 誤差の分解と K の式 | × | 切片が無いので切片と向きに分けない．雑音なしの回帰なので標本数で減る誤差の式は無い．LP の誤差は特徴の誤差 ε に比例する（Lemma A.14）．次元 k は FT の下界の 1/√k（Eq. (3.7)）と閾値（Eq. (A.170)）に入るが，標本複雑度の項としてではない．実効次元の概念は無い |
| A3 ソースの価値のラベル数換算 | × | 事前学習の価値は ε = d(B₀, B*) で測り，ラベル数には換算しない．クラスあたりの標本数が出るのは Kornblith et al. 2019 の引用「fine-tuning does better on all 11 of their datasets once we have more than just 30 examples per class」（Section 3.2.2）だけで，本論文の理論ではない．余弦² と標本数の関係は無い |
| A4 合併での切片 | × | 切片が無い．ソースのラベル付きデータとターゲットの合併は扱わない（ソースは事前学習 B₀ を通してだけ入る．Section 2：「unlabeled or weakly supervised x inputs」） |
| A5 固定の少数ショット | × | n ≥ m（例 5K），実験は 20K〜100 万超．少数ラベルは「If the training dataset is very small, then even the support of the ID distribution P_id may not be spanned by the training data, and the ID error can be large」と述べるだけ（Section 3.2.2） |
| A6 ドメインのずれ | ○（型は限定） | ずれが主題．型は「OOD が訓練データの張らない方向を含む」部分空間のずれ（Σ が可逆，Assumption 3.4）．平均のずれ・共分散・ラベル比率のずれ・w* の変化は扱わない |
| A7 マルチモーダル | × | 画像分類のみ．CLIP は画像と言葉で事前学習した模型だが，下流は画像だけ |
| A8 弱いラベル・疑似ラベル | × | 扱わない．Section 5 で「prior work self-trains on large amounts of unlabeled data」と対比し，LP-FT は「uses no extra unlabeled data」と述べる |
| A9 出発点への L2 罰則 | △ | 付録 B.4 で特徴抽出器の重みを事前学習の重みに引き寄せる L2 罰則（L2-SP，Li et al. 2018）を Living-17 で比較（OOD 80.0 vs LP-FT 82.6）．頭への罰則（仮説転移）ではなく，理論も無い（脚注 10：「a more extensive theoretical and empirical study on this is important」） |
| A10 実データの検証 | ○ | 10 のずれ，6 の ID データ集合，3 種の事前学習，3 seed の平均と 90% 信頼区間（Table 1〜9） |
| A11 予言と実測の照合 | △ | 定性的な予言（ID の特徴が OOD より大きく動く，LP-FT は 20〜30 倍動かない，早期終了で直らない，特徴が悪いと順序が逆転する）を実験で確かめた（Section 4.3，付録 B.2・B.3）．式が予言する数値と実測の数値を突き合わせる解析は無い |

### 5. 穴

**扱っていないこと（本記録の整理）**
- 切片（バイアス）．模型に無く，ガウスの場合はデータが平均 0
- 雑音とラベル数による統計誤差．雑音なしなので，n はほぼ条件としてだけ入る
- 少数ショット（クラスあたり数本）
- 分類の理論（理論は回帰のみ）
- ソースのラベル付きデータの合併．ソースは B₀ を作る事前学習データとしてだけ入る
- 平均のずれ・ラベル比率のずれ．ID と OOD で w* は共通
- 不完全な特徴での LP-FT の上界．Section 3.4：「The case where we do not have perfect features (d(B₀, B*) > 0) is challenging to analyze because except in very special cases, there is no closed form for the fine-tuning iterates」「we were not able to show an upper bound」

**限界として自分で書いていること（原文）**
- Section 3.2.2：「Ideally we would like the lower bound to increase if we have worse features (so "+ε" instead of "−ε" in the bound)—the reason we do not is that the errors of the pretrained feature extractor d(B₀, B*) and the fine-tuning step can potentially cancel out.」
- Section 3.2.2（Conjectures）：「We believe it may be possible to improve cos θ_max(R₀, S^⊥) to the cosine of the minimum principal angle.」「we show that it would have errors in some directions」（すべての方向ではない）
- 脚注 3：「We compare fine-tuning with linear probing by adapting earlier work on linear probing which requires well-specification.」
- Section 1：「Note that LP-FT is not meant to be a SOTA method but a simple, principled way to get good ID and OOD accuracy」
- 付録 B.1（事前学習の質の条件）：「our results say that the pretraining has to be good (e.g., at least get reasonable accuracy ID) for linear probing to outperform fine-tuning OOD.」

**今後の課題（原文）**
- Section 6：「There are several open questions and extensions such as dealing with non-linear activations, different layerwise learning rates, and the effect of explicit regularization.」
- 脚注 10：「We found that LP-FT outperforms explicit regularization and using a higher learning rate for the linear layer on Living-17 (Appendix B.4), but a more extensive theoretical and empirical study on this is important.」
- 脚注 4：「Extending our results to the NTK regime could be an interesting future direction.」
- 脚注 7：「Intuitively this cancelation is very "unlikely" to happen, and we hope future work can capture this intuition.」
- Section 6：「LP-FT is just a first step in leveraging the intuition from our theoretical analysis and we hope that this work inspires new methods of leveraging powerful pretrained models.」

### 6. 本研究との差の書き方

**結論**：本研究の主張 A・B の式は，本論文のどの式も書き直したものではない．重なるのは「凍結した特徴の上の線形の頭」と「分布のずれ」の 2 点だけで，誤差の源が異なる．本論文の LP の誤差（Lemma A.14）は特徴の誤差 ε に比例する決定論的な項で，本研究の 1/(2K)・q/K はラベル数で減る統計的な項である．本研究の設定に本論文の模型を当てはめると，ε の項は K を増やしても消えない床（偏り），K の項は分散に相当する（本記録の推論．本論文は雑音を置かないので両者を同時には扱っていない）．本論文の ϕ は頭の「初期値」の向きのずれで，本研究の向きの誤差 q/K は「学んだ頭」の向きのずれである．本論文の ε（特徴の部分空間の距離）と本研究の e（ソースの向きの余弦）はどちらも「受け継いだものの良さ」を表すが，測る対象（特徴か重みの向きか）が違う（本記録の推論）．

**関連研究に書く案**
- 案 1（凍結したエンコーダを使う根拠として）：「Kumar ら [arXiv:2202.10054] は，2 層の線形網の回帰模型で，事前学習した特徴が良く分布のずれが大きいとき，全層の微調整が訓練データの張らない方向の特徴を歪め，凍結した特徴の上の線形の頭より OOD 誤差が大きくなることを示した（Theorem 3.3・3.5）．10 のずれの実験で微調整は線形プローブより OOD 正解率が平均 7 ポイント低い．本研究が凍結したエンコーダの上の線形の頭を測定の対象に選ぶ根拠はこの結果である．」
- 案 2（差として）：「同論文の模型は雑音のない回帰で，標本数は特徴の張る空間の次元以上（n ≥ m）であり，切片を持たず，線形の頭の OOD 誤差は特徴の誤差 ε に比例する上界で与えられる（Lemma A.14）．本研究は，クラスあたり K 本のラベルで線形の頭を学ぶときの誤差を切片（1/(2K)）と向き（q/K）に分け，ソースの価値をターゲットのラベルの本数 n = q e²/(1 − e²) に換算する点で，同論文が扱わない少数ラベルの体制を対象とする．」
- 補足（本記録の推論）：本論文の Table 9 の L2-SP（事前学習の重みへの L2 罰則）は特徴抽出器への罰則で，本研究の仮説転移（ソースの頭への L2 罰則）とは罰則の対象が異なる．引用するなら「微調整の歪みを抑える手として L2-SP と LP-FT を比べた」程度に留める．E63（微調整との比較）の報告で，微調整が凍結した流儀に負ける方向は本論文の主張と同じ向きであると述べられる（数値の対応づけは E63 の結果ファイルを示してから）．

### 7. 注意

- **要確認の書誌**：掲載先 ICLR 2022 は本文に無い（読んだのは arXiv v1，2022-02-21）．arXiv の版の更新（v2 以降）の有無も未確認
- **式の復元**：Eq. (3.7)/(A.14) の分数の配置，(A.132)・(A.133)・(A.170) の形は pdftotext の崩れを前後の証明（Eq. (A.145)〜(A.160)，(A.172)〜(A.182)）から復元した．(A.133) の因子を log(n/δ)（平方根なし）と読んだ根拠は (A.157)・(A.158) の比が O(log(n/δ)) になること
- **原本の誤植の疑い**：Eq. (A.171) の右辺が LP の記号になっている（FT のはず）
- **本記録の推論と明示した箇所**：「n は条件としてだけ入る」（2・3 節），「log(n/δ) は上界の技術的な因子」（3 節），「ε は床，K の項は分散に相当」「ϕ・ε と本研究の q/K・e の対応」（6 節），A2・A5 の判定の理由づけ
- **実験の LP の切片**：ℓ2 罰則つきロジスティック回帰に切片があるかは本文に無い．要確認（コードを見る必要がある）
- **未読の節**：A.1，A.2 の証明本体（Lemma A.4〜A.7），A.3.1〜A.3.2 の証明，A.5，A.6 の補題の証明．主定理の仮定と結論の確認には影響しない範囲と判断した
- Section 3.2.2 の「30 examples per class」は Kornblith et al. 2019 の引用であり，本論文の実験結果ではない．本研究の K との対応づけに使うなら，引用元を直接確かめる必要がある

---

## 6. Ha & Chen 2025/2026（arXiv 2507.14661v2）

### 1. 書誌と読んだ範囲
- Ha, W.（KAIST，Department of Mathematical Sciences），Chen, Y.（ETH Zürich，Department of Mathematics），"When few labeled target data suffice: a theory of semi-supervised domain adaptation via fine-tuning from multiple adaptive starts," arXiv:2507.14661v2 [stat.ML]，2026-06-30（本文の注記で確認）[arXiv](https://arxiv.org/abs/2507.14661)．掲載先は本文に無い（**要確認**）．全 93 頁（本文 26 頁＋付録 A〜J）．
- 読んだ範囲（◎）：Abstract，§1（1.1 貢献，1.2 関連研究），§2（2.1 設定，Assumption 1，2.2 UDA の基準手法 OLS-Src・DIP・CIP），§3（Proposition 3.1，3.2 の加法的平均ずれの例，式 (11) の一般の微調整の定式化），§4（Assumption 2〜5，Theorem 4.1〜4.6，4.1 の式 (12)〜(17)，4.2 の式 (19)〜(20)，4.3 の式 (22)〜(30)），§5（MASFT，Corollary 1），§6（Table 2〜4），§7，付録の目次と A.1 の標本数の記載．飛ばした：付録 B〜J の証明と詳細（A.2〜A.4 の実験の詳細，J の DIP の失敗例を含む）．

### 2. 設定
- 模型：反因果の線形構造因果模型（Assumption 1）：X = B X + b Y + ε_X，Y = ε_Y（ラベル Y が共変量 X を生む）．解くと X = (I − B)⁻¹ b Y + (I − B)⁻¹ ε_X．ドメイン m ごとに (B^(m), b^(m), 雑音の分布) が変わる．
- 予測器は線形 fβ(x) = βᵀx，損失は二乗誤差．**切片は無い**（Assumption 2 で雑音は平均 0．共変量は中心化されている）．分類は理論で扱わない："Extending the finite-sample analysis to logistic, cross-entropy, or other losses is left for future work"（§2.1）．
- 微調整の一般形（式 (11)）：UDA の推定量 θ_UDA を出発点に，θ − θ_UDA ∈ Θ_lo（低次元の部分空間か小さい集合）の制約でターゲットのラベル付き標本の損失を最小化．
- 標本数の体制：**有限で少数**．ターゲットのラベル付き標本 n(0) ≥ r log(1/δ)（r は介入の次元）．ソースとラベル無しの標本は大きい（n(1), n_u ≳ d）．比例の極限ではない．
- ずれの型（3 種）：CA（confounded additive shift，Assumption 3）：雑音 (ε_X, ε_Y) に r_ca 次元の共通の隠れ因子 Z が加わる（平均 0 なので共分散のずれ．§3.2 の決定論的な平均のずれの例はその非中心化版）．SC（sparse connectivity，Assumption 4）：B の r_sc 列が変わる（回帰係数の疎な変化）．AW（anticausal weight，Assumption 5）：b が変わる（ラベルの現れ方の変化．複数ソース）．ラベルの周辺分布は反因果なので不変．

### 3. 示したこと
記号の対応：論文の d ↔ 本研究の p．論文の n(0) ↔ 本研究の K × クラス数（回帰なのでクラスは無い）．論文の r（介入の次元）↔ 本研究で「ターゲット固有に直す部分の次元」．切片だけ直す本研究の場合は r = 1 に相当し，向きも直す場合は q に相当する（本記録の対応づけ）．e・o・d に当たる量は無い．
- Proposition 3.1：ターゲットだけの最小最大の超過リスクは c ν_Y² min{d/n(0), 1}（ラベル無しのターゲット標本は役に立たない）．
- §3.2 の例（Figure 1）：3 次元で共変量の平均が (1,1,1) → (−1,0,1) にずれるとき，DIP（平均を合わせる射影）の解と真の解の差 β* − β_DIP は v_DIP = Σ_X^(0)⁻¹(E X(1) − E X(0)) に平行で，この 1 方向はラベル無しのターゲット標本から分かる．"the unlabeled target sample reveals the direction in which fine-tuning should occur"．**平均のずれは，切片を持たない回帰では「1 方向の向きの修正」として現れる**（本記録の読み）．
- Theorem 4.1（FT-DIP，CA）：n(0) ≥ r_ca log(1/δ) などの条件で δR(0) ≤ c η r_ca log(1/δ)/n(0) + C[d log(1/δ)/n(1) + d_δ/n_u(1) + d_δ/n_u(0)]．Theorem 4.2：下界 c σ_Y² r_ca/n(0)．"the labeled target sample complexity scaling with the intervention dimension rather than the ambient dimension"．
- Theorem 4.3／4.4（FT-OLS-Src，SC）：ℓ1 制約つきの微調整で r_sc log d/n(0)，下界 r_sc/n(0)．Theorem 4.5／4.6（FT-CIP，AW）：r_aw/n(0)（ラベル無しのターゲット標本無しで達成．付録に無い場合は r_aw + 1 方向）．
- Corollary 1（MASFT）：候補 L = 2M + 1 個から検証集合で選ぶ損は √(log(M/δ)/n_val)．
- 実験（§6，Table 2）：人工データ d = 100，n(0) = 100，n_val = 100，20 回．CA r = 5 で FT-DIP 0.28 ± 0.13，OLS-Src 53.6，OLS-Pool 145，FT-OLS-Src-L2 1.07，OLS-Tar > 1000，OLS-Tar-MoreData 0.17．SC r = 5 で FT-OLS-Src-L1 0.075，L2 0.43．AW r = 4 で FT-CIP 0.15，FT-CIP-Tar 0.13．**OLS-Tar-MoreData は「ターゲットだけで同じ率に届く標本数」d/n_new = r/n(0) で作る**（ソースの価値をターゲットの本数に換算する操作に当たる．本記録の読み）．Light Tunnel（実機，n(0) = 40，Table 3）：lt CA で FT-DIP 0.043 vs OLS-Tar 0.054 vs MoreData（4000 本）0.041．NIR（corn n(0) = 24，mango n(0) = 60，Table 4）：FT-DIP・FT-CIP・MASFT が UDA と OLS-Tar に勝ち MoreData に迫る．
- ℓ2 の微調整（FT-OLS-Src-L2，ソースの OLS を出発点に ‖β − β_LS‖² を制約）は実験の基準手法としてだけ出る（理論無し）．CA で FT-DIP の 4 倍，SC で L1 の 6 倍の超過リスク．

### 4. 軸との対応
| 軸 | 判定 | 根拠 |
|---|---|---|
| A1 | △ | 線形予測器・二乗損失・生の共変量．凍結した深層特徴ではないが，"fine-tuning in large neural networks ... often happens in a small subspace" を動機に挙げる |
| A2 | △ | 誤差 ∝ r/n(0)（直す部分の次元／ターゲットのラベル数）．切片と向きには分けない（切片無し）．実効次元 q に当たるのは介入の次元 r |
| A3 | △ | OLS-Tar-MoreData（d/n_new = r/n(0)）で「同じ率のターゲットの本数」を作る．余弦や閉じた換算式は無い |
| A4 | △ | DIP の平均合わせが「平均のずれの 1 方向」を切り出す（§3.2）．合併（OLS-Pool）は基準手法で，切片の直る割合は扱わない |
| A5 | ○ | n(0) ≥ r log(1/δ) の有限標本．実験は n(0) = 24〜100（回帰なのでクラスあたりではない） |
| A6 | ○ | 平均（§3.2），共分散（CA），回帰係数（SC），ラベルの現れ方（AW）．ラベル比率は不変 |
| A7 | × | — |
| A8 | × | — |
| A9 | ○ | 出発点（UDA の解）への制約つきの微調整が主題．ℓ2 版は実験のみ |
| A10 | ○ | Light Tunnel（実機）と NIR（corn，mango）．小規模 |
| A11 | △ | 理論の率に合わせた MoreData との比較．予言値と実測値の曲線の照合は無い |

### 5. 穴
- 扱っていないこと：分類・ロジスティック損失，切片，凍結した深層特徴，複数モダリティ，ソースの価値の閉じた換算，弱いラベル．
- 限界として書くこと："Extending the finite-sample analysis to logistic, cross-entropy, or other losses is left for future work."（§2.1）；"Extensions to mixed causal-anticausal SCMs are possible, but the anticausal model suffices"（§2.1）．
- 今後の課題："extending our analysis beyond anticausal linear SCMs to mixed causal-anticausal or nonlinear SCMs is a promising direction ... Integrating ideas from causal representation learning to recover latent structure and adapt in representation space remains an open challenge."（§7）．

### 6. 本研究との差の書き方
「Ha and Chen（2026）は，反因果の線形構造因果模型でソース・ラベル無しターゲット・少数のラベル付きターゲットを使う半教師ありドメイン適応を定式化し，ずれが低次元の介入で生じるとき，UDA の解を出発点に介入の次元 r の部分空間だけを微調整すれば，ターゲットのラベル数 n(0) に対する超過リスクが d/n(0) から r/n(0) に下がり最小最大最適であることを示した．本研究の『少数ラベルが主に直すのは切片』は，この枠で r = 1（切片）と r = q（向き）に分けた場合に当たるが，本研究は二乗損失の回帰ではなく凍結特徴の上のロジスティック回帰を扱い，直す部分を切片と向きに分けて K の関数として測る点，ソースの価値をラベル数に換算する点が異なる．」
式の対応：本研究の 1/(2K)・q/K は Ha & Chen の r/n(0) の r を 1 と q に置いたものと同じ形（本記録の対応づけ．論文に切片は無い）．

### 7. 注意
- 「平均のずれ = 1 方向の修正」「OLS-Tar-MoreData = 換算」は本記録の読み．
- 付録 A.2〜A.4（実験の詳細），証明は未読．Theorem 4.1 の定数 η・C の形は pdftotext で崩れており本文の形で記した．掲載先は要確認．

---

## 7. De Silva, Ramesh, Priebe, Chaudhari, Vogelstein 2023（ICML 2023，PMLR 202）

確かめた水準：◎ 本文（下請けの精読記録．式 (7)・付録 A.3 の存在を grep で確認）．[PMLR](https://proceedings.mlr.press/v202/de-silva23a.html)

### 1. 書誌と読んだ範囲

- 題名：**The Value of Out-of-Distribution Data**（本文冒頭で確認．前回の記憶どおり）
- 著者：Ashwin De Silva*（Johns Hopkins Univ.），Rahul Ramesh*（Univ. of Pennsylvania），Carey E. Priebe（JHU），Pratik Chaudhari†（UPenn），Joshua T. Vogelstein†（JHU）．脚注に「* † Equal contribution」
- 掲載先：脚注「Proceedings of the 40th International Conference on Machine Learning, Honolulu, Hawaii, USA. PMLR 202, 2023.」．**頁番号 7366–7389 は本文に無い（要確認）**
- コード：https://github.com/neurodata/value-of-ood-data（§1）
- 確かめた水準：◎ 本文（pdftotext 出力 1929 行，24 頁ぶん）
- 読んだ範囲：全 24 頁．§1（1.1 Outlook），§2（2.1 FLD の例，Theorem 1，Remark 2，2.2 ニューラルネットの実験），§3（Theorem 3，3.1 α* の選び方，3.2 α 重みつき学習，3.3 上界は非単調を予言するか），§4 関連研究，§5 謝辞，付録 A.1〜A.8（式 (3)〜(14)，Theorem 4），付録 B.1〜B.6（Table A1，図 A1〜A9）．図 1〜12・A1〜A9 の数値は軸の目盛と図の説明文から拾った（図の中の数値は pdftotext で崩れている）
- 本文に Theorem 2 と Remark 1 は無い．番号は Theorem 1 → Remark 2 → Theorem 3 → Theorem 4 と続くので，定理と注意が番号を共有していると読める（本記録の推論）

### 2. 設定

- **模型**（付録 A.1）：1 次元・2 クラス．ターゲットのクラス条件付き分布は N(−μ, σ²) と N(+μ, σ²)．OOD（out-of-distribution．ターゲットと異なる分布のデータ．本研究のソースに相当）はターゲットを Δ だけ平行移動したもので N(Δ−μ, σ²) と N(Δ+μ, σ²)．分散は両分布で同じ．クラスの事前確率は両分布とも 1/2
- **分類器**（付録 A.2）：Fisher の線形判別（FLD．各クラスを同じ共分散の正規分布と仮定し，平均と共分散を最尤推定して事後確率の大きいクラスに分類する．LDA と同じ）．多次元では向き ω = Σ̂⁻¹(μ̂₁−μ̂₀) と閾値 c = ω⊤(μ̂₀+μ̂₁)/2 + log(π̂₀/π̂₁) に分かれて定義される（§2.1：「ω is a projection vector which acts as a feature extractor and c is a threshold that performs one-dimensional discrimination」）．解析は 1 次元・等事前確率に限るので閾値 ĉ = (μ̂₀+μ̂₁)/2 だけになる（式 (5)）．**罰則は無い**
- **標本数の体制**：有限．ターゲット n 本，OOD m 本．各クラスちょうど半分（n_k = n/2，m_k = m/2 と仮定）．学習器が OOD の有無を知らず合併して学ぶ場合（OOD-agnostic，式 (1) の平均損失）と，知っていて α で重みづける場合（OOD-aware，Theorem 3）の 2 つ
- **ずれの型**：平均の平行移動 Δ のみ（両クラスが同じ向きに同じ量だけずれる）．共分散のずれとラベル比率のずれは無い．実験では回転・ぼかし（幾何的なずれ），別のクラス対（Split-CIFAR10），別のドメイン（PACS，DomainNet，CINIC-10）

### 3. 示したこと

記号の対応：本研究の o = ĉ/σ（最適な閾値 0 からのずれ），d = 2μ/σ（クラス間のマハラノビス距離），K = n/2（2 クラスで n_k = n/2 なので，クラスあたりの本数が K）．論文の本文は σ² = 1 で書き，σ ≠ 1 なら μ → μ/σ，Δ → Δ/σ と置き換えるとしている（付録 A.3 末尾）．

**(a) 合併した閾値の分布**（付録 A.3）．ĉ ~ N(μ_h, σ_h²)，μ_h = mΔ/(n+m)，σ_h² = 1/(n+m)．本研究の記号では
- ターゲットのみ（m = 0）：E[o] = 0，**Var[o] = 1/n = 1/(2K)**
- 合併：**E[o] = (m/(n+m))·(Δ/σ)，Var[o] = 1/(n+m) = 1/(2K+m)**．閾値は OOD の側へ Δ の m/(n+m) 倍だけ引かれ，分散は 1/(2K) から 1/(2K+m) に減る

**(b) 閾値を与えたときのターゲット誤差**（式 (7)）．p(誤り | ĉ) = ½[1 + Φ(ĉ−μ) − Φ(ĉ+μ)]．書き直すと 誤差 = ½[Φ(−(d/2+o)) + Φ(−(d/2−o))]，**正解率 = ½[Φ(d/2+o) + Φ(d/2−o)]**．本研究の主張 A の式と一致する（本記録の照合．論文は誤差の形で書く）．

**(c) 期待誤差**（式 (2) ＝ 式 (8)）．
e_t(ĥ) = ½[Φ((mΔ − (n+m)μ)/√((n+m)(n+m+1))) + Φ((−mΔ − (n+m)μ)/√((n+m)(n+m+1)))]．
(b) を ĉ の正規分布で平均したもので，恒等式 ∫Φ(cx+d)φ(x)dx = Φ(d/√(1+c²)) を使う．本研究の記号では e_t = ½[Φ(−(d/2 − ō)/√(1+v)) + Φ(−(d/2 + ō)/√(1+v))]，ō = E[o]，v = Var[o]（本記録の書き直し）．

**(d) 非単調性**．Theorem 1 は「ターゲット分布と OOD 分布の組で，合併して学ぶと誤差が OOD の本数に対して非単調になるものが存在する」という存在定理で，証明は (c) の例．**非単調になる条件を一般の形で述べた定理・命題は無い**．Remark 2 は，Bayes 誤差に達するのに要るターゲットの本数を N として「n < N のときに起きやすい」と述べる（直観で，証明は無い）．ずれ Δ が「the threshold beyond which the error is monotonic」を決めるとも述べる．図 2 は ĉ の MSE を分散と偏り² に分けて示し，少量では分散の減少が勝ち，多量では偏りが勝つと説明する．
- 図 1（n = 100，μ = 5，σ = 10，Δ ∈ {0, 0.8, 1, 1.2, 1.4, 1.6, 1.8}）：誤差は 0.3088〜0.3100 の範囲．Δ = 1.6 では m = 28 で最小
- 図 3（μ = 6，σ = 16，n = 500）：Bayes 誤差が大きい設定では n = 500 でも非単調（0.3539〜0.3542）
- 図 A2：式 (8)・(11) とモンテカルロが一致
- 本記録の復元（論文には書かれていない）：MSE = σ²/(n+m) + (mΔ/(n+m))² を m で最小化すると，Δ/σ > 1/√(2n) のとき m* = n/(2n(Δ/σ)² − 1)，K で書くと m*/n = 1/(4K(Δ/σ)² − 1)．Δ/σ ≤ 1/√(2n) なら MSE は m について単調減少．m → ∞ で MSE → Δ² なので，大量の OOD がターゲットのみより損になるのは Δ/σ > 1/√n = 1/√(2K) のとき．m = 0 での微係数は −σ²/n² < 0 なので，MSE の意味では少量の OOD は Δ によらず常に得．図 1 の設定で m* ≈ 24（論文の式 (2) の最小は 28．誤差は MSE の単調関数ではないので少しずれる）

**(e) α 重みつき FLD**（付録 A.4〜A.5，式 (9)〜(11)）．μ̂_k = (nα x̄_{t,k} + m(1−α) x̄_{o,k})/(nα + m(1−α))．ĉ_α ~ N(m(1−α)Δ/(nα+m(1−α)), (α²n + (1−α)²m)/(αn + (1−α)m)²)．誤差は式 (11)（(c) と同じ形）．α* は数値で求める．図 9（μ = 1，σ = 1，n = 4，誤差 0.165〜0.185）：α* を使えば誤差は m に対して単調に減る．α* は m とともに 1 に近づき，Δ = 0 なら α* = 1/2．

**(f) Theorem 3**（Ben-David ら 2010 の Theorem 3 の言い換え）と α*（§3.1）．e_t(ĥ_α) ≤ e_t(h*_t) + 4√((α²/n + (1−α)²/m)(V_H − log δ)) + 2(1−α) d_H(P_t, P_o)（本文の形．付録の Theorem 4・式 (12) には λ の項と log(2(n+m+1)) が付く）．ρ = √(V_H − log δ)/d_H とおくと α* = 1（n ≥ 4ρ² のとき），それ以外は α* = (n/(n+m))·(1 + m/√(4ρ²(n+m) − nm))．後者は pdftotext で崩れていたので，上界を α で微分して本記録が復元した（境界 n = 4ρ² で 1 に連続する．復元の自信は高い）．

**(g) 上界の検証**（§3.3，付録 A.7〜A.8，式 (13)〜(14)）．1 次元閾値の VC 次元を d = 2，α = n/(n+m)，λ = Φ((−Δ/2−μ)/σ) + Φ((Δ/2−μ)/σ)，d*_H(Δ) は数値で計算．図 12（μ = 5，σ = 10，n = 100）：上界は 2.35〜2.5（1 を超えて無意味）で m に対して単調増加，真の誤差は 0.3093〜0.3097 で非単調．図 A3：Δ が大きいと形は似る．

**(h) 実験**（§2.2，付録 B．Table A1）．ネットワークは SmallConv（0.12M），WRN-10-2，WRN-16-4 をゼロから学習（SGD Nesterov，cosine，lr 0.01，weight decay 1e-5）．ターゲット n を固定し OOD m を増やす．10 seeds，95% 信頼区間．
- 図 4 左：Rotated CIFAR-10（T2 = 鳥 vs 猫，n = 100，回転 0〜135°，WRN-10-2）．大きい角度で非単調．「For 60° and 135° in particular, the generalization error at m/n = 20 is worse than the generalization error with a fewer OOD samples」．誤差 0.24〜0.40
- 図 4 中：Blurred CIFAR-10（T4 = 蛙 vs 馬）．強いぼかしで非単調，弱いぼかしでは単調減少
- 図 4 右・A6・A7：Split-CIFAR10 の 20 対すべて（2 ネットワーク）．多くの対で非単調
- 図 5：Rotated MNIST（10 クラス，10 本/クラス，θ = 15〜45°）大きい θ で非単調．PACS（犬・象・馬，10 本/クラス，n = 30，WRN-16-4）sketch を OOD にすると非単調．DomainNet（鳥・飛行機，25 本/クラス，n = 50）も同様
- 図 6：CINIC-10（CIFAR-10 がターゲット，ImageNet が OOD，10 クラス，n = 100）．ずれはあるが単調減少（誤差 0.60〜0.78）．非単調は常には起きない
- 図 7（T2 → T5，WRN-10-2，n = 100）：データ拡張，事前学習（CINIC-10 の ImageNet 鳥・猫 14000 枚で 100 epoch → 線形プロービング 50 epoch + 全層微調整 50 epoch，lr 0.001），ハイパーパラメータ最適化（Ray，ターゲットのみの検証集合）のいずれも，OOD が増えるにつれての悪化を止めない
- 図 8（n，m ともクラスあたり）：正規模型 n = 25〜500（解析式），MNIST θ = 30° n = 5〜100，DomainNet 40 クラス n = 10〜100．**n が小さいときだけ非単調で，n が大きいと消える**．「When we have a large number of target samples, the model is closer to the Bayes error and benefits less from more OOD」
- 図 A9：DomainNet 40 クラス（5 本/クラス，n = 200，3 seeds）．art・quickdraw・sketch のうち 2 つで非単調．誤差 0.87〜0.915（一様予測なら 0.975）

**(i) α 重みつき学習**（§3.2，図 10・11・A4）．PACS photo → sketch と Split-CIFAR の 3 対で，OOD の有無を知らない学習（赤）は悪化，α = 1/2（naive）は PACS では改善するが CIFAR では不十分，格子探索の α*（緑）は単調に改善．α* は 0.9〜1.0 で「very close to 1 but it is never exactly 1」．データ拡張 + α = 1/2 でも単調減少（図 11）．ミニバッチ内のターゲット割合 β = 0.5, 0.75 で α* のときに改善（図 A4）．

### 4. 軸 A1〜A11 との対応

| 軸 | 判定 | 根拠 |
|---|---|---|
| A1 学習器 | △ | 理論は 1 次元の FLD（LDA と同じ）で罰則なし，特徴の凍結なし．実験は CNN をゼロから学習（図 7 のみ事前学習 → 線形プロービング → 微調整の 1 変種） |
| A2 切片と向き | △ | 切片（閾値）だけ．Var[o] = 1/(n+m)，ターゲットのみで 1/(2K)，偏り (m/(n+m))·Δ/σ（付録 A.3）．向きの誤差の式は無い（1 次元）．次元 p は入らない（上界に VC 次元 2 が入るのみ） |
| A3 ソースの価値の換算 | × | OOD の価値をターゲットの本数に換算しない．m/n の比と α* で語る．余弦は出ない |
| A4 合併の切片 | △ | 合併で閾値が OOD の側へ Δ の m/(n+m) 倍だけ引かれる式がある（付録 A.3）．ドメイン別の切片や中心化は無い |
| A5 少数ショット | ○ | 有限標本．解析は n = 4（2 本/クラス）〜500，実験は 5〜50 本/クラス．比例の極限は使わず，Remark 2 で漸近的な理論は捉えないと述べる |
| A6 ずれの型 | ○（平均のずれのみ） | 理論は平均の平行移動 Δ だけ．共分散・ラベル比率のずれは無い．実験のずれは回転・ぼかし・クラス対・ドメイン |
| A7 マルチモーダル | × | 画像のみ |
| A8 弱いラベル | × | 弱いラベル・疑似ラベルは無い．OOD 検出は関連研究で触れるだけ（「We neither propose nor use an explicit method to do this in our paper」） |
| A9 出発点への罰則 | × | リッジ・仮説転移は無い．weight decay 1e-5 は実装の既定値 |
| A10 実データ | ○ | 5 データ集合（CIFAR-10，CINIC-10，Rotated MNIST，PACS，DomainNet），3 ネットワーク，10 seeds．ただし FLD の式は実データで検証していない |
| A11 予言と実測 | △ | 解析式とモンテカルロの一致（図 A2）のみ．実データとの照合は非単調の有無の定性的な一致．Ben-David の上界は真の誤差に対して無意味な大きさ（図 12） |

### 5. 穴

**扱っていないこと**（本記録の整理）：向き（重みベクトル）の誤差と次元依存．多次元での解析（Δ が判別の向きに直交する成分を持つ場合は閾値が動かないはずだが，論じていない）．凍結特徴の上の線形の頭．罰則つき学習．共分散のずれ・ラベル比率のずれ．多クラスの理論（理論は 2 クラス）．複数のソース．ソースの価値のターゲット本数への換算．中心化．非単調になる条件の一般形（Remark 2 の直観のみ）．

**限界として自分で書いていること**：
- 「The main contribution of this paper is to highlight the importance of this phenomenon. We leave the development of a practical solution for future work.」（§1.1）
- 「While its practical utility is limited, this does suggest that if we can detect OOD samples, then there may be ways to benefit from them.」（Abstract）
- 「However, this procedure is dependent on two critical ideal conditions: (1) We must know which samples in the dataset are OOD, and (2) We must have a held out dataset of target samples to tune the weight α.」（§3.2）
- 「The upper bound in Figure 12 is vacuous and does not follow a non-monotonic trend when the true error does.」（§3.3）
- 「Although we do not observe this in Figure 8, we believe that Remark 2 that non-monotonicity could theoretically occur even at large target sample sizes, if the number of samples required to attain the Bayes optimal error is high.」（§2.2）

**今後の課題**：
- 「In future work, we hope to formally characterize this non-monotonic trend in generalization error by building new learning-theoretic tools.」（Remark 2）
- 「we hope that it would serve as a proof-of-concept solution that motivates future research into accurately identifying OOD samples within datasets, designing ways of determining the optimal weights, and developing better procedures for exploiting OOD samples to achieve a lower generalization.」（§3.2）
- 「While they do not identify non-monotonic trends in target generalization error, their tools [Bu et al. 2022] can potentially be useful to characterize the phenomenon discovered in our work.」（§4）

### 6. 本研究との差の書き方（関連研究の案）

案 1（式の対応を書く）：
「本研究の正解率の式 ½[Φ(d/2+o) + Φ(d/2−o)] は，De Silva ら (2023) が 1 次元の Fisher 線形判別で導いた閾値つきの誤差（同論文の式 (7)）を o = ĉ/σ，d = 2μ/σ と置いたものである．同論文は，ソースを合併すると閾値が Δ の m/(n+m) 倍だけソース側に引かれ分散が 1/(n+m) に減ること（同 A.3）から，誤差がソースの本数に対して非単調になることを示した．本研究はこの切片の項に向きの誤差 q/K を加え，ソースの価値をターゲットのラベルの本数 n = q e²/(1−e²) に換算し，中心化で切片のずれを消す操作を実データで測る点で異なる．」

案 2（位置づけを書く）：
「De Silva ら (2023) はソースの価値がソースの本数に対して非単調であることを 1 次元の模型と画像分類で示したが，分類器は向きを持たず，ソースの価値をターゲットの本数に換算せず，凍結した特徴の上の罰則つき学習も扱っていない．本研究は同論文の切片の式を向きつきに拡張し，K（クラスあたりの本数）を軸に得と損の境目を実データで照合する．」

式の対応の要点：本研究の 1/(2K) は同論文の σ_h² = 1/(n+m) の m = 0 の場合（n = 2K）である．本研究の「合併で切片がずれる」は同論文の μ_h = mΔ/(n+m) に対応する．同論文には向きの誤差・q・e・p に対応する量が無い．

### 7. 注意

- 頁番号 7366–7389 は本文に無い（要確認）．掲載先「PMLR 202, 2023」とタイトル・著者は本文で確認した
- Theorem 2・Remark 1 が無いのは定理と注意の番号共有と読んだ（本記録の推論）
- §3.1 の α* の式と式 (13)・(14) の定数は pdftotext で崩れていた．α* は本記録が上界の微分から復元し，境界での連続性で確かめた．式 (14) は構造（Bayes 誤差 + O(√(log(n+m)/(n+m))) の複雑さの項 + (m/(n+m))·d*_H + λ）だけを記した
- 3(d) の m* の式と「少量なら常に得」「Δ/σ > 1/√(2K) なら大量で損」は本記録が MSE から導いたもので，論文には無い．論文の図 1 の最小 m = 28 は式 (2) の誤差で求めた値であり，MSE の最小（約 24）とは少しずれる
- 3(b) で正解率の式が本研究の主張 A と一致するとしたのは本記録の照合．本研究の「切片の誤差 1/(2K)」が閾値の分散（σ 単位）を指すとして対応づけた
- 図の数値（誤差の範囲，α* の値）は軸の目盛から読んだので，±1 目盛の誤差がありうる
- 図 8 だけは n・m がクラスあたりの本数で，他の図では総数である（図 8 の説明文に明記）
- 読めなかった節は無い．参考文献は本文の引用に必要な範囲だけ確認した

---

## 8. Hernandez, Kaplan, Henighan, McCandlish 2021（arXiv 2102.01293）

確かめた水準：◎ 本文（下請けの精読記録）．[arXiv](https://arxiv.org/abs/2102.01293)

結論を先に書く．この論文は，事前学習の価値を「ターゲットの学習データの量に換算した値 D_T（effective data transferred）」として実測し，その値がモデルの大きさ N と微調整データ量 D_F の冪乗則 D_T = k (D_F)^α (N)^β に従うことを，変換器の言語モデル（テキスト→Python コード）で経験的に示した論文である．線形の頭・切片・向き・次元・ラベルの本数は一切出てこない．定理は無く，すべて実測の当てはめである．本研究の主張 B（ソースの価値をターゲットのラベルの本数 n に換算する）と「換算の発想」だけが対応し，換算の単位（文字数 vs ラベルの本数），換算の導き方（実測 vs 理論），換算が依存する量（N vs q と e）がすべて異なる．

### 1. 書誌と読んだ範囲

- 著者：Danny Hernandez（連絡先 dannyhernandez@gmail.com），Jared Kaplan（OpenAI で実施，Johns Hopkins University），Tom Henighan，Sam McCandlish．所属 OpenAI
- 題：Scaling Laws for Transfer．arXiv:2102.01293v1 [cs.LG]，2021 年 2 月 2 日（本文 1 頁の arXiv 注記で確認）
- 掲載先：本文に arXiv 以外の掲載先の記載は無い．**要確認**（本記録では arXiv プレプリントとして扱う）
- 読んだ範囲：全 19 頁．Abstract，1 Introduction（1.1 Key Results，1.2 Notation），2 Methods，3 Results（3.1 Ossification，3.2 Compute efficiency），4 Related Work，5 Limitations，6 Discussion（6.1〜6.7），7 Conclusion，Appendix A〜G．式は 1.1，1.2，6.1，B.1，B.2，C.1，E.1 のすべてを読んだ．定理・命題は無い
- 読めなかったもの：図 1〜15 の本体（軸・データ点）は pdftotext で消えており，キャプションと軸ラベルの断片だけを読んだ．表は Table 1 のみで，これは読めた
- 確かめた水準：◎ 本文

### 2. 設定

- 模型：変換器の言語モデル（原文 "transformer language models"．デコーダ型かどうかは本文に明記が無い）．学習は全パラメータの微調整と読める（本記録の推論．凍結・線形の頭・プローブの記述が無い）．損失は "the cross-entropy loss in nats per token"（1.2 節）．教師なし（次トークン予測）であり，分類器もラベルも無い
- 罰則：出発点への罰則は無い．最適化は Adam，バッチ 256，系列長 2048 トークン，warm-up 3000 步，語彙 50257．ハイパーパラメータは [KMH+20]（Kaplan らの言語モデルの冪乗則）のものを流用（2 節）．"trained to convergence to the optimal early stopping point"（2 節）
- 事前学習のデータ：(a) テキストのみ（WebText2，Common Crawl，英語 Wikipedia，Internet Books の混合．合計 24 billion 文字），(b) テキスト 50% と非 Python コード 50%（非 Python コードは 240 billion 文字／340 GB．.c 21%，.java 18%，.js 17%，.cpp 12% など．脚注 3）
- 微調整の課題：Python コードの言語モデル（公開 GitHub から 22 billion 文字／31 GB，3% を評価用に保持．2 節）．ターゲットの評価は held-out の Python テストセットの損失
- モデルの大きさの範囲：本文は "Model size and dataset size each spanned 4 orders of magnitude"（2 節）とだけ書く．本文に出る具体値は 1M（図 5 キャプション），40M（図 1），700M（図 6 の軸ラベル "700M zero-shot"）．最大値の数値は本文に無い．6.2 節で GPT-3 の大きさへ "extrapolated up in model size by two orders of magnitude" と書くので，最大は 10^9 のけたと読める（本記録の推論）．**要確認**
- 微調整データ量の範囲：当てはめは "3 orders of magnitude in fine-tuning dataset size"（図 2 キャプション）．本文に出る具体値は 3e5 文字（図 1），3e8 文字（図 4），5.5e9 文字（図 12），10B 文字（図 7），最小は "our smallest dataset (100,000 tokens)"（Appendix G）
- 標本数の体制：有限で，文字数で数えた連続量．比例の極限ではない．クラスごとの標本数という概念は無い．「少ないデータ（low-data regime）」の定義は "having 10% or less of the amount of data it would take to get to 99% of the performance that infinite data would yield"（脚注 2）．この基準量 D(N) はモデルの大きさごとに図 3 の曲線の交点から推定（Appendix A）し，"for which we had a noisy fit"（図 6 キャプション）と自認している
- ずれの型：分布のずれはテキスト→Python という 1 組（混合事前学習を入れて 2 組）だけ．ずれを平均・共分散・ラベルの比率に分けていない．ずれの大きさは α という 1 つの指数で "directed proximity"（向きつきの近さ）として表す（1.1 節）

### 3. 示したこと

定理は無い．示したことはすべて実測と当てはめである．

**(a) D_T の定義（原文）**
"DT - effective data transferred, the amount of additional python, in characters, that a model trained on python from-scratch of the same size would have needed to achieve the same loss on python as a pre-trained model."（1.2 節 Notation）．全体は D_E = D_T + D_F（Eq. B.1）で，D_E が "total effective data"．

**(b) 測り方**
同じ大きさ N の「Python だけで最初から学習したモデル」の損失曲線（横軸：データ量）を用意し，微調整したモデルが達した損失と同じ損失に，最初から学習したモデルが何文字で到達したかを曲線の補間で読む．原文："Our analysis focuses on units of data while holding performance and model size constant."（1 節）．補間であることは Appendix D "The estimation of effective data through interpolation is poorly conditioned in the high data regime" から確かめた．図 1 の例：40M パラメータのモデルを 3e5 文字で微調整すると "DT is approximately 1000x bigger than DF"（図 1 キャプション）．

**(c) 冪乗則と係数（Eq. 1.1，Table 1）**
D_T = k (D_F)^α (N)^β．N は語彙と位置の埋め込みを除いたパラメータ数．

| 事前学習 → ターゲット | k | α | β |
|---|---|---|---|
| Text ⇒ Python | 1.9e4 | 0.18 | 0.38 |
| 50% Text and 50% non-python code ⇒ Python | 2.1e5 | 0.096 | 0.38 |

当てはめの範囲は "over 4 orders of magnitude in model size and 3 orders of magnitude in fine-tuning dataset size"（図 2），ただし低データ体制（D_F/D(N) < 0.1）に限る．β が 2 つの事前学習で同じ 0.38 だったことから "the exponent beta appears to depend only on the model architecture and target distribution"（1.1 節）と仮説を置き，α を "a useful measure of the directed proximity of two distributions, with smaller α indicating closer proximity" と呼ぶ．k が 11 倍（2.1e5/1.9e4）大きいことを "mixture models transfer more readily than plain text in the low-data regime"（Table 1）と読み，α が小さいことを "the benefit diminishes as we approach the high-data regime" と読む．テキスト→Python では β ≈ 2α なので "a 10x increase in model size, N, would be worth approximately a 100x increase in fine-tuning dataset size, DF"（1.1 節）．

当てはめ方（Appendix C）：D_F ごとに，移された割合 D_T/(D_F + D_T) の logit を N に対して直線で当てはめ（図 10），傾きがどの D_F でもほぼ同じだったので，式 C.1 "fraction effective data from transfer = 1/(1 + (N_*/N)^0.38)" の形で各 D_F の N_* を求め，その N_* を D_F の関数として当てはめて全体の式 1.1 を得た（"fitting the fits"）．式 C.1 の形は pdftotext で崩れており，上の形は Eq. B.2 から逆算して復元した（N_*^β = D_F^{1−α}/k．復元に自信あり）．

**(d) 「事前学習は微調整データを何倍にするか」の読み方：本文にある**
Abstract："We find that pre-training effectively multiplies the fine-tuning dataset size."．本文："An implication of Equation 1.1 is that pre-training effectively multiplies the fine-tuning dataset, DF, in the low-data regime. We find the multiplier formulation helpful in building intuition. Note that the multiplier goes down as DF increases."（1.1 節）．式 1.2：effective data multiplier = (D_F + D_T)/D_F ≈ D_T/D_F = k (N)^β / (D_F)^{1−α}．テキスト→Python なら倍率は D_F^{−0.82} で減る．図 6 左は "Data Multiplier vs Data Regime" で，横軸 D_F/D(N)，"Pre-training helps / Pre-training hurts" の境界を描く．

参考（本記録の計算）：式 1.1 に N = 4e7，D_F = 3e5 を入れると D_T ≈ 1.4e8 文字で D_F の約 470 倍になる．図 1 の "approximately 1000x" は当てはめ値ではなく実測点の値と読める．

**(e) データが少ないときと多いときの違い（ossification）**
- 少ないとき（D_F/D(N) < 0.1）：D_T ≫ D_F（脚注 4），式 1.1 が良く当てはまる．モデルを大きくすると，最初から学習したモデルは "hit a wall"（データで頭打ち）になるが，微調整したモデルは傾きが減るだけで改善を続ける（図 3，"only a change in slope when they are data-limited"）
- 中くらい以上（D_F/D(N) > 0.1）："the fit given by equation 3.2 breaks down"（Appendix D，図 13．この "3.2" は本文に無い式番号で，Eq. 1.1 か B.2 を指すと読める．7 節参照）
- 多いとき（D_F/D(N) → 1 以上）："When D/D(N) approaches 1.0 we observe that pre-training reduces our effective data, and our small fine-tuned models were unable to reach trained form scratch performance even as they were trained on 10x or 100x D(N)."（3.1 節）．図 5："we get better performance training the 1M parameter models (purple) with our larger size datasets (>1e8) from-scratch"．これを ossification（骨化）と呼び，"one could think of the pre-training as a particularly bad initialization that the model has trouble recovering from"（3.1 節）と述べる．調整で回復できるか，大きいモデルでも起きるかは "we didn't investigate that question" "It's entirely possible that large models would have different behavior in this regime"（3.1 節）

**(f) その他の数値**
- 計算効率（3.2 節，図 4・7・8）：少ないデータでは微調整の方が同じ計算量で損失が低く，微調整の学習曲線は "lie tangent to the frontier for most of training"．多いデータでは収束後の性能が最初からの学習より悪くなる
- ゼロショット・少数ショットの外挿（6.2 節，"relatively speculative"）：GPT-3 の大きさのモデルをテキストで事前学習すると，1 文字での微調整相当で Python 3.7e8 文字分，300 文字なら D_E が 2.8 倍．テキスト+コード事前学習なら 4.1e9 文字，300 文字で 1.7 倍．モデルで 2 桁上，データで 5 桁下への外挿で，"We also equated data in context to data fine-tuned on."
- 対照実験（Appendix F）：事前学習テキストに Python を 0.3% 混ぜると実効 Python データは 2 倍ではなく 4.5 倍になった．これを根拠に，測ったものは "transfer from a text distribution to a python distribution rather than the uncontrolled amount of python in the pre-training distribution" と結論
- エポック数（Appendix G）：大きいデータでは最初からの学習と同程度（1〜10 エポック），最小データ（100,000 トークン）では 2〜5 倍少ない
- 統一した損失の式（Eq. 6.1）：[KMH+20] の式 1.5 のデータ量 D に D_T を代入しただけで，"speculatively" と断っている．Python を最初から学習した結果に [KMH+20] 型の全体当てはめ（Eq. E.1）を試みたが "wasn't a good enough"（Appendix E）
- 乱数の種の数や反復の数は本文に書かれていない

**本研究の記号との対応**：o（切片のずれ），d（マハラノビス距離），K（クラスごとのラベル数），q（向きの実効次元），p（次元）のどれにも対応物が無い．対応するのは主張 B の n（ソースの価値をターゲットのラベルの本数に換算した値）と D_T（事前学習の価値をターゲットの文字数に換算した値）の「換算の発想」だけである．6.1 節で [SK20] のデータ多様体の内在次元 d に触れるが，"Speculatively" と断った 1 文で，式には入らない．

### 4. 軸 A1〜A11 との対応

| 軸 | 判定 | 根拠（1 行） |
|---|---|---|
| A1 学習器 | × | 変換器全体を微調整する．凍結特徴の上の線形の頭（ロジスティック回帰・LDA・最小二乗）は出てこない（grep で linear/frozen/probe/classifier の語が本文に無いことを確認） |
| A2 理論 | × | 誤差を切片と向きに分けない．K で減る誤差の式は無い．損失の式は Eq. 6.1（[KMH+20] の冪乗則に D_T を代入）だけで，次元 p・q は入らない |
| A3 理論 | △ | 事前学習の価値をターゲットのデータ量 D_T に換算する（Eq. 1.1，1.2）．ただし単位は文字数（ラベルの本数ではない），値は実測の当てはめ（理論の導出ではない），余弦² は出てこない |
| A4 理論 | × | 切片・ドメイン別切片・合併で直る割合は扱わない |
| A5 設定 | × | 固定の少数ショットは扱わない．D_F は文字数で 3〜4 桁を動かし，最小 100,000 トークン．1 文字・300 文字への外挿は 6.2 節で "relatively speculative" と自認 |
| A6 設定 | △ | テキスト→Python の分布のずれを扱う．ずれの型（平均・共分散・ラベル比率）は分けず，α という 1 つの指数で "directed proximity" として表す |
| A7 設定 | × | 言語（テキストとコード）のみ．マルチモーダルではない |
| A8 手法 | × | 教師なしの次トークン予測で，弱いラベル・疑似ラベルは無い |
| A9 手法 | × | 出発点への L2 罰則（リッジ，仮説転移）は無い．事前学習は初期化として働くだけで，ossification を "a particularly bad initialization" と表現する |
| A10 評価 | ○ | 実データ．事前学習 24B 文字（テキスト）と 240B 文字（コード），微調整 22B 文字（Python），モデル 4 桁，データ 4 桁（当てはめは低データ体制の 3 桁） |
| A11 分析 | △ | 式 1.1 の当てはめと実測の照合はある（図 2，図 10）．予言（Eq. 6.1 の統一式，6.2 節のゼロショット・少数ショットの推定）は検証していない．Appendix F の対照実験は「移ったのは Python 混入ではない」ことの確認 |

### 5. 穴

**扱っていないこと（本記録の整理）**
- 分類課題・ラベル・クラスごとの標本数・評価指標としての正解率（6.3 節 2 項で "It's easier to generate simple equations for scaling laws where performance is measured in loss rather than accuracy" と述べ，正解率は避けている）
- 凍結特徴・線形の頭・切片と向きの分解・次元
- 複数のソースの選択・合併（事前学習データの混合比は 6.7 節の今後の課題 6・7 に置く）
- ossification の仕組み（"we didn't investigate that question"．3.1 節）
- 乱数の種・反復の数・誤差棒の記載

**限界として自分で書いていること（5 節 Limitations，原文）**
1) "Models weren't tuned for fine-tuning or code. We leveraged hyperparameters that were tuned for training from scratch on natural language [KMH+20]."
2) "Models weren't tuned for small datasets. For small datasets, training ended before the warmup was finished, so the learning schedule could confound the results."
3) "We only measured transfer when fine-tuning on python. It's unclear if we'd observe a power law fit for a broad set of distribution pairs."
4) "We only measured transfer between distributions in an unsupervised setting. It's not clear to what degree the findings would generalize to a supervised or reinforcement learning setup."
5) "We didn't find a good closed-form model for from-scratch results as was seen in [KMH+20] ... If we had such results we expect we could generate a closed-form equation for overall performance for fine-tuned models rather than rely on relative performance in our definition."
6) "We only measured performance on transformers."
7) "Equations 3.1 and 3.2 don't handle zero-shot case unless we use an approximation (i.e. fine-tuning on 1 character) for the zero-shot case."
8) "We didn't explore the ability to make sure measurements more cheaply, either in context or through KL divergence."
加えて Appendix D："We no longer get a good fit for these models once we leave the low-data regime."，図 6："the parallelism of the purple curves and from-scratch D(N) baseline in orange relies on D(N), for which we had a noisy fit"．

**今後の課題（6.7 節，原文）**
1. "Measuring transfer coefficients between more unsupervised distributions."
2. "Measuring transfer coefficient where the target distribution is much more general and/or a set of tasks."
3. "Transfer scaling laws where the from-scratch comparisons are better behaved and can generate a unified predictive equation, such as 6.1."
4. "Transfer scaling laws in supervised and reinforcement learning settings."
5. "Transfer scaling laws comparing Transformers to other architectures."
6. "A method to cheaply predict the ideal pre-training ratio for a pair of datasets A and B to maximize performance on a target distribution C, for which we have limited data."
7. "Given multiple target distributions/downstream tasks, each with limited amounts of data, a set of alpha measurements could potentially enable the construction of an optimal pre-training data split ..."

### 6. 本研究との差の書き方（関連研究の案）

案 1（換算の発想の先例として置く）：
「ソースの価値をターゲットのデータ量に換算する見方は Hernandez ら [arXiv 2102.01293] が先に示した．彼らは変換器の言語モデルを全パラメータ微調整し，事前学習の価値を『最初から学習したモデルが同じ損失に達するのに要した追加の文字数 D_T』として損失曲線の補間で実測し，D_T = k D_F^α N^β（テキスト→Python で k = 1.9e4，α = 0.18，β = 0.38）という冪乗則を当てはめた．本研究の n = q e²/(1 − e²) は同じ換算を，凍結特徴の上の線形の頭という設定で理論から導いたもので，換算の単位はクラスごとのラベルの本数，値を決めるのはモデルの大きさ N ではなく向きの実効次元 q とソースの向きの効率 e（推定した向きと真の向きの余弦）である．」

案 2（データが多いときの損の先例として置く）：
「ターゲットのデータが多いとき事前学習が損になる現象（Hernandez らの ossification．小さいモデルでは D(N) の 10〜100 倍のデータでも最初からの学習に届かない）は，本研究の『K が大きいと合併が損になる』境目 K* と対応する．Hernandez らはその仕組みを調べていない（"we didn't investigate that question"）．本研究は切片の 1/(2K) と向きの q/K の減り方から損の境目を導く．」

**本研究の式はこの文献のどの式を書き直したものか**：どの式も書き直したものではない．対応するのは Eq. 1.1（D_T）と Eq. 1.2（倍率）の「換算」の発想のみで，式の形（冪乗則 vs 有理式 q e²/(1 − e²)），変数（N と D_F vs q と e），導出（当てはめ vs 理論），単位（文字数 vs ラベルの本数）がすべて異なる．関連研究には「換算の単位を提案した先例」と書き，「式を借りた」とは書かない．

### 7. 注意

- **図が読めない**：図 1〜15 の本体は pdftotext で失われている．数値はキャプション・本文・Table 1 から拾った．図 2 の軸の値や図 13 の破れ方の程度は確かめていない
- **本記録の推論**：(i) 全パラメータ微調整であること（凍結の記述が無いことからの推論），(ii) デコーダ型であること（明記無し），(iii) 最大モデルが 10^9 のけたであること（6.2 節の「2 桁上へ外挿して GPT-3」からの逆算），(iv) 図 1 の "approximately 1000x" が実測点であること（当てはめ値は約 470 倍．本記録の計算），(v) Eq. C.1 の形（Eq. B.2 から復元）
- **原文の番号の不整合**：5 節 Limitations 7) と Appendix D は "Equations 3.1 and 3.2" "equation 3.2" を参照するが，本文に 3.1・3.2 という式番号は無い（3 節に式は無い）．Eq. 1.1・1.2 を指すと読める（本記録の推論）．Appendix C の題は "Generating fit for Equation 1.2" だが，内容は Eq. 1.1 の全体当てはめの手順である
- **未読の節**：無し（全節を読んだ）．ただし図の本体は上記のとおり読めていない
- **要確認の書誌**：掲載先（本文は arXiv v1 のみ）．モデルの大きさの最大値（本文に数値無し）．乱数の種・反復の数（記載無し）
- **本研究の論点との関係での注意**：この論文の「少ないデータ」は文字数で 10^5 以上であり，本研究の K（クラスごとに数本〜数十本）とは 3〜4 桁離れている．6.2 節の 1 文字・300 文字への外挿は著者自身が "relatively speculative" と断っているので，少数ショットの先例として引用しない方がよい（本記録の判断）
- 前回の調べは「後続の研究（arXiv 2408.16947）が D_T = k(D_F)^α(N)^β と書き直した」としたが，この式は原論文の Eq. 1.1 そのものである（本記録で訂正）

---

## 9. Dobriban & Wager（arXiv 1507.03003．Ann. Statist. 2018 と記憶するが本文で未確認）

確かめた水準：◎ 本文（下請けの精読記録．系 3.3 の "price" の文を grep で確認）．[arXiv](https://arxiv.org/abs/1507.03003)

### 1. 書誌と読んだ範囲

**書誌（◎ 本文で確認）**
- 題：High-Dimensional Asymptotics of Prediction: Ridge Regression and Classification
- 著者：Edgar Dobriban，Stefan Wager（Stanford University．連絡先 {dobriban, swager}@stanford.edu）
- 版：arXiv:1507.03003v2 [math.ST]，2015 年 11 月 4 日（1 頁目のヘッダ）．pdftotext の出力は 2708 行，48 頁ぶん
- 掲載先：**本文に記載が無い．要確認**（依頼にある「Annals of Statistics 2018」は本文からは確かめられない）
- 資金：NSF grant DMS-1418362（謝辞）．再現コード：https://github.com/dobriban/high-dim-risk-experiments/（1.2 節）

**読んだ範囲**
- 本文：1 節（1.1〜1.4），2 節（2.1〜2.3），3 節（3.1〜3.5），謝辞．定理 2.1，系 2.2，定理 3.1，定理 3.2，系 3.3，系 3.4，定理 3.5，系 3.6．式 (1)〜(13)，(16)〜(25)．図 1〜5 の説明文（軸の目盛りは読めるが曲線上の数値は読めない）
- 補遺：4 節（構成），5 節（計算法），6.1 節の冒頭（式 (14) まで），7.1 節の証明の概略と 7.1.1〜7.1.2（補題 7.1・7.2 の証明），7.2（定理 3.2 の証明），7.3（系 3.3 の証明），7.4（Ledoit–Péché の注），7.5（定理 3.5 の証明），7.6（系 3.6 の証明）
- 飛ばした：6.1 の残り，6.2，6.3（リッジの極限の証明），7.1.3〜7.1.6（補題 7.3〜7.6 の証明）
- この論文には結論・考察・今後の課題の節が無い．本文は 3.5 節の文献展望と謝辞で終わる

### 2. 設定

**共通の標本数の体制（Assumption A）**：n，p がともに無限大，p/n → γ ∈ (0, ∞)（比例の極限）．特徴は X = Z Σ^{1/2}（Z は平均 0，分散 1 の i.i.d. 成分，Σ は決定論的な共分散）．Σ の固有値の分布 F_Σ が極限の分布 H（population spectral distribution）に収束する．**固定の少数ショット（K 本で固定）は扱わない．**

**リッジ回帰（2 節）**：y_i = x_i·w + ε_i，ε の分散は 1．推定は ŵ_λ = (XᵀX + nλI)⁻¹XᵀY．切片は無い．ランダム効果の仮定（Assumption B）：E[w] = 0，Var[w] = (α²/p) I．α² = E‖w‖² は信号の強さ．評価はテスト点の予測二乗誤差 E[(y − ŵ_λ·x)²]．

**正則化判別分析 RDA（3 節）**：2 クラスのガウス模型 (2)：y ∈ {±1}，P[y=1] = 1/2，x ~ N(μ_y, Σ)．μ_{±1} = μ̄ ± δ，ベイズの重みは w* = Σ⁻¹δ，ベイズ誤り率は Φ(−Δ)，Δ = √(δᵀΣ⁻¹δ) は**クラス間マハラノビス距離の半分**（本研究の d/2 に当たる）．
- ランダム効果の仮定（Assumption C）：δ の各座標が i.i.d.，E[δ_i]=0，Var[δ_i] = α²/p，4+η 次モーメントの上界．μ̄ は固定か δ・X・y と独立で ‖μ̄‖² ≤ C p^{1/2−ζ}（全体の平均は次元に比べて小さい）
- 分類器：ŷ = sign(ŵ_λ·(x − μ̂))，ŵ_λ = (Σ̂_c + λI)⁻¹δ̂，δ̂ = (μ̂_{+1} − μ̂_{−1})/2，μ̂ = (μ̂_{+1} + μ̂_{−1})/2，Σ̂_c は中心化した標本共分散（1/(n−2)）．罰則は共分散への λI であり，ロジスティック回帰の重みへの L2 罰則ではない
- Σ の固有値は 0 < b ≤ λ_min ≤ λ_max ≤ B．定理 3.1 は均衡（P[y=1]=1/2，n_{−1} = n_{+1} = n/2）．定理 3.2 は不均衡（π_±，p/n_{±1} → γ_{±1}）
- **ずれの型：学習と評価は同じ分布．**例外は定理 3.2 で，学習データのクラス比が母集団のクラス比に一致しなくてよい（"We do not assume that n_{−1}/(n_{−1}+n_{+1}) → π_{−}"）．これはラベルの比率のずれだけで，平均や共分散のずれは扱わない

### 3. 示したこと

記号の対応：論文の Δ = 本研究の d/2．論文の n は総標本数（均衡なら n_{±1} = n/2 = K）．論文の γ = p/n = p/(2K)．論文の余弦 Γ = 本研究の e（向きの効率）．論文の切片 b̂ は本研究の o に対応（正確には (b̂ + ŵᵀμ̄)/√(ŵᵀΣŵ) が o）．本研究の q（向きの実効次元）に対応する量は論文に無い（後述の推論で p/α² が最も近い）．

**(a) リッジの予測リスク（定理 2.1）**．予測誤差は確率 1 で決定論的な極限に収束し，極限は α²，γ，λ と Σ̂ の固有値分布のスティルチェス変換 m(−λ) = lim p⁻¹tr((Σ̂+λI)⁻¹) とその相方 v（式 (4)：γ(m + 1/z) = v + 1/z）だけで決まる．最適な λ* = γ/α² で Err → 1/(λ* v(−λ*))（式 (1)）．観測量で書くと Err − [ (γ²/(α² p)) tr(Σ̂ + (γ/α²)I)⁻¹ + 1 − γ ]⁻¹ → 0．一般の λ では R_λ = (1/(λv))[1 + (λα²/γ − 1)(1 − λv'/v)]．Σ = I のときは式 (7)（m_I の閉じた形）と式 (8)．
- 学習の体制（2.1 節）：弱い信号では (R* − 1)/α² → E_H[T]（平均の固有値）で γ に依らない．強い信号では γ < 1 で R* → 1/(1−γ)（OLS と同じ），γ > 1 で R* ∝ α²（式 (9)，係数 1/(γ v(0))），γ = 1 で R* ∝ α（式 (10)）
- 不正確性原理（系 2.2）：1 − 1/R_P = γ(1 − R_E/α²)．γ = 1 なら R_E R_P = α²（予測と推定が同時に易しい Σ は無い）
- **「実効次元」「有効自由度」という語は本文に無い**（grep で "degrees of freedom"，"effective" とも 0 件）．Σ は m(−λ)，v(−λ) とその導関数（p⁻¹tr((Σ̂+λI)⁻¹)，p⁻¹tr((Σ̂+λI)⁻²) の極限）だけを通して入る．**本記録の推論**：式 (4) を z = −λ で書き直すと λv(−λ) = 1 − γ(1 − λm(−λ)) で，γ(1 − λm(−λ)) は tr(Σ̂(Σ̂+λI)⁻¹)/n（リッジの通常の有効自由度を n で割ったもの）の極限に等しい．よって式 (1) は R* = 1/(1 − df*/n) と読める．論文はこの読み替えを書いていない

**(b) RDA の誤り率（定理 3.1，均衡）**．Err(ŵ_λ) → Φ(−Θ(λ))，Θ = α²τ/√(α²η + ξ)，τ = λmv，η = (v − λv')/γ，ξ = v'/v² − 1．Θ は「推定した向きで測ったマージン」で，証明では Θ = lim ŵᵀδ/√(ŵᵀΣŵ)（式 (20)）．
- 出発点は式 (16)：任意の線形分類器 sign(w·x + b) の誤り率 Err_0(w,b) = π_− Φ((wᵀμ_{−1}+b)/√(wᵀΣw)) + π_+ Φ(−(wᵀμ_{+1}+b)/√(wᵀΣw))．論文はこれを "the well-known finite-sample formula" と呼ぶ．**本研究の主張 A の式 ½[Φ(d/2+o) + Φ(d/2−o)] は，式 (16) で π_± = 1/2，w = w*（真の向き），b を閾値のずれ o に取った特別な場合である（本記録の確認．w = Σ⁻¹δ，μ_{±1} = μ̄ ± δ を入れると wᵀδ = Δ²，√(wᵀΣw) = Δ となり，1 − Err = ½[Φ(Δ − o) + Φ(Δ + o)]）**
- **切片の扱い**：RDA の切片は 2 クラスの標本平均の中点で決める（b̂ = −μ̂ᵀŵ_λ，7.1 節では b̂ = δ̂ᵀ(Σ̂_c+λI)⁻¹μ̂ と書く）．**補題 7.1：均衡なら b̂ → 0（確率 1）．**よって極限の誤り率に切片の推定誤差の項は無く，Err_0 → Err_1 → Err_2 = Φ(−wᵀδ/√(wᵀΣw)) と簡約する（式 (17)，(18)）．証明（7.1.1）では μ̂ = μ̄ + Σ^{1/2}W/√n，δ̂ = δ + Σ^{1/2}Z/√n と表し（式 (21)），b̂ を 4 項 T_1〜T_4（式 (22)〜(25)）に分けてそれぞれ 0 に収束することを示す．T_1 = δᵀLμ̄ が消えるのは ‖μ̄‖² ≤ C p^{1/2−ζ} の仮定による．T_3 = n^{−1/2}δᵀLΣ^{1/2}W（閾値の標本誤差）は独立な 2 つのベクトルの双線形形式として 0 に収束する．**本記録の推論**：T_3 の分散は 1/n の位数であり，本研究の主張 A の「切片の誤差は 1/(2K) で減る」と同じ位数だが，論文は O(1) のマージン Θ に対して無視できる項として扱い，大きさを式にしない
- 向きの寄与：補題 7.3 で ŵᵀδ → α² m(−λ)，補題 7.4・7.5 で ŵᵀΣŵ → α²(v − λv')/(γ(λv)²) + (v' − v²)/(λ²v⁴)（式 (19) の分解 Ã + 2B̃ + C̃．Ã は真の δ の項，C̃ は δ̂ − δ の雑音の項）

**(c) 不均衡（定理 3.2，式 (11)）**．Err → π_− Φ(−Θ_−) + π_+ Φ(−Θ_+)，Θ_± = ∓[±α²m(−λ) + ((γ_{−1} − γ_{+1})/4)(1/γ)(1/(λv) − 1) + c]/√Q，Q = α²(v−λv')/(γ(λv)²) + ((γ_{−1}+γ_{+1})/4)(v'−v²)/(λ²v⁴)．ここで分類器は f̂(x) = xᵀŵ_λ − μ̂ᵀŵ_λ + c（c は使い手が足す定数）．**7.2 節の証明：切片 −μ̂ᵀŵ_λ は (γ_{−1} − γ_{+1})/4 · κ(λ) に収束する**（κ(λ) = (1/γ)(1/(λv)−1) = lim p⁻¹tr(Σ(Σ̂+λI)⁻¹)，Ledoit–Péché の式 (5)）．つまりクラスごとの標本数が違うと切片に決定論的な偏りが残り，2 つのクラスのマージンに逆符号で入る．これは本研究の「マージン ± 閾値のずれ」と同じ構造である（本記録の対応づけ）．論文は c を最適化しない（偏りを c で打ち消せることも書いていない．本記録の推論）

**(d) 余弦の幾何（3.2 節）**．Σ の計量での余弦 cos_Σ(w*, ŵ_λ) = ŵᵀδ/√(ŵᵀΣŵ · δᵀΣ⁻¹δ) は決定論的な極限 Γ(H, γ, α², λ) に収束し，**Γ = Θ/Δ**（Δ = α√E_H[T⁻¹] は脚注 6）．pdftotext では "Γ = Θ ∆" と分数線が落ちているが，系 3.4 の式 lim Γ = τ/√(η E_H[T⁻¹]) と定理 3.1 の Θ/α → τ/√η，Δ/α = √E[T⁻¹] の整合から Θ/Δ と確認した．論文は Γ を "the inefficiency of the RDA estimator relative to the Bayes one" と呼ぶ．
- **系 3.3（Σ = I）**：Γ = [α/√(α²+γ)] · √((1 + γλ m_I²)/(1 + γ m_I))．論文の解釈："We need to pay a price α/√(α²+γ) for estimating μ_{±1}, while the cost of estimating Σ is ([1+γλm_I²]/[1+γm_I])^{1/2}."  λ → ∞（Σ = I を知っている場合）で第 2 因子は 1 になり，損失は α/√(α²+γ) だけになる
- **系 3.4（α → ∞）**：lim Γ = τ(λ)/√(η(λ) E_H[T⁻¹]) < 1．信号が無限に強くても RDA はベイズの向きに一致しない．最適な λ は H だけで決まる 0 でない極限をもつ（リッジでは λ* = γ/α² → 0 で対照的）．図 4（AR-1(0.9)，α = 0.1〜2）で最適な λ* が α とともに増えることを示す
- 低次元（p 固定，n → ∞）では Efron (1975) により Γ → 1，n(1 − Γ) → (定数倍の) χ²_{p−1}．高次元での標本分布は "interesting future work"

**(e) 主張 B との対応（本記録の推論．論文はこの書き換えを書いていない）**
- Σ = I，λ → ∞（平均の差の向き，独立則）なら Γ² = α²/(α² + γ) = α²/(α² + p/n) = **n/(n + p/α²)**．q := p/α² と置くと **n = q Γ²/(1 − Γ²)** となり，主張 B の n = q e²/(1 − e²) と形が一致する．クラスごとの本数 K = n/2，d = 2α で書くと Γ² = K/(K + 2p/d²)，1 − Γ² ≈ 2p/(K d²)（K ≫ p/d² のとき）．すなわち主張 A の「向きの誤差は q/K」の q に当たる量は，この論文の枠では p/(2α²) = 2p/d²（次元を信号の強さで割ったもの）で，**p より小さい実効次元という概念は無い**（δ の各座標が同じ分散をもつ等方的なランダム効果の仮定のため）
- 一般の Σ で λ → ∞（定理 3.5 の Θ_IR）：Γ²_IR = [1/(E[T]E[T⁻¹])] · 1/(1 + γE[T²]/(α²E[T]))．第 1 因子は無限の標本でも残る天井（Jensen により ≤ 1），第 2 因子は n/(n + q) の形で q = pE[T²]/(α²E[T])
- LDA（λ → 0，γ < 1），Σ = I：Γ²_LDA = (1 − p/n) · n/(n + p/α²)．共分散の推定の損失 (1 − p/n) と平均の推定の損失の積
- **論文は余弦をソースの価値やラベルの本数に換算しない．**Γ は「ベイズに対する非効率」として扱う

**(f) LDA と独立則（定理 3.5，系 3.6）**．Θ_LDA = α²√(1−γ) E_H[T⁻¹]/√(α²E_H[T⁻¹] + γ)（γ < 1），Θ_IR = α²/√(α²E_H[T] + γE_H[T²])．Raudys (1967)・Deev (1970)・Raudys (1972)・Saranadasa (1993) の既知の式を定理 3.1 の極限として再現する（Saranadasa との差は μ_± をランダムにしたことによる分母の γ の項）．系 3.6：固有値が [k_1, k_2]，平均 1 の H の族での最悪値は LDA が Σ = I で α²√(1−γ)/√(α²+γ)，独立則が 2 点混合で α²/√(α² + γ(k_1+k_2−k_1k_2))．独立則が最悪値で LDA に勝つ条件は α² + 1 > (1−γ)(k_1+k_2−k_1k_2)．

**(g) 数値実験（すべて人工データ）**
- 図 1（RDA）：BinaryTree 共分散は p = 1024，n = γ⁻¹p（γ = 0.5, 1, 2 → n = 2048, 1024, 512）．Exponential 共分散（指数分布の等間隔分位点が固有値）は n = 500．μ_{±1} ~ N(0, α²p⁻¹I) で α はベイズ誤り率が 1% になるように調整．テスト 10,000 本．x 軸は λ（0〜5 または 0〜3）．主張：条件数が同程度でも Exponential は BinaryTree より "vastly more difficult"．理論曲線は "accurate even in moderate sample sizes"
- 図 2（リッジ）：BinaryTree は p = 2⁴ = 16（本文の "p = 24" は上付きが落ちたものと読む．本記録の推論），n = γ⁻¹p（γ = 0.5, 1, 2 → n = 32, 16, 8）．Exponential は n = 20．α² = 1．本文は「500 回の反復，テスト 100 本」，図の説明文は「100 組の学習データ，各 500 本のテスト，計 50,000 本」と食い違う．主張："accurate, even in small sized problems"．2 つの共分散でリッジの予測誤差は似ており，RDA（図 1）との対照を指摘
- 図 3：Σ = I，γ = 0.25, 0.5, 0.8, 0.9, 1, 1.1, 1.3, 2, 4, 8，α² = 1〜10⁴ の両対数で R* − 1 の傾きが γ > 1 で 1，γ = 1 で 1/2，γ < 1 で 0
- 図 4：AR-1(0.9)，α = 0.1〜2（刻み 0.05），λ = 0〜1.5，y 軸 "Relative Margin"（= Γ）は 0〜0.25．この図の γ は本文に書かれていない
- 図 5（3.3 節）：n = p = 500，Σ_ij = ρ^{|i−j|}，ρ = 0.1, 0.5, 0.9．左列 α² = 1，右列 √E[Δ²] = 2.3（ベイズ誤り率 1%）．AR-1 の極限のマージン Δ = α√((1+ρ²)/(1−ρ²))．テスト 10,000 本．主張：λ = 0 の LDA は VC 理論の予言どおり失敗，ρ が大きいと小さい正の λ の RDA が良い．Rademacher 理論は λ ≈ 1 を説明できない（"Rademacher theory is not tight enough to predict what will happen for λ ≈ 1"）

### 4. 軸 A1〜A11 との対応

| 軸 | 判定 | 根拠 |
|---|---|---|
| A1 学習器 | △ | 線形の頭（リッジ回帰，共分散を縮めた LDA = RDA）を扱うが，ロジスティック回帰ではなく，凍結した特徴という設定も無い（x は生の確率変数） |
| A2 誤差の分解 | △ | 誤差は向き（余弦 Γ → マージン Θ = ΓΔ）で決まり，切片は均衡で 0 に収束（補題 7.1），不均衡で決定論的な偏り (γ_{−1}−γ_{+1})κ(λ)/4（定理 3.2）．K で減る式は無く，p は γ = p/n と固有値分布 H を通して入る．実効次元は無い |
| A3 余弦² と標本数 | △ | 余弦の極限 Γ を閉じた形で与える（定理 3.1，系 3.3，3.4，定理 3.5）．Σ = I なら Γ² = n/(n + p/α²)（本記録の推論）．ソースの価値をラベルの本数に換算する議論は無い |
| A4 合併と切片 | × | 合併もドメイン別の切片も無い．近いのは定理 3.2 のクラス数不均衡による切片の偏りだけ |
| A5 少数ショット | × | p/n → γ の比例の極限．固定の K は扱わない．数値実験では n = 8〜32（図 2）でも式が合うと述べる |
| A6 ドメインのずれ | △ | 学習と評価は同じ分布．定理 3.2 で学習データのクラス比が母集団と違ってよい（ラベルの比率のずれだけ）．平均・共分散のずれは無い |
| A7 マルチモーダル | × | 無い |
| A8 弱いラベル | × | 無い |
| A9 出発点への罰則 | × | 罰則は原点向け（リッジは w を 0 へ，RDA は共分散に λI）．仮説転移は無い |
| A10 実データ | × | 人工データのみ．文書分類・バイオ情報学（Bernau et al. 2014 等）は動機として引用するだけ |
| A11 予言と実測 | ○ | 図 1，2，5 で理論曲線と模擬実験を重ね，n = p = 500 や p = 16 でも合うと述べる（実データでの照合は無い） |

### 5. 穴

**扱っていないこと**
- 固定の少数ショット（K 本）での有限標本の誤差の式．切片の推定誤差の大きさ（極限で 0）
- 向きの実効次元が p より小さい状況（δ は等方的なランダム効果で，信号は全次元に薄く広がる）
- 学習と評価のドメインのずれ（ラベル比率以外），複数ソースの合併，転移
- ロジスティック回帰（損失が 0-1・二乗のみ），多クラス（2 クラスのみ），実データ，特徴抽出器
- 高次元での Γ の標本分布（ばらつき）

**限界として自分で書いていること（原文）**
- 仮定の強さ："Our hypothesis is of course very strong; however, it yields a qualitatively different theory for high-dimensional prediction than popular approaches"（1 節）
- ガウス性と独立性："In this section we assumed Gaussianity, but by a Lindeberg–type argument it should be possible to extend the result to non-Gaussian observations with matching moments. It should also be interesting to study how sensitive the results are to the particular assumptions of our model, such as independence across samples."（定理 3.2 の直後）
- RDA の正則化の位置づけ："for classification, the ridge regularization is merely a heuristic to help with the ill-conditioned sample covariance."（3.1 節末）

**今後の課題（原文）**
- "Establishing the sampling distribution in high dimensions is interesting future work."（3.2 節末）
- "These formulas may prove useful for the analysis of other statistical methods under high-dimensional asymptotics, such as principal component regression and kernel regression."（1.1 節）
- "We hope that further work motivated by generalizations of the random-effects hypothesis could yield a new theoretical underpinning for dense high-dimensional prediction."（1 節）

### 6. 本研究との差の書き方（関連研究の案）

案 1（主張 A の置き場所）：
「Dobriban and Wager（2015）は，2 クラスのガウス模型で正則化判別分析の誤り率が Φ(−Θ) に収束し，Θ がベイズの向きと推定した向きの余弦（Σ の計量）とマハラノビス距離の積で決まることを，p/n → γ の比例の極限で示した．同論文では切片は 2 クラスの標本平均の中点で決め，均衡なら切片の誤差は極限で消える（補題 7.1）．本研究はクラスあたり K 本が固定の有限標本を扱い，同じ誤り率の式（同論文の式 (16)）の中で，切片の誤差 1/(2K) と向きの誤差 q/K を分けて測る点が異なる．」

案 2（主張 B の置き場所）：
「同論文の系 3.3 は Σ = I のとき余弦の極限を α/√(α² + p/n) と平均の推定に伴う損失として与え，これは余弦² = n/(n + p/α²) と書き直せる．本研究の n = q e²/(1 − e²) はこの関係を逆に解いた形であるが，q を p/α² ではなく実測の実効次元（16〜22）とし，e をソースで推定した向きの効率と読み替えて，ソースの価値をターゲットのラベルの本数に換算する点が新しい．同論文は余弦をベイズに対する非効率として扱い，標本数への換算は行っていない．」

本研究の式との対応（要点）：主張 A の正解率の式は同論文の式 (16)（論文は "well-known" と呼ぶ）の均衡・真の向きの場合．主張 B の余弦² = n/(n+q) は系 3.3（または定理 3.5 の Θ_IR）の第 1 因子 α²/(α²+γ) を n で書き直したものに当たり，q = p/α² に相当する．違いは (i) 比例の極限か有限の K か，(ii) q が p に比例する量か実測の小さい実効次元か，(iii) ソースとターゲットの区別の有無，(iv) 学習器（RDA かロジスティック回帰か）．

### 7. 注意

- Γ = Θ/Δ は pdftotext の崩れ（"Θ ∆"）を系 3.4 との整合から復元した（本記録の推論．分数線が落ちたと判断）
- 図 2 の p = 2⁴ = 16 は本文の "p = 24" からの推論（BinaryTree は深さ d で p = 2^d）．図 2 の反復回数とテスト本数は本文と図の説明文で食い違う（500 回×100 本 と 100 組×500 本）
- 3 節 (e) の n/(n+q) への書き換え，1 − Γ² ≈ 2p/(Kd²)，R* = 1/(1 − df*/n) の読み替え，定理 3.2 の切片の偏りと本研究の o の対応，T_3 の分散が 1/n の位数という指摘は，すべて本記録の推論であり論文の主張ではない
- 定理 3.1 の Θ の式（τ, η, ξ）は pdftotext で判読できたが，定理 3.2 の Θ_± の符号の配置は崩れが大きく，7.2 節末の U の式と照らして復元した．c の符号の扱いに自信が無い
- 図 4 の γ は読んだ範囲に書かれていない．図の曲線上の数値は読めないため，実験の「主な数値」は設定と本文の定性的な主張に限る
- 未読：6.1 後半，6.2，6.3，7.1.3〜7.1.6．定理 2.1 の第 2 部（一般の λ）と補題 7.3〜7.5 は主張だけ確認し，証明は読んでいない
- 掲載先（Annals of Statistics 2018 か）は本文に無く要確認．読んだのは arXiv v2（2015-11-04）であり，掲載版で定理番号や式番号が変わっている可能性がある
- 補足（本記録）：Raudys の T_μ = 1 + 2p/(δ²N)（2 節）と系 3.3 の第 1 因子 α²/(α²+γ) は同じ関係である．Raudys の δ は Dobriban & Wager の 2Δ = 2α（Σ = I）に当たり，N 本ずつ（n = 2N）なら γ/α² = p/(2Nα²) = 2p/(Nδ²) で一致する

---

## 10a. Zhang, Fan, Luo, Hu 2026（arXiv 2607.06936．転移の LDA）

### 1. 書誌と読んだ範囲
- Zhang, Y., Fan, Y.（Northeast Normal University），Luo, W.（Zhejiang University of Finance and Economics），Hu, J.（Northeast Normal University），"Transfer Learning for Linear Discriminant Analysis with a Shared Classification Signal," arXiv:2607.06936v1 [stat.ME]，2026-07-08（本文の日付は July 9, 2026）[arXiv](https://arxiv.org/abs/2607.06936)．掲載先は本文に無い（**要確認**）．全 66 頁．
- 読んだ範囲（◎）：Abstract，§1，§2（式 (1)，Assumption 1〜5，式 (4)〜(6)，Theorem 1・2，Proposition 1，Corollary 1，式 (11)〜(15)，Theorem 3），§2.2 の冒頭（Assumption 6・7 の見出しまで），§3（3.1 の設定，3.4 の切片の補正の実験，Figure 4 の説明），§4（Table 1・2），§5．飛ばした：§2.2 の本体（Theorem 4・5，Proposition 2，Corollary 2），§3.2・3.3，付録の証明（Lemma 1〜9）．

### 2. 設定
- 2 クラス，ドメイン K 個（ターゲットは k = K，ソースは k = 1..K−1．**論文の K はドメイン数で，本研究の K ではない**）．Assumption 1：X_k = 平均 + Z_k Σ_k^{1/2}（Z は i.i.d.，4 次モーメント有限．ガウスでなくてよい）．Assumption 2：平均の差 μ_k = μ̄ + δ_k（μ̄ は共有の決定論的な信号，δ_k は座標ごと i.i.d. で分散 α_k²/p のランダムなドメイン固有のずれ）．Assumption 3（同質）：共通の spiked 共分散 Σ = σ²(I_p + Σ_j λ_j v_j v_jᵀ)．Assumption 4：p, n_{k,s} → ∞，p/n_{k,s} → γ_{k,s} ∈ (0, ∞)（**比例の極限**．クラス s = ± ごと）．Assumption 6・7（異質）：ドメインごとの spiked 共分散．
- 分類器（式 (5)(6)）：向き d̂(w) = Σ_k w_k Σ̂⁻¹ μ̂_k（ドメインごとの LDA の向きの重みつき和．Σ̂ はスペクトルを直した共分散の推定量 (4)），切片 b̂_K(w) = −d̂(w)ᵀ(μ̂_{+1,K} + μ̂_{−1,K})/2（**ターゲットのクラス平均の中点で決める**）．罰則は無い（共分散の縮小に当たる (4) はある）．
- 評価（式 (1)）：Gaussian-calibrated error Err(d̂, b̂) = ½Φ((d̂ᵀμ_{−1,K} + b̂)/√(d̂ᵀΣ_K d̂)) + ½Φ(−(d̂ᵀμ_{+1,K} + b̂)/√(d̂ᵀΣ_K d̂))．テスト分布のクラス比は均衡．**本研究の ½[Φ(d/2 + o) + Φ(d/2 − o)] と同じ形**（d は推定した向きで測った標準化マージン，o は切片のずれ．本記録の対応づけ）．
- ずれの型：平均のずれ（δ_k），共分散の異質性（§2.2），ターゲットのクラス比の不均衡（Δ_K）．

### 3. 示したこと
- Theorem 1：spiked 共分散の標本固有ベクトルの射影の極限 (λ_j² − γ)/(λ_j(λ_j + γ))．
- Theorem 2（式 (7)）：Err(w) − ½Φ(−(u_pᵀw + w_KΔ_K)/(2√(wᵀA_pw))) − ½Φ(−(u_pᵀw − w_KΔ_K)/(2√(wᵀA_pw))) → 0．u_p（信号）の k 成分は ‖μ̄‖²/σ² − Σ_j b_j a_j (μ̄ᵀv_j)²/σ² + (α_K²/σ²)1{k=K}，A_p（分散）の kk′ 成分は ‖μ̄‖²/σ² + Σ_j c_j (μ̄ᵀv_j)²/σ² + (α_k²/σ² + τ_k)1{k=k′}，τ_k = γ_{k,+} + γ_{k,−}（クラスごとの次元と標本数の比の和），Δ_K = γ_{K,+} − γ_{K,−}．"The diagonal term α_k²/σ² + τ_k shows that stronger domain-specific variation and larger classwise dimension-to-sample-size ratios increase the variability of using domain k. When the target-domain class sample sizes are unbalanced, the additional term w_KΔ_K appears in the two margins."
- Proposition 1：切片に t を足したときの最適な補正は t̂* − ½ w_K Δ̂_K → 0，補正後の極限は ½Φ(−u_pᵀw/(2√(wᵀA_pw))) × 2 = Φ(−u_pᵀw/(2√(wᵀA_pw)))．"the optimal intercept correction removes the target-domain imbalance term w_KΔ_K"．
- Corollary 1：最適な重み w* ∝ A_p⁻¹u_p（"generalized signal-to-noise maximizer"．共有信号が大きいドメインほど重く，α_k² や τ_k が大きいドメインは軽い）．Theorem 3：差し込み推定量 ŵ* は一致（κ̂_p = 平均の内積の平均 (11)，β̂_k (12)）．
- 実験（§3）：K = 3，α = (0.2, 0.5, 1)，σ² = 1，スパイク 3 本，1000 回．3.1：p = 20..400，p/n = 0.5 で極限の式と実測が一致（図の数値は読めない）．3.4（切片の補正）：p = 100，n_k = 200，n_{k,+} = 30..170．補正無しの誤差は不均衡で上がり（約 0.29→0.33），理論の補正はテスト集合で最適化した oracle の切片に近い（Figure 4）．
- 実データ（§4）：ADHD-200（4 サイト，310 人，p = 60..100，ターゲットの学習標本 n = 28〜115，50 分割）．Table 1 の例：site 3（n = 28）p = 60 で LDA 0.633，LDA-POOL 0.667，TL-RDA 0.410，TLDA-E 0.750，TLDA-O 0.753．site 1（n = 115）p = 80 で LDA 0.674，POOL 0.689，TLDA-O 0.766．PPMI（4 サイト，612 人，p = 160..200，n = 53〜218）．Table 2 の例：site 2（n = 122）p = 160 で LDA 0.584，POOL 0.605，TLDA-E 0.733．"The proposed TLDA methods achieve the best performance in all reported settings"．
- 記号の対応と本記録の推論：(i) 式 (1) は本研究の正解率の式と同じ形．(ii) w_KΔ_K はターゲットのクラス比の不均衡で切片が偏る量（本研究の o の 1 つの源．本研究はクラスごとに K 本で均衡なので Δ_K = 0）．(iii) A_p の対角項 α_k²/σ² + τ_k は「ソース k の向きのずれ（α_k²）」と「ソース k の標本数による向きの雑音（τ_k = 2p/n_k 相当）」を同じ単位で足している．向きの効率で言えば，ソース k を 1 つだけ使うときのマージン² は u_kk²/A_kk で，スパイク無しなら ≈ δ⁴/(δ² + α_k²/σ² + τ_k)（δ² = ‖μ̄‖²/σ²）．これは Raudys の 1/T_μ・Dobriban & Wager の Γ² と同じ構造で，ドメイン固有のずれ α_k² が「次元と標本数の比 τ」と同じ単位で入るので，**ソースの向きのずれをターゲットの標本数に換算する式がこの極限から 1 段で出る**（2p/n_eq = τ_k + α_k²/σ² と置く）．論文はこの換算を書いていない．

### 4. 軸との対応
| 軸 | 判定 | 根拠 |
|---|---|---|
| A1 | △ | LDA（t 検定で選んだ生の特徴）．凍結深層特徴・ロジスティック回帰ではない |
| A2 | △ | 切片の偏り（w_KΔ_K）と向きの分散（τ_k，α_k²）を式 (7) で分けて書く．p/n → γ の極限で，1/K の式ではない．実効次元は無い |
| A3 | △ | 最適な重み w* ∝ A⁻¹u は与えるが，ソースをターゲットの本数に換算しない．換算は A_kk の構造から出せる（本記録の推論） |
| A4 | △ | 切片はターゲットの平均で明示的に作る．合併で直る割合は扱わない |
| A5 | × | 比例の極限．実データのターゲット n = 28〜218 |
| A6 | ○ | 平均のずれ（ランダム），共分散の異質性，ターゲットのクラス比の不均衡 |
| A7 | × | 表形式の特徴（PPMI は多モーダルなデータ集合だが特徴は単一） |
| A8 | × | — |
| A9 | × | 向きの重みつき和（罰則ではない） |
| A10 | ○ | ADHD-200・PPMI の各 4 サイト，50 分割 |
| A11 | ○ | 極限の式と実測（§3.1），切片の補正と oracle（§3.4）．実データでは理論値との照合は無い |

### 5. 穴
- 扱っていないこと：有限の少数ショット，ロジスティック回帰，凍結特徴，ソースの価値の換算，合併（LDA-POOL は基準手法のみ），中心化，多クラス，実効次元．
- 限界・今後の課題（§5）："the present model focuses on linear discriminant rules, while nonlinear classification methods may be more suitable for some complex data. Second, it would be interesting to study how to select useful source domains automatically when some source domains are weakly related to the target domain. Finally, future work may consider more general types of domain heterogeneity beyond the spiked covariance models".

### 6. 本研究との差の書き方
「Zhang ら（2026）は高次元 LDA の転移で，ドメインごとの判別の向きの重みつき和とターゲットのクラス平均で決めた切片からなる分類器の誤り率の極限を，共有信号・ドメイン固有のずれ・次元と標本数の比で表し，最適な重みとクラス比の不均衡による切片の偏りの補正を与えた（Theorem 2，Proposition 1，Corollary 1）．本研究は同じ 2 つの Φ の和の誤り率の式を使うが，比例の極限ではなくクラスあたり K 本の有限の体制で，切片の誤差 1/(2K) と向きの誤差 q/K を分け，ソースの価値をラベルの本数に換算する点が異なる．Zhang らの分散項 α_k²/σ² + τ_k は，ソースの向きのずれを次元と標本数の比と同じ単位で数えており，本研究の換算と同じ考え方を含む．」
式の対応：本研究の正解率の式 = Zhang らの式 (1)．本研究の o の源の 1 つ（クラス比）= w_KΔ_K．本研究の n = q e²/(1 − e²) は A_kk の対角項から導ける関係に対応する（本記録の推論）．

### 7. 注意
- A_kk からの換算，式 (1) との同一視は本記録の推論．§2.2（異質な共分散）と証明は未読．掲載先は要確認．

---

## 10b. Luthra, Salunkhe, Galanti 2026（arXiv 2603.03530．少数ショットの上界）

確かめた水準：◎ 本文（下請けの精読記録．Theorem 4.1 の条件文を grep で確認）．[arXiv](https://arxiv.org/abs/2603.03530)

### 1. 書誌と読んだ範囲

- 題名：Directional Neural Collapse Explains Few-Shot Transfer in Self-Supervised Learning
- 著者：Achleshwar Luthra*, Yash Salunkhe*, Tomer Galanti*（* は equal contribution．所属は Department of Computer Science & Engineering, Texas A&M University．連絡先 galanti@tamu.edu）
- 版：arXiv:2603.03530v1 [cs.LG]，2026-03-03 投稿．本文の脚注に "Preprint. March 5, 2026." とある．掲載先は本文に無い（要確認．書式に "Impact Statement" 節があり ICML 系の体裁と読めるが，本記録の推論）．コードは "[project page]" と書かれているだけで URL は無い
- 確かめた水準：◎ 本文（pdftotext の出力 1601 行）
- 読んだ範囲：Abstract，§1（1.1 Contributions を含む），§2 Related Work，§3 Problem Setup，§4.1 Background，§4.2 Results（Theorem 4.1，定数 4 の最適性），§4.3（Proposition 4.2，因子模型），§5.1–5.2 Experiments（Fig. 2–5 の説明文），Impact Statement，App. A（実験の詳細），App. B（階層クラスタ，Fig. 6），App. C（Proposition C.1 と証明，Theorem C.2，Theorem 4.1 の証明，Proposition C.3 と証明），App. D（D.1–D.4，Lemma D.1，Proposition D.2）
- 読まなかった範囲：参考文献（約 780–1045 行）．図の曲線の値は文字として残っていないため読めず，軸の注記（相関係数と p 値）だけ拾った
- 本文に Conclusion の節と Limitations の節は無い（§5.2 の後に Impact Statement が続き，本文が終わる）

### 2. 設定

- 模型：凍結した表現 f: X → R^d（d は特徴の次元．本研究の p に相当）の上で，クラス c ごとに分布 D_c をおく．分布の形は仮定しない（正規性なし）．仮定は，クラス内で i.i.d.，クラス間で独立，クラス平均 μ_c と共分散 Σ_c が存在，4 次モーメント M_{4,c} = E‖f(x) − μ_c‖^4 < ∞（Proposition C.1 と Theorem 4.1），評価はクラスを一様に選ぶ均衡混合（§3）．さらに "we assume throughout that E[Δ_{i→j}] > 0"（§3）
- 分類器：主役は最近傍重心（NCC）．m 本の支援標本の平均 μ̂_c で重心を推定し，ŷ(z) = argmin_c ‖z − μ̂_c‖² = argmax_c (2 μ̂_c^T z − ‖μ̂_c‖²)．線形の頭（LP）は "the minimizer of the population 0–1 error over all linear rules g(z) = argmax_c (w_c^T z + b_c)"（§3）と定義され，NCC が線形則の一つであることから err_LP ≤ err_NCC（§3）という形でしか扱われない．ロジスティック回帰も L2 罰則も出てこない
- 標本数の体制：有限．クラスあたり m 本（本研究の K に相当）で，非漸近の上界．Theorem 4.1 は m ≥ 10，Theorem C.2 は m ≥ 1 で成り立つ．実験は m = 1〜500
- ずれの型：無し．支援標本とテスト標本は同じ D_c から独立に引く（§3，App. C 冒頭）．ドメインの違いは扱わない．クラスの比率は均衡を仮定（Proposition 4.2 では "balanced and independent"）
- 記号の対応（本記録の整理）：論文の m ↔ 本研究の K．論文の C′（または K_1, K_2）はクラス数（論文の K はクラス数を指すので注意）．論文の d は特徴の次元 ↔ 本研究の p．論文の d_ij = ‖μ_j − μ_i‖_2 はユークリッドのクラス間距離（本研究の d はマハラノビス距離なので同じではない）．論文の Ṽ_ij = u_ij^T Σ_i u_ij / d_ij² は「判定の軸に沿ったクラス i のばらつき ÷ クラス間距離の 2 乗」で，軸に沿った標準偏差を σ_axis とすれば Ṽ_ij = 1/(d_ij/σ_axis)²．Σ が u_ij の固有方向を持つときに限り本研究の 1/d² と一致する（本記録の推論）

### 3. 示したこと

**(a) 主定理（Theorem 4.1，§4.2 と App. C）．**NCC のクラス平均誤差は次で抑えられる（pdftotext から復元．App. C の証明と整合するので形には自信がある）．

err^NCC_{m,C}(f) ≤ (1/C′) Σ_i Σ_{j≠i} [ 4 Ṽ_ij / (1 + (v_j − v_i)/(m d_ij²))² ] + (1/C′) Σ_i Σ_{j≠i} [ (√E¹_ij + √E²_ij + √E³_ij)² / (1 + (v_j − v_i)/(m d_ij²))² ]

ここで v_c = tr(Σ_c)（全方向の級内分散の和），V_ij = (v_i + v_j)/d_ij²（古典の CDNV），Θ_ij = (M_{4,i} + M_{4,j})/d_ij⁴，
E¹_ij = (1/m)(V_ij² + V_ij/4)，E²_ij = V_ij/m，E³_ij = (Θ_ij + 2(m−1)V_ij²)/m³．

日常語で言うと：誤差の主項は「クラスを分ける軸に沿ったクラス内のばらつき ÷ クラス間距離の 2 乗」の 4 倍で，m によらない．残りの項は重心を m 本で推定したことによる漏れで，古典の CDNV（全方向のばらつき）の 1 乗と 2 乗が 1/m で，4 次モーメント（裾の重さ）が 1/m³ で減る．原文："E¹_ij captures linear centroid-estimation leakage (≍ V_ij/m), E²_ij captures quadratic leakage (≍ V_ij²/m), and E³_ij captures tails and higher-order terms (≍ Θ_ij/m³ + V_ij²/m²)"（§4.2）．

分母の (1 + (v_j − v_i)/(m d_ij²))² は，クラスごとの全分散の差から生じる 1/m 次の系統的な閾値のずれ．App. D.4 の原文："the effective threshold in the one-dimensional reduction is rescaled from d_ij/2 to (d_ij/2)(1 + α_ij)"，α_ij = (v_j − v_i)/(m d_ij²)．

次元の入り方：定理に d（次元）は陽に現れない．次元は v_c = tr(Σ_c) を通してだけ入る．等方の雲（Σ = σ² I_p）では V_ij = 2pσ²/d_ij² = 2p Ṽ_ij なので，補正項は p Ṽ/m と p² Ṽ²/m の桁になる（本記録の計算．論文に p の記述は無い）．実効次元の概念は無い．

**(b) 補正の分け方（Proposition C.1 と証明，App. C）．**Theorem 4.1 は，重みづき係数つきの対ごとの上界（Eq. (2)）で λ_T, λ_S, λ_Q をコーシー・シュワルツで最適化したもの（Theorem C.2 は λ をすべて 1 にしたもの）．証明は余裕 Δ を Δ = d_ij² − 2 d_ij X − 2T + S + Q に分解する：
- X = (z_i − μ_i)^T u_ij：テスト点の軸方向の揺れ（主項 4Ṽ の源）
- T = A^T(δ_j − δ_i)：テスト点と重心誤差の差 δ_j − δ_i との積（δ_c = μ̂_c − μ_c）
- S = (μ_j − μ_i)^T(δ_j + δ_i)：重心誤差の和を真の軸に射影したもの
- Q = ‖δ_j‖² − ‖δ_i‖²：重心のノルムの差の誤差
Var(T) = (1/m) tr((Σ_i + Σ_j)(Σ_i + ¼(μ_j − μ_i)(μ_j − μ_i)^T))，Var(S) = (1/m)(μ_j − μ_i)^T(Σ_i + Σ_j)(μ_j − μ_i)，Var(Q) = Σ_c [(M_{4,c} − v_c²)/m³ + 2(m−1) tr(Σ_c²)/m³]．最後に Chebyshev で Pr(Δ ≤ 0) ≤ Var(Δ)/(EΔ)²．

本記録の推論（論文は言っていない）：NCC は w_c = 2μ̂_c，b_c = −‖μ̂_c‖² の線形則なので，δ_j − δ_i は向きの誤差，S は閾値（中点）の軸方向のずれ，Q は切片のノルム項の誤差に当たる．Var(S)/d_ij⁴ = (Ṽ_ij + Ṽ_ji)/m は次元によらず 1/m で減り，等方の雲では Var(T)/d_ij⁴ ≈ 2p Ṽ²/m + Ṽ/(2m) で次元 p に比例する．つまり証明の中には本研究の主張 A（切片 1/(2K)，向き q/K）と同じ構造の分け方が存在する．ただし論文は Var(T), Var(S) を全分散 V_ij で上から抑えて（tr(Σ_c²) ≤ v_c² を使う）一つの「重心推定の漏れ」にまとめるため，定理の文面ではこの分け方は見えない．等方の雲ではこの上から抑える段で p 倍ゆるくなる（本記録の計算）．

**(c) 主項の定数 4 の最適性（§4.2，App. D）．**重心既知（m → ∞）では，対ごとの誤差は 1 次元の裾の事象 p_{i→j} = Pr(X ≥ d_ij/2)（Eq. (3)）に帰着し，Cantelli の不等式で p_{i→j} ≤ 4Ṽ_ij/(1 + 4Ṽ_ij) ≤ 4Ṽ_ij（Eq. (4)）．2 点分布でこれは等号になる（Proposition D.2）．原文："any universally valid second-moment bound of the form p_{i→j} ≤ c Ṽ_ij + (lower-order terms) must have c ≥ 4 unless additional tail assumptions are imposed (e.g., subgaussianity)"（§4.2）．本記録の対応づけ：Eq. (3) を正規の雲に特化すると誤差は Φ(−d_M/2)（d_M はマハラノビス距離）となり，本研究の ½[Φ(d/2 + o) + Φ(d/2 − o)] の o = 0 の場合に一致する．論文の Eq. (4) はその分布によらない版 1/(1 + d_M²/4) である．

**(d) 多タスクの直交性（Proposition 4.2 = Proposition C.3，§4.3）．**独立で均衡な 2 つのラベルづけの判定軸 u^(1), u^(2) について |u^(1)T u^(2)| ≤ min{ (d^(1)/d^(2)) √(2K_2 Ṽ^(1)), (d^(2)/d^(1)) √(2K_1 Ṽ^(2)) }．軸方向のばらつきが両タスクで小さければ，2 つの判定軸はほぼ直交する．因子模型 z = Σ_ℓ (Δ_ℓ/2) t^(ℓ) v_ℓ + η + ξ で，軸に直交する η の分散を大きくしても Ṽ は変わらず古典 CDNV だけが大きくなることを示す（§4.3）．

**(e) 先行の上界との比較（§4.1）．**Luthra et al. (2025b) の上界は a = 16 で (C′ − 1)(8Ṽ_f + (8/√m) V_f^s + [8/√m + 4/m] V_f)．原文："existing guarantees can be loose or vacuous at practical shot sizes (e.g., m ∈ [1, 500])"（§4.2）．

**(f) 実験（§5，App. A–B）．**
- データ：mini-ImageNet（ImageNet-1K の 100 クラス，224×224，train 50000 / val 10000 / test 5000）
- Fig. 2（学習中の CDNV と directional CDNV）：MAE，SimCLR，DINO-v2，VICReg を mini-ImageNet で 1000 エポック自前学習．原文："directional CDNV drops dramatically over training, from roughly 2^{−1}–2^{1} down to about 2^{−3}–2^{−5}. In contrast, standard CDNV decreases only modestly and in some cases even increases transiently"（§5.2）
- Fig. 3（Theorem 4.1 の検証）：公開済みの凍結エンコーダ（視覚のみの SSL は IM-1K，CLIP は 4 億の画像文対，SigLIP は WebLI で事前学習）．MAE，CLIP，I-JEPA，SigLIP（ViT-B/16），SimCLR，VICReg（ResNet-50）の 6 つ．2 クラス（C = 2）をランダムに選び，クラスあたり m 本で重心を推定して NCC のテスト誤差を測る．m の目盛は 1, 5, 10, 20, 50, 100, 200, 500．クラスの選び方と m 本の引き方で平均し，5 回繰り返す．図には NCC 誤差，Luthra [2025] の上界，本論文の上界，m → ∞ の極限（"Lim bound"）の 4 本．各パネルの注記に "ρ, p-val (Train)/(Test)" があり（記号 ρ は pdftotext で欠けたので本記録の推定），値は MAE 0.99/0.99，CLIP 0.98/0.48（Test の p = 0.22），I-JEPA 0.99/0.99，SigLIP 0.99/0.98，SimCLR 0.97/0.88，VICReg 1.00/0.98（Train/Test．p 値は CLIP の Test 以外すべて 0.00）．ρ が何と何の相関か（上界と誤差の m にわたる相関と読める）は本文に定義が無い．原文："for moderate m, our certified error drops below the 0.5 chance threshold, whereas the bound of (Luthra et al., 2025b) remains above 0.5 here and is therefore vacuous"（§5.2）
- Fig. 4：20 のクラス対で，級内共分散を軸方向と直交方向（上位 k = 1, 10, 100 の固有方向と全体）に分解．軸方向の分散は急速に落ち，直交方向は大きいまま
- Fig. 5（多タスク）：色・背景・形・大きさの独立な因子から生成した 64×64 の人工画像で ResNet-18（SimCLR）と ResNet-50（VICReg）を自前学習．異なるラベルづけの判定軸どうしの |cos| の中央値（25–75% 帯）が学習とともに 0 近くへ落ち，Proposition 4.2 の上界（破線）の下にある．数値は本文に無い
- Fig. 6（App. B）：GPT-4o で作った 10 の上位クラスでも directional CDNV の方が大きく落ちる

### 4. 軸 A1〜A11 との対応

- A1 学習器：△．凍結特徴の上の NCC（最近傍重心）が主役．LP は母集団 0–1 誤差の最小化として定義され err_LP ≤ err_NCC で抑えるだけ．ロジスティック回帰・L2 罰則・LDA・最小二乗の解析は無い．実験も NCC のみ
- A2 理論（切片と向き，K と p）：△．誤差を「軸方向のばらつき（m によらない）」＋「重心推定の漏れ（V_ij/m，V_ij²/m）」＋「裾（Θ_ij/m³）」に分ける（Theorem 4.1）．切片と向きには分けない（証明の S・T・Q の分解にその構造があるが，定理では全分散 V_ij にまとめる．本記録の推論）．次元 p は陽に出ず v_c = tr(Σ_c) を通して入る．実効次元は無い
- A3 理論（ソースの価値を標本数に換算）：×．ソースは存在しない．余弦は多タスクの軸どうしの直交性（Proposition 4.2）にだけ出て，標本数との関係は無い
- A4 理論（合併で切片が直る割合，ドメイン別切片）：×．ただし分母の (1 + α_ij)² は分散の不均衡による 1/m 次の系統的な閾値のずれを扱う（App. D.4）．ドメイン由来ではない
- A5 設定（固定の少数ショット）：○．クラスあたり m 本の有限・非漸近の上界（Theorem C.2 は m ≥ 1，Theorem 4.1 は m ≥ 10）．実験は m = 1〜500
- A6 設定（ドメインのずれ）：×．支援とテストは同じ D_c．平均のずれ・共分散のずれ・ラベル比率のずれのいずれも扱わない（ラベル比率は均衡を仮定）
- A7 設定（マルチモーダル）：×．CLIP・SigLIP というマルチモーダル事前学習のエンコーダを使うが，下流は画像の特徴だけで分類する
- A8 手法（弱いラベル・疑似ラベル）：×
- A9 手法（出発点への L2 罰則）：×
- A10 評価（実データ）：○．mini-ImageNet，6 つの公開エンコーダ（Fig. 3）＋ 4 手法の自前学習（Fig. 2, 4）＋人工の多タスクデータ（Fig. 5）．ただし下流は 2 クラスの NCC のみで，多クラスでの上界の検証は無い
- A11 分析（予言と実測の照合）：○．Fig. 3 で上界と NCC 誤差の曲線を m = 1〜500 で重ね，相関 0.88〜1.00（CLIP の Test だけ 0.48）．Fig. 5 で余弦の実測と Proposition 4.2 の上界を重ねる．いずれも上界との照合で，等式の予言ではない

### 5. 穴

扱っていないこと（本記録の整理）：
- ドメインのずれ（ソースとターゲットの違い）．単一ドメインの少数ショットの解析である
- 切片と向きの誤差の分離．証明内の S（軸方向の中点のずれ，分散 (Ṽ_ij + Ṽ_ji)/m）と T（向きの誤差との積，等方なら p に比例）は，定理では全分散 V_ij の項にまとめられ，区別が消える
- 次元 p の陽な依存．v_c = tr(Σ_c) で代用するため，等方の雲では p 倍ゆるい（本記録の計算）
- ロジスティック回帰や罰則つきの線形の頭．LP は err_LP ≤ err_NCC でしか出てこない
- 中心化や正規化などのラベル無しの前処理
- 多クラスの実験（上界は C′ − 1 に比例する和の形だが，実験は C = 2 のみ）
- 上界が実測に対してどれだけ余裕があるか（倍率）の数値．本文には相関と「0.5 を下回る」の記述しかない
- 図の "ρ" の定義

限界として自分で書いていること：本文に Limitations の節は無い．条件として書かれているのは "Theorem 4.1. Let C′ ≥ 2 and m ≥ 10 be integers"，"Assume all samples are i.i.d. within each class and independent across classes"（App. C），"we assume throughout that E[Δ_{i→j}] > 0"（§3），定数 4 について "unless additional tail assumptions are imposed (e.g., subgaussianity)"（§4.2）．因子模型については "an orthogonal-factor structure is a convenient sufficient mechanism ... but it is not required for Proposition 4.2"（§4.3）．

今後の課題：本文に該当する節は無い．Impact Statement の "These results inform the high-level design and evaluation of future self-supervised learning algorithms" が唯一の前向きの記述で，具体的な課題は挙げていない．

### 6. 本研究との差の書き方

関連研究に書く案（日本語）：

「Luthra ら（2026）は，凍結した自己教師あり特徴の上の最近傍重心分類の誤差を，クラスを分ける軸に沿ったクラス内分散（directional CDNV）を主項とし，重心を m 本で推定することによる漏れを全方向の分散 V_ij の 1/m と 1/m² の項として足した非漸近の上界で評価した（Theorem 4.1）．彼らの解析は単一ドメインの少数ショットを対象とし，重心推定の誤差を切片（閾値）と向きに分けず，ソースのデータやドメイン間のずれ，ラベル無しの中心化も扱わない．本研究は同じ「凍結特徴の上の線形則」の枠で，誤差を切片の 1/(2K) と向きの q/K に分けて測り，ドメイン間の切片のずれが中心化で消える割合と，ソースの価値をターゲットのラベル数に換算する式を与える点で異なる．」

本研究の式との対応：
- 本研究の正解率 ½[Φ(d/2 + o) + Φ(d/2 − o)] は，この文献の重心既知の帰着 p_{i→j} = Pr(X ≥ d_ij/2)（Eq. (3)）を正規の雲に特化し，閾値のずれ o を足したものに当たる．文献の Eq. (4)（Cantelli，4Ṽ/(1 + 4Ṽ)）は分布を仮定しない対応物で，Ṽ = 1/d_M² とおくと 1/(1 + d_M²/4)．本研究の式は分布を仮定するぶん値が小さく（Φ(−d_M/2) ≤ 1/(1 + d_M²/4)），o を持つ点で広い
- 本研究の切片 1/(2K)・向き q/K の分け方は，文献の Proposition C.1 の証明の S・T・Q の分解に対応する構造を持つが，文献はこれを全分散で抑えて一つにまとめるので，「書き直した」のではなく「文献がまとめた項を分けて測る」と書くのが正確（本記録の推論）
- 文献の分母 (1 + (v_j − v_i)/(m d_ij²))² は分散の不均衡による閾値のずれで，本研究の o に対応するが，ドメインのずれではなく 1/m で消える系統誤差である

### 7. 注意

- 本記録の推論である箇所：(i) S・T・Q を切片・向きに対応づけた読み（§3(b)，§4 A2，§6）．(ii) 等方の雲で補正項が p Ṽ/m と p² Ṽ²/m の桁になり，真の Var(T) より p 倍ゆるいという計算．(iii) Fig. 3 の注記の記号を ρ（相関係数）と読み，上界と誤差の相関と解釈したこと．本文に定義は無い．(iv) 掲載先が ICML 系の体裁という推測．(v) Ṽ_ij と本研究のマハラノビス距離の対応は Σ が軸方向を固有方向に持つときだけ厳密
- 式の復元：Theorem 4.1 と Proposition C.1 の式は pdftotext で崩れていたが，App. C の証明（λ = 1 で Theorem C.2，コーシー・シュワルツで Theorem 4.1）と整合する形に復元したので形には自信がある．Luthra et al. (2025b) の上界の一般形（a に依存する項）は崩れが大きく，a = 16 の具体形だけを信用した
- Fig. 3 で描かれた「Our bound」が Theorem 4.1 と Theorem C.2 のどちらか，また m < 10 の点をどう描いたかは本文に無い（要確認）
- 未読：参考文献．図の曲線の実際の値（誤差の数値，上界の倍率）は文字情報から読めない
- 要確認の書誌：掲載先，project page の URL，v1 以降の改訂の有無

---

## 余力の分：Le Scao & Rush 2021（NAACL-HLT 2021）

- 書誌（◎ 本文の脚注で確認）：Le Scao, T., Rush, A. M., "How Many Data Points is a Prompt Worth?" Proceedings of NAACL-HLT 2021, pp. 2627–2636 [ACL Anthology](https://aclanthology.org/2021.naacl-main.208)．読んだ範囲：Abstract，§1〜§7（本文 4 頁）．付録 A〜C（プロンプトの例，全課題の曲線）は未読．
- 設定：roberta-large（355M）を，頭つき微調整（head）とプロンプトつき微調整（PET 型，pattern と verbalizer）で同じ条件で比べる．SuperGLUE の 7 課題と MNLI，訓練例 10 本から全量まで指数的に増やし，各点 4 回，計 1892 回の学習．各点で「そこまでで最良の性能」を報告．
- 測り方（§5）："To quantify how many data points the prompt is worth, we first isolate the y-axis band of the lowest- and highest-accuracy where the two curves match in accuracy ... We then take the integral in this region, i.e. area between the linearly-interpolated curves, divided by the height of the band."．この値が average data advantage（訓練例の本数の単位）．
- 主な数値（Table 1，P vs H）：MNLI 3506 ± 536，BoolQ 752 ± 46，CB 90 ± 2，COPA 288 ± 242，MultiRC 384 ± 378，RTE 282 ± 34，WiC −424 ± 74，WSC 281 ± 137．null verbalizer（意味の無い語への対応づけ）でも MNLI で 3355 の得があり，プロンプトの得の多くは pattern 側から来る．
- 軸：A1 ×（全層微調整），A2 ×，A3 △（ソース（プロンプト）の価値を訓練例の本数に換算．曲線の面積による実測で理論は無い），A4〜A9 ×，A10 ○（8 課題，4 回），A11 ×．
- 穴・限界："We assume asymptotically the two curves would match, but are limited by data."（脚注 3）；"In future work, we hope to study the mechanism and training dynamics of the prompting benefits."（§7）．
- 差の書き方：「プロンプトの価値を『同じ正解率に届くのに要る追加の訓練例の数』で測る Le Scao and Rush (2021) は，本研究の『ソースの価値をターゲットのラベルの本数で測る』と同じ単位を使うが，2 本の学習曲線の面積から経験的に測る．本研究は向きの効率 e と実効次元 q から閉じた式で換算する．」
- 注意：Wiesenfarth & Calderazzo 2020（Biometrics 76(1):326–336．DOI 10.1111/biom.13124 を DKFZ の記録で確認）は arXiv 版が見つからず本文未読（○ 書誌のみ）．McLachlan 1992 は未読．

---

## 表：10 本 × 軸 A1〜A11

記号：○ 扱う／△ 一部か暗黙／× 扱わない．各欄の語は根拠の要約（詳細は各記録の 4 節）．Raudys は 1998 の Kybernetika，Okamoto は Shutoh の再掲で判定した．

| 論文 | A1 線形の頭 | A2 切片／向き，K，p | A3 価値の換算，余弦² | A4 合併の切片 | A5 固定の少数ショット | A6 ずれの型 | A7 多モーダル | A8 弱いラベル | A9 出発点の罰則 | A10 実データ | A11 予言と実測 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 Okamoto 1963（Shutoh 経由） | △ 生の特徴の LDF | ○ 1/N の項が Δ/8（閾値）と (p−1)/(2Δ)（向き）に分かれる（解釈は本記録） | × 無し | × | △ p 固定・N 大の展開 | × 同一分布 | × | × | × | × | △ Shutoh のみ |
| 2 Raudys & Jain 1991／Raudys 1998 | △ EDC・Fisher LDF | ○ 向き 2p*/(δ*²N)，p* が実効次元．閾値の項無し | △ 1/T_μ = N/(N+2p*/δ*²) が暗黙 | × | △ N=5 の例，式は p,N 大 | × | × | × | × | × | ○ 式とシミュレーション |
| 3 DARE 2022 | ○ 凍結特徴＋多項ロジスティック | △ 平均の誤差と残差の誤差に分解．K の式無し | × Table 3 で余弦は測る | △ ドメイン平均を予測から消す制約 | × DG／UDA，n ≳ d² | ○ 平均と線形変換 | × | × | × | ○ DomainBed 4 集合 | △ |
| 4 Kini 2021 | ○ 固定特徴の線形（罰則なし SVM 極限） | △ 比例極限で切片 b_δ と向き ρ_δ を分ける | × | × | × 比例極限 | △ 群の平均の混合，ラベル比 | × | × | × 重み減衰の実験のみ | ○ CIFAR，Waterbirds | ○ 人工データ |
| 5 Kumar 2022 LP-FT | ○ 凍結特徴の線形プローブ | × 切片無し，雑音無し，ε に比例 | × | × | × n ≥ m（数千） | ○ 部分空間のずれ | × | × | △ L2-SP の実験 | ○ 10 のずれ | △ 定性的 |
| 6 Ha & Chen 2026 | △ 線形回帰（二乗損失） | △ 誤差 ∝ r/n(0)（直す次元／ラベル数）．切片無し | △ OLS-Tar-MoreData（d/n_new = r/n） | △ DIP の平均合わせ | ○ n(0) ≥ r log(1/δ) | ○ 平均・共分散・係数・反因果重み | × | × | ○ UDA 解からの制約つき微調整 | ○ 実機・NIR（小） | △ |
| 7 De Silva 2023 | △ 1 次元 FLD | △ 閾値のみ：Var = 1/(n+m)，偏り mΔ/(n+m)．向き無し | × | △ 合併で閾値が m/(n+m) 引かれる | ○ n = 4〜500 | ○ 平均の平行移動 | × | × | × | ○ 5 集合 | △ 式とモンテカルロ |
| 8 Hernandez 2021 | × 全層微調整 | × | △ D_T（文字数）を実測 | × | × | △ α で近さ | × | × | × | ○ 大規模 | △ 当てはめ |
| 9 Dobriban & Wager | △ RDA・リッジ（生の特徴） | △ 向き Γ で決まる．切片は均衡で 0，不均衡で偏り | △ Γ² = n/(n+p/α²)（Σ=I，本記録の推論） | × | × 比例極限 | △ ラベル比のみ | × | × | × | × 人工のみ | ○ 人工 |
| 10a Zhang 2026 | △ 転移 LDA | △ 切片の偏り w_KΔ_K と向きの分散 τ_k，α_k² | △ 重み w* ∝ A⁻¹u．換算は無い | △ ターゲットの平均で切片 | × 比例極限 | ○ 平均・共分散・クラス比 | × | × | × | ○ ADHD-200，PPMI | ○ |
| 10b Luthra 2026 | △ 凍結特徴の NCC | △ 軸方向の分散（主項）＋重心の漏れ V/m．切片と向きは分けない | × | × | ○ m ≥ 1〜10 | × 同一分布 | × | × | × | ○ mini-ImageNet，6 エンコーダ | ○ 上界と実測 |
| 余力 Le Scao & Rush 2021 | × | × | △ 訓練例の本数で実測 | × | × | × | × | × | × | ○ | × |

本研究（参考）：A1 ○（凍結特徴＋L2 ロジスティック），A2 ○（1/(2K)，q/K，q 実測），A3 ○（n = q e²/(1−e²)），A4 ○（λ = ρ/(1+ρ)．本記録の対象外），A5 ○（K = 数本〜），A6 ○（コーパス間），A7 ○，A8 ○，A9 ○，A10 ○（7 群），A11 ○．

---

## この群の先行研究に空いている穴

### Takeaway
「切片と向きの分け方」「正解率の 2 つの Φ の式」「余弦² と標本数の関係」「価値をデータ量に換算する発想」はいずれも先例がある．空いているのは，これらを (1) 凍結した深層特徴の上の L2 罰則つきロジスティック回帰で，(2) クラスあたり数本の固定の少数ショットで，(3) ドメインのずれのもとで，(4) 切片と向きを分けて同じ分割で測る，という組み合わせと，向きの実効次元 q の実測である．

### Cited Findings（穴の根拠）
- 切片と向きの分け方の式は古典にあるが，解釈の文は再掲に無い．Okamoto の係数は (p−1) に比例する項と p を含まない項に分かれる（Shutoh の式 (3) [TR10-04](https://www.math.sci.hiroshima-u.ac.jp/stat/TR/TR10/TR10-04.pdf)）．Raudys は向きの項だけを T_μ = 1 + 2p*/(δ*²N) として書き，p* を実効次元と呼ぶ [Kybernetika 1998](https://dml.cz/handle/10338.dmlcz/135232)．Dobriban & Wager は切片が均衡で 0 に収束するとして落とす（補題 7.1 [arXiv 1507.03003](https://arxiv.org/abs/1507.03003)）．Luthra らは証明の中で S（中点のずれ）と T（向きの誤差）を分けるが定理では全分散にまとめる（Proposition C.1 [arXiv 2603.03530](https://arxiv.org/abs/2603.03530)）．De Silva らは閾値だけを扱う（1 次元．付録 A.3 [PMLR 202](https://proceedings.mlr.press/v202/de-silva23a.html)）．
- 固定の少数ショットでドメインのずれを扱う理論は Ha & Chen（回帰，切片無し．Theorem 4.1〜4.6 [arXiv 2507.14661](https://arxiv.org/abs/2507.14661)）と De Silva（1 次元の閾値）だけで，いずれもロジスティック回帰ではない．Ha & Chen は "Extending the finite-sample analysis to logistic, cross-entropy, or other losses is left for future work" と自ら書く．
- 凍結した深層特徴の上の線形の頭を扱う理論（DARE，Kumar，Kini，Luthra）は，ターゲットのラベル K 本で減る誤差の式を持たない．DARE は "simply retraining the last linear layer on this data will suffice" を推測として書くだけ [arXiv 2202.06856](https://arxiv.org/abs/2202.06856)．Kumar の LP の誤差は特徴の誤差 ε に比例する決定論的な項で，ラベル数の項は無い（Lemma A.14 [arXiv 2202.10054](https://arxiv.org/abs/2202.10054)）．Kini の理論は正則化が 0 に向かう極限で，有限の罰則では "If early-stopped (appropriately) before entering TPT, LA-loss can give better balanced performance" と実験で述べるのみ（付録 C.2 [arXiv 2103.01550](https://arxiv.org/abs/2103.01550)）．
- 余弦² = n/(n + p/α²) の形は Dobriban & Wager 系 3.3（"We need to pay a price α/√(α²+γ) for estimating μ±1"）にあり，Raudys の 1/T_μ と同じ関係である．ソースの価値をラベルの本数に換算した文献は無い．換算の発想は Hernandez（D_T，文字数，冪乗則の当てはめ [arXiv 2102.01293](https://arxiv.org/abs/2102.01293)），Le Scao & Rush（average data advantage，面積 [ACL](https://aclanthology.org/2021.naacl-main.208)），Ha & Chen（OLS-Tar-MoreData，d/n_new = r/n(0)）にある．
- 実効次元：Raudys の p* は分布と分類器で決まる量で，罰則の効果は入らない．Dobriban & Wager には "effective" "degrees of freedom" の語が無い（grep）．凍結した深層特徴で q を実測した文献は無い．
- ラベル無しの中心化：DARE は平均の調整を制約で行い，DG ではテスト時の平均の推定を "substantially worse accuracy" として退けた（§3）．JIT-UDA ではラベル無し標本の共分散を使う．「少数ラベルの得のうち中心化で置き換えられる割合」を測った文献は無い．
- 合併の切片（本記録の対象外だが見つけたもの）：De Silva の μ_h = mΔ/(n+m) は，合併で閾値がソース側に m/(n+m) の割合で引かれることを 1 次元の FLD で閉じた形にしており，主張 C（λ = ρ/(1+ρ)）の最も近い先例である．
- 多モーダル・弱いラベル：10 本のどれも扱わない．

### Inferences（論文での書き方）
- 主張 A は「新しい分け方」ではなく「古典の分け方の実測」として書く．根拠として Okamoto（係数の構造），Raudys（向きの項と実効次元），Dobriban & Wager（切片が極限で消える理由），Luthra（少数ショットの上界で切片と向きをまとめている）を並べ，本研究が足すのは (i) L2 罰則つきロジスティック回帰での q の実測（16〜22 ≪ p），(ii) K が数本の有限の体制で 1/(2K) と q/K が実コーパスの組で合うこと，(iii) ドメインのずれによる切片のずれを中心化が消す割合を同じ分割で測ったこと，と書く．
- 正解率の式は De Silva の式 (7)・Dobriban & Wager の式 (16)・Zhang の式 (1) と同じ形なので，「標準の式」として引き，導出を主張しない．
- 主張 B は「Raudys の 1/T_μ（Dobriban & Wager の Γ²）を標本数について逆に解き，ソースの向きの効率 e に当てはめた換算」と書く．換算の発想の先例として Hernandez，Le Scao & Rush，Ha & Chen の OLS-Tar-MoreData を引き，違いを「閉じた式」「ラベルの本数という単位」「q と e という 2 つの実測量」に置く．Zhang の A_kk の構造（ソースのずれ α_k² と τ_k を同じ単位で足す）は同じ考え方を含むので，対比して書く．
- 「少数ラベルが直すのは主に切片」は，Ha & Chen の「直す部分の次元 r で標本複雑度が決まる」の r = 1 の場合として位置づけられる．ただし Ha & Chen は切片を持たない回帰なので，対応は本記録の推論であり，論文では「同じ考え方」と書くにとどめる．
- Kini らの「加法的な補正は効かない」は，罰則無しの分離可能な極限の主張である．本研究が L2 罰則で解のノルムを有限に保つことを，切片の補正が効く条件として明示する．

### Gaps
- Okamoto 1963 の原著と McLachlan 1992 の該当章は未読．閾値と向きへの解釈が原著にあるかは不明．
- Raudys & Jain 1991 の原著は未読（式は Raudys 1998 の再掲）．閾値の項の有無は未確認．
- Zhang 2026 の §2.2（異質な共分散），Ha & Chen の付録，Dobriban & Wager の証明の一部，Kini の証明の本体，Kumar の付録の一部は未読．
- Wiesenfarth & Calderazzo 2020 は本文未入手．
- 掲載先の要確認：DARE（NeurIPS 2022 DistShift workshop．本会議は不明），Kumar（ICLR 2022 は本文に無い），Dobriban & Wager（Ann. Statist. 2018 は本文に無い），Ha & Chen，Zhang，Luthra（いずれも本文に無い）．

## 取得した本文の置き場所
- `/tmp/claude-1000/-home-snics-MER-DA/2f47b0d7-4ef9-4797-9837-84fb3de61717/scratchpad/g4_pdfs/`（PDF と pdftotext の出力．セッション限りの一時領域）．検算の台本は同じ scratchpad の `check_okamoto.py`，下請けへの指示書は `g4_brief.md`．
