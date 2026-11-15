# 関門 1（再検査）：実装と出力（計算の後の解析 dai1_balance_shares，2026-11-16，コミット 045a46d）

## 判定
判定：通す．前回の指摘 5 件（[I-定義]，[R-乱数]，[I-照合]，[記録]，[B-足すもの]）はすべて直っており，4 コーパスを別々に流すと新しい照合 1〜6 がすべて通った．残る点は結果の数値を変えない記録の不足だけである．

検査の対象：`tools/dai1_balance_shares.py`（045a46d．990df58 からの差分は tools/dai1_balance_shares.py だけで，exp/ に差分なし．作業木の差分なし）．値の大小（joint と kd のどちらが高いか）は報告に書かない．

## 前回の指摘への対応

| 前回の指摘 | 直した箇所 | 確かめた結果 |
|---|---|---|
| [I-定義] 答えの割合の定義 | :105-109（全単位の取り違えの数 M を足して M.sum(0)/M.sum()） | 照合 4（:114-122）：none と joint(one,1,1) の答えの割合が emotion_margin_confusion.csv.gz と 4 コーパスで最大 1.1e-16．balance_confusion.csv.gz の列の和から作った割合も balance_shares.csv と 1.1e-16 で合った（MELD で確かめた） |
| [R-乱数] MOSEI の区間 | 説明 :15-16，照合 2（:137-146），照合 3（:147-156） | AFEW・IEMOCAP・MELD：recall.csv family1 の lo・hi と最大 3.6e-15，ablation.csv 行 d の lo・hi と最大 4.4e-16．MOSEI：effect は 4.4e-16 で一致し，lo・hi は family1 と最大 0.172 点，行 d と最大 0.0029 点違った（別の乱数の列．説明どおり） |
| [I-照合] 照合 1 が恒等式 | 照合 1（:96-99．prop == joint(one,1,1)），照合 2〜4 を追加 | 48 条件で通った．恒等式の照合は照合 5 として残り，説明に「作りの確かめ」と書いてある |
| [記録] 単位のコミット | :88-91，provenance :172-175 | 4 コーパスとも単位の commit は ba98974583df…，frozen_sha256 は 6926fb3e…で 1 通り．provenance.json に台本と入力 6 つの sha256，commit 045a46d，dirty = False が入った |
| [B-足すもの] | 取り違えの表 :112-113，se・df・interval :160-161，v ∈ {one, sel} × ρ ∈ RHO_GRID の対 :100-104，:125-129 | 下の「小さく流した結果」 |

## A．設計と台本の対応表（変わった行だけ）

| 設計の項目（台本の冒頭の説明） | 台本の箇所 | 合っているか |
|---|---|---|
| 版 v ∈ {one, sel}，ρ ∈ {1/16, 1/4, 1, 4} の joint(v, ρ, a = 1) と kd(v, T = 1, ρ) | :100-104 | 合う．2 つの違いはどの v・ρ でも重みの合計をそろえるかだけ（前回の A-2 の読みは ρ と v に依らない．exp/dai1.py:402，404-406，448） |
| 対 = (joint, none)，(kd, none)，(joint, kd) を 8 通り | :125-129 | 合う．balance_recall.csv の pair は 24 通り |
| 主の対の印 main_pair | :158 | 合う．各コーパス 144 行（3 対 × 4 分類器 × 3 K × 4 感情） |
| 答えの割合・正解の割合 = 全単位の数を足して割る | :105-109 | 合う |
| 取り違えの表 = 単位あたりの平均の数 | :112-113 | 合う（M / 50 単位） |
| 乱数の列 = "ablation" の番号から (分類器，K，対，感情) の順 | :123，:131-132 | 合う．ctx.rng はコーパスごとに種を作るので，--targets でコーパスを分けて流しても同じ値になる（読んだだけ） |

## A．小さく流した結果

- コマンド：`CUDA_VISIBLE_DEVICES="" ./.venv/bin/python tools/dai1_balance_shares.py --out-dir checks/dai1_balance_shares/code_run_v2/<T> --targets <T>` を AFEW，IEMOCAP，MELD，MOSEI の順に 1 つずつ流した．
- 所要（実時間）と最大のメモリ：AFEW 69 秒，IEMOCAP 17 秒，MELD 372 秒，MOSEI 485 秒．メモリは 0.92〜1.19 GB．GPU は使っていない．落ちず，警告なし．
- 行の数（コーパスごと）：balance_shares.csv 816 行（4 分類器 × 3 K × 17 版 × 4 感情），balance_recall.csv 1152 行（4 × 3 × 24 対 × 4 感情），balance_confusion.csv.gz 3264 行（4 × 3 × 17 × 16），balance_check.csv 363 行．
- NaN・Inf：balance_recall.csv の df が MOSEI の 1152 行で NaN（重みの引き直しの区間には自由度が無い．interval = boot_cl+seed）．ほかは 0．balance_check.csv の NaN は予測の照合の行の student・K・pair の空欄 9 個．答えの割合と正解の割合の 4 感情の和はすべて 1．
- 照合の値の最大（4 コーパス）：照合 4 は 1.1e-16，照合 5 は 6.2e-15，照合 2 の effect は 1.8e-15，照合 3 の effect は 9.7e-17．予測の照合は none 600 件，joint 4800 件，kd 4800 件で欠けと食い違いが 0．
- 退化の確かめ：joint − kd の 384 行（MELD）に，effect = 0 かつ se = 0（2 つの学び直しの予測が全単位で同じ）の行は無かった．

## A．本番の規模の見積もり

4 コーパスを 1 回で流すと，上の所要の和で約 16 分（CPU）．10 分を超えるので tmux の `mer-da` で流し，`CUDA_VISIBLE_DEVICES=""` を付ける（import する dai_solver が CUDA を見ると GPU に文脈を作る）．出力は 5 つのファイルで合わせて数 MB．

## B．予想される指摘と，出力から答えられるか（前回の表から変わった行）

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 答えの割合 | 要点 17 の割合と同じ定義か | 9.3 節の I | (a) | 照合 4 で 1e-9 の一致を確かめた | なし |
| 取り違え | joint・kd・none の取り違えの表 | 依頼の (3) | (a) | balance_confusion.csv.gz | なし |
| 感情ごとの差 | 主の提案 prop で同じ比べをせよ | 依頼の (3) | (a) | 照合 1 で prop = joint(one,1,1) を確かめた | なし |
| 感情ごとの差 | ρ と教師の版を変えても向きは同じか | 9.3 節 R | (a) | main_pair = False の 21 対（1 つの (T, 分類器, K) あたり） | なし |
| 感情ごとの差 | se・自由度・区間の作り | S 項 | (a) | se，df，interval，uar_se の列 | なし |
| 答えの割合 | 答えの割合の差に区間を付けよ | S 項 | (a)（未実装） | 前回と同じ．単位の予測・正解・塊から CPU で作れる．今の出力は割合の点と取り違えの表だけで，答えの割合の区間は無い | CPU 数分 |
| 答えの割合 | 候補の上の学び直した分類器の答え | 依頼の (3) | (a)（未実装） | 前回と同じ．W・b と sample_pool と標準化の作り直しが要る．疑似ラベルの候補の上の判定は emotion_margin_confusion.csv.gz の teacher_pool にある | CPU 数分 |

(b) にあたる行は無い．答えの割合の区間と候補の上の答えは，求められても単位の保存から CPU で出せるので，学び直しにはならない．

## 指摘

### [記録] 照合 2〜4 は比べる値が NaN のときに止まらない
- 重さ：記録する
- 該当箇所：tools/dai1_balance_shares.py:118-121，:141-145，:151-155
- 問題：`if d4 > 1e-9:` などの比べは，d4 が NaN なら偽になり通る．照合 4 で emotion_margin_confusion.csv.gz の行が見つからないと，`Mr.sum()` が 0 になり d4 は NaN になる（読んだだけ）．今回の 4 コーパスの実行では，比べる表の行は条件ごとに 16 行（none_test，retrained_test の main_q）と 4 行（family1）と 1 行（行 d）がそろい，balance_check.csv の照合の値はすべて有限だったので，今の結果は照合を本当に通っている．
- 計算の後で直せるか：直せる（balance_check.csv の値が有限かを見ればよい）．
- 求める対応：`if not d4 <= 1e-9:` の形にするか，`len(x) == 16` を確かめる．直さないなら，本計算の後の点検で balance_check.csv に NaN の値が無いことを確かめる．

### [記録] 説明の「端が最大 0.15 点ほど違う」は，流した値では 0.172 点だった
- 重さ：記録する
- 該当箇所：tools/dai1_balance_shares.py:16
- 問題：MOSEI の joint − none の感情ごとの区間の端と recall.csv family1 の差の最大は 0.172 点，joint − kd の UAR の区間の端と ablation.csv 行 d の差の最大は 0.0029 点だった．
- 計算の後で直せるか：直せる．
- 求める対応：結果の文には balance_check.csv の vs_recall_family1_lohi と vs_ablation_d_lohi の MOSEI の最大を書く．

## 目録に無い種類の不備
- 照合の比べを `差 > 許容` で書くと，NaN の差が照合を通る．9.3 節の I に「照合は `not (差 <= 許容)` で書くか，比べる行の数を先に確かめる」を足す案．

当てた項目：I（指定の対応，部品の引数と戻り値，乱数の列，照合の意味と NaN），R（後から求められる表），S（区間の作り，se・df），L（範囲の外：温度 T ≠ 1，IEMOCAP の台本）．
