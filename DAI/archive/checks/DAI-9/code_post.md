# 計算の後の点検：結果ファイルの欠けと重複（DAI-9 の下調べ，2026-11-08，単位と集計のコミット fcfe634）

対象は `results/dai9_probe/` の全ファイルと，ログ `logs/dai9_probe_launch.log`・`logs/dai9_probe_units.log`・`logs/dai9_probe_sum.log` である．この報告には行の数・欠け・重複・一致の有無だけを書き，UAR の差，規則の可否，S2 の読みの結果などの判定の値は書かない．点検の台本は `DAI/checks/DAI-9/code_run_post/check_post.py`（単位の npz）と `check_sum.py`（集計のファイル）で，集計の作り直しは `DAI/checks/DAI-9/code_run_post/resum/` に書いた．`results/` の下のファイルは読むだけにした．

## 判定

**通す．**予定の 40 単位と 20 個の Q3 の CSV が欠けも重複も無くそろい，保存すると決めた鍵 283 個はすべての単位にあり，集計の 5 ファイルを同じ台本で作り直すとバイト単位で一致したので，再計算も集計のやり直しも要らない．

記録が 2 つある（下の「記録」の節）．どちらも数値を変えない．

## 1．単位の出力（`results/dai9_probe/rows/`）

| 点検 | 予定 | 実際 |
|---|---|---|
| 単位の npz | 40 個（ターゲット 4 × 分割 5 × 種 2） | 40 個．欠け 0，余り 0．ファイル名が同じものは無い |
| Q3 の `_q1.csv` | 20 個（各 (c, f) の r = 0） | 20 個．欠け 0，余り 0 |
| 各単位の評価のターゲット T | c 以外で辞書順の最初 | AFEW は IEMOCAP，ほかの 3 つは AFEW．設計 2 節と一致 |
| 単位の中の commit | 1 つ | 40 単位すべてで fcfe634 |
| provenance の `runs` | 40 | 40．名前の重複 0 |

### 1.1 各単位の鍵の数

台本 `DAI/exp/dai9_probe.py` の `run_unit` が書く鍵を設計 5 節から数え，40 単位のすべてで次のとおり過不足が無かった（予定と違う単位 0）．

| 種類 | 中身 | 予定の数 | 合っていない単位 |
|---|---|---|---|
| テストの判定（int8） | τ 6 ×（LDA_O，EM_truth，EM_rand の 3 ＋ K 4 ×（LDA_K，EM_fromK の t = 10，収束）の 12） | 90 | 0 |
| 候補の割り当て `P_` | τ 6 ×（EM_truth，EM_rand の 2 ＋ K 4 ×（t = 10，収束）の 8） | 60 | 0 |
| 係数 `C_A`・`C_c`・`C_mu`・`C_pi` | τ 6 ×（LDA_O ＋ K 4 の EM_fromK の t = 10）× 4 | 120 | 0 |
| 基底 `B_` | nap10，nap50，pca64，pca256 | 4 | 0 |
| 付帯 | commit，T，test_sample_id，test_y，test_cluster，pool_y，pool_group，pool_dialogue，diag | 9 | 0 |
| 計 | | 283 | 0 |

### 1.2 各単位の中身の形

40 単位のすべてで，次の問題は 0 件だった．

- テストの判定 90 鍵：どれも int8 で，長さはテストの本数と同じで，値は 0〜3 である．
- 割り当て 60 鍵，`pool_group`，`pool_dialogue`：長さは候補の本数と同じで，割り当ての値は 0〜3 である．
- `test_cluster` の長さはテストの本数と同じである．`test_y` と `pool_y` には 4 つの感情がすべて現れる．
- テストの `sample_id` に重複は無い．テストの本数と候補の本数が同じ単位は無い．このため，集計の段が「int8 でテストと同じ長さの鍵」を判定として拾うときに，割り当てや `pool_y` を取り違えることは無い．cells.csv に `P_` か `pool` で始まる鍵は 0 行だった．
- 係数と基底は 4,960 鍵（1 単位あたり 124 鍵）のすべてが有限である．`C_pi` の 1,200 個はどれも和が 1（誤差 1e-4 以内）で，負の値を含まない．`C_A` の形は raw・gc・nap が (2304, 4)，pca64 が (64, 4)，pca256 が (256, 4) である．
- 基底の形：pca64 と pca256 はそれぞれ (2304, 64)，(2304, 256) である．nap10 と nap50 の列の数は，diag に記録した `nap_r_used` と 80 個すべてで一致した．
- diag の鍵は 1 単位あたり 48 個（τ 6 ×（info，LDA_O，EM_truth，EM_rand，K 4 の EM_fromK））で，予定と一致した．
- 入力との照合：各単位のテストの `sample_id` の並びは，入力 `results/dai1/pseudo/units/<T>_<c>_f<f>_r<r>_p1.npz` の `sample_test` と 40 単位すべてで一致した．候補の本数は入力の `sample_pool` の本数と一致した．入力には K 本の `sample_K1`・`sample_K5`・`sample_K10`・`sample_K25` があり，本数はどれも 4K である．
- 同じ (c, f) の r = 0 と r = 1 のテストの並びは，20 組すべてで同じである．このため，種の間で同じテストの発話を突き合わせられる．

## 2．版の一致

- `provenance.json` の `script_sha256` は a57b46f6…（`DAI/exp/dai9_probe.py`），`dai8_sha256` は 199efb1a…（`DAI/exp/dai8_probe.py`）である．どちらも，起動の台本 `DAI/run/dai9_probe.sh` が流す前に確かめた値と，検査したコミット 4154f9a の版のファイルの sha256 と一致した．作業木の今の 2 つのファイルの sha256 も同じである．
- provenance の `commit` は fcfe634 で，`dirty` は偽である．集計の段の `summarize_commit` も fcfe634 で，`summarize_dirty` は偽，`rows_commits` は fcfe634 の 1 つだけである．
- 検査したのは 4154f9a で，流したのは fcfe634 である．`git diff --stat 4154f9a fcfe634` が変えたのは，設計の文書，起動の台本，検査の報告，台帳，先行研究の報告の 6 ファイルだけで，`DAI/exp/` の下のファイル（`dai9_probe.py`，`dai8_probe.py`，読み込む `dai1.py`・`dai1_stats.py`・`dai_data.py`・`dai7_probe_v2.py`）は変わっていない．起動の台本の差分は，sha256 を確かめる 3 行だけである（gate-code の第 2 回の条件への対応）．
- 入力の 40 個の `_p1.npz` の sha256 は，provenance の `inputs_sha256` と 40 個すべてで一致した．
- 定数（α，τ の並び，t = 10，最大 100 回，1e-4，0.5，−2，0.9，0.2，2，3，4，1e-10，0.05）は provenance に書かれ，設計 3・4 節の値と一致した．

## 3．集計のファイル

| ファイル | 予定の行 | 実際の行 | 重複 | NaN と範囲 |
|---|---|---|---|---|
| `cells.csv` | 4 コーパス × 判定の鍵 90 ＝ 360 | 360（各コーパス 90） | (c, key) の重複 0 | NaN 0．各行の `count` はすべて 10．UAR は百分率で，0〜100 の外は 0 行 |
| `diffs.csv` | 4 コーパス ×（τ 6 ×（3 ＋ K 4 × 4））＝ 456 | 456（各コーパス 114） | (c, compare, K) の重複 0．予定した比べの欠け 0，余り 0 | `effect`・`lo`・`hi`・`se`・`p` に NaN 0，Inf 0．`lo > hi` 0 行，点推定が区間の外 0 行．`n_folds` はすべて 5，`n_units` はすべて 10 |
| `diag_units.csv` | 40 単位 × 48 ＝ 1,920 | 1,920 | (c, fold, seed, key) の重複 0 | 鍵の種類ごとに要る列の NaN は 0（下） |
| `q1_units.csv` | 20 × τ 6 ＝ 120 | 120（各コーパス 30） | (c, fold, tau) の重複 0 | NaN 0，Inf 0．commit は fcfe634 の 1 つ |
| `criteria.csv` | 4 行 | 4 行 × 133 列 | c の重複 0 | 下 |

### 3.1 diffs.csv の NaN（どれも作りから来る）

- `df` の 114 個は MOSEI の行である．MOSEI は塊の引き直しで区間を作るので，自由度を持たない．
- `boot_lo`・`boot_hi` の 342 個は AFEW・IEMOCAP・MELD の行である．この 3 つは jackknife なので，引き直しの分位を持たない．
- `t` の 132 個は，MOSEI の 114 行と，ほかの 3 コーパスの 18 行である．18 行はどれも `identical` が真の行である．`identical` が真の行は各コーパス 6 行で，raw を raw と比べる 6 つ（`raw_ldaO_vs_raw`，`raw_emtruth_vs_raw`，K 4 の `raw_ldaK_ldaKraw`）だけである．
- `folds_dropped` の 228 個の NaN は MOSEI と MELD の行である．MOSEI はこの列を持たない．MELD は外した分割が無いので空の文字列になり，pandas が NaN と読んだ．IEMOCAP は 114 行すべてで 5 つの分割を外した．これは「IEMOCAP は塊 ＝ 分割」という `dai1_stats.effect_stats` の作りのとおりで，設計 4 節の「IEMOCAP の区間は 5 つのセッションの揺れしか表さない」に対応する．AFEW は 114 行すべてで分割 0 を外した．これは設計 3.1 節の記録 5（AFEW の分割 0 では 1 つの塊がテストのほとんどを占める）と合う．

### 3.2 diag_units.csv の要る列

| 鍵の種類 | 行 | 要る列 | NaN |
|---|---|---|---|
| LDA_O | 240 | pool_uar | 0 |
| EM_truth・EM_rand | 480 | pool_uar，keep_init，n_iter_done，hist | 0 |
| EM_fromK | 960 | pool_uar，pool_uar_init，moved_at10，hist_at10，n_iter_done，hist | 0 |
| nap の info | 80 | nap_r_used，dim，n_over_d，eig_top，mod_share | 0 |
| pca の info | 80 | dim，n_over_d，mod_share | 0．dim は 80 行すべてで p と一致 |
| gc の info | 40 | dim，n_over_d，small_group_share_pool，small_group_share_test | 0 |
| raw の info | 40 | dim，n_over_d | 0 |

`hist` が空の行は 0 である．ほかの列の NaN は，その鍵の種類が持たない列（例えば LDA_O の行の `keep_init`）だけに出る．

### 3.3 criteria.csv の規則の列

τ ごとに，設計 4 節の規則が要る列を数え，すべてあり，NaN は無かった（例外は下の HYP_taus）．

| τ | 要る列 | 無い列 | NaN のある列 |
|---|---|---|---|
| raw，pca64，pca256 | 各 18（S0 の 4，S1 の 5，S2 の 4，S4 の 4，S5 の比） | 0 | 0 |
| gc | 24（上の 18 ＋ HYP の 3，S5 の LDA_O の差と「感情の違いも消した」，`blocked_by_emotion_loss`） | 0 | 0 |
| nap10，nap50 | 各 25（gc と同じ 24 ＋ `r_used_min`） | 0 | 0 |

- S5 の比の列（6 τ × 4 コーパス）が NaN になるかどうかを，q1_units.csv から n2 の条件（raw の top3_emo が帰無の最大を超えた分割が 3 つ以上）で作り直し，24 セルすべてで criteria.csv と一致した．
- `HYP_n_taus`，`HYP`，`HYP_n_corpora` は 4 行とも埋まっている．`HYP_taus` は満たした τ の名前を並べる文字列なので，空の文字列のときは pandas の既定の読み方で NaN になる（第 2 回の試走で記録済み）．`HYP_taus` が空であることと `HYP_n_taus` が 0 であることの対応，および `HYP` と `HYP_n_taus > 0` の対応は，4 行すべてで一致した．
- `HYP_nap_same` は，nap10 と nap50 の両方が予想の 3 条件を満たしたときにだけ作られる列なので，有無を欠けとして数えない．
- S3 は記述で，criteria.csv ではなく diffs.csv の `*_emK_ldaKraw` と `*_ldaK_ldaKraw` の行（各コーパス τ 6 × K 4 × 2 ＝ 48 行）に入っている．欠けは無い．
- criteria.csv に Inf は無い．

### 3.4 集計の作り直し

`rows/` を指す記号リンクを `DAI/checks/DAI-9/code_run_post/resum/` に置き，同じ台本の `--summarize` を CPU で流し直した（40 秒）．cells.csv，diffs.csv，criteria.csv，diag_units.csv，q1_units.csv の 5 ファイルは，`results/dai9_probe/` のものとバイト単位で一致した．引き直しの種（0）を固定しているので，区間も同じになる．集計の段は `rows/` に何も書かなかった．

## 4．ログ

- `logs/dai9_probe_launch.log`：DAI-8 の単位の段を 05:00:36 から待ち，06:55:32 に始め，07:33:57 に単位の段を，07:34:38 に集計の段を終えた．sha256 の確かめで止まっていない．`logs/dai9_probe.done` があり，`logs/dai9_probe.failed` は無い．
- `logs/dai9_probe_units.log`：開始の行（コミット fcfe634，単位 40 個），40 単位の所要の行，「40 単位を終えた」だけである．例外，収束しない警告，NaN の警告，メモリ不足からのやり直しは無い．
- `logs/dai9_probe_sum.log`：4 コーパスの区間の行と，書いたファイルの行（diffs.csv 456 行）だけである．例外と警告は無い．台本の止める条件（40 単位，コミット 1 つ，S2 の段 1 の量が有限，Q3 が 120 行）は，どれにも当たらずに通った．

## 5．後で求められそうな解析と，保存した出力から出せるか

(a) は保存した出力（と，sha256 を記録した入力と発話一覧）から学び直しをせずに CPU で出せるもの，(b) は計算を足せば出せるもの，(c) は範囲の外とするのが妥当なものである．

| 予想される求め | (a)／(b)／(c) | 出し方か，足すもの |
|---|---|---|
| 話者・塊を単位にした区間の作り直し | (a) | npz の `test_y`・`test_cluster` と判定の 90 鍵から `dai1_stats.counts` で塊ごとの数を作る．IEMOCAP の話者（Ses0x_F／M）は `test_sample_id` を `data/metadata/iemocap_utterance_manifest.csv` の `speaker_id` に突き合わせて作る |
| 性別ごとの UAR | (a) は AFEW と IEMOCAP，(c) は MELD と MOSEI | AFEW は発話一覧の `gender`（欠け 0），IEMOCAP は `speaker_id` の末尾で分ける．MELD と MOSEI の発話一覧には性別の列が無い |
| ターゲットごと・分割ごと・種ごとの内訳 | (a) | `diag_units.csv` と npz の判定から単位ごとに出る．cells.csv は (c, key) の平均だけなので，単位ごとの UAR は npz から作る |
| 感情ごとの再現率 | (a) | 判定と `test_y` から出る．設計 4 節で感情ごとの主張はしないと決めてある |
| EM の割り当てがどのまとまりへ流れたか（DAI-8 の R10 と同じ形） | (a) | `P_` の 60 鍵，`pool_y`，`pool_group`，`pool_dialogue` から出る．候補の `sample_id` は入力の `_p1.npz` の `sample_pool` と同じ並びで（本数の一致を確かめた），K 本の位置は `sample_K<K>` から引ける |
| EM の回ごとの変化，収束までの回の数，t = 10 で動いた割合の K ごとの内訳 | (a) | `diag_units.csv` の `hist`・`n_iter_done`・`moved_at10`（K ごとの行） |
| pca の S5（LDA_O^τ − LDA_O^raw）と「τ は感情の違いも消した」 | (a) | criteria.csv は gc と nap だけに作るが，diffs.csv の `pca64_ldaO_vs_raw`・`pca256_ldaO_vs_raw` の行（K = 0）にある．−2 との比べはこの行の `effect` でできる（下の記録 2） |
| nap10 と nap50 が同じ操作になった単位の数 | (a) | `diag_units.csv` の `nap*_info` の `nap_r_used` |
| 消した向きの固有値とモダリティごとの割合 | (a) | `diag_units.csv` の `eig_top`・`mod_share` |
| 判定の作り直しと，LDA_O・EM_fromK（t = 10）の判別の値 | (a) と一部 (b) | 係数と基底は保存してある．raw・nap はテストの特徴に基底を当てれば値を作れる．pca は成分ごとの標準偏差，gc はテストのグループの平均を保存していないので，特徴（`dai1.Unit`）から作り直す．どれも閉じた式で，CPU で 1 単位数秒である |
| テストの発話ごとの確率（較正，確信度で絞る解析） | (b) | 判定は argmax の int8 だけである．LDA_O と EM_fromK（t = 10）は上の作り直しで出る．EM_fromK の収束，EM_truth，EM_rand，LDA_K の確率は係数を保存していないので，その単位を流し直す（本計算は 40 単位で 38 分，ディスクは今 97 MiB） |
| t = 10 と収束以外の回の数での判定 | (b) | 回ごとの判定は保存していない．`snaps` を足して流し直す（同じく 38 分程度） |
| α，MIN_GROUP，NAP の階数の閾の感度 | (b) | 1 通りずつしか計算していない．下調べの規則は計算の前に固定したので，主張をこの設定に限れば足りる |
| テストの話者の印を使わない gc | (b) | 計算していない．設計 4 節の向きの注に「gc の値はテストの話者が分かる場面の値」と書いてあるので，主張がこの範囲に限られていれば足りる |
| 同じ本数のでたらめな向きを消す nap の対照 | (c) | 設計 8 節に，置かないことと，nap の得をグループの向きによると書くときにこの対照が無いと添えることが書いてある |
| 単独のモダリティの空間，頭の学び直し | (c) | 設計 8 節で範囲の外とした |

(b) の項目は，どれも下調べの結論の文を設計で決めた範囲（AVT，α = 0.5，t = 10，gc はテストの話者が分かる場面）に限れば要らない．結論を手法の固定に使うときは，改めて設計して関門を通すと設計 8 節に書いてある．

## 記録（数値を変えない）

1. **流したコミットと台帳の書き方**：検査したのは 4154f9a で，流したのは fcfe634 である．`DAI/exp/` の下は 2 つのコミットで同じで，台本の sha256 も一致したので，検査した版で流れた．台帳 `docs/seen-data.md` の起動の行は「台本 4154f9a」と書いている．結果を文書に書くときは，結果ファイルに書かれた fcfe634 と台本の sha256 の両方を書く．
2. **pca の S5 の列が criteria.csv に無い**：設計 4 節の S5 は「LDA_O^τ − LDA_O^raw を並べる．点推定が −2 ポイント以下のコーパスでは『τ は感情の違いも消した』と書く」で，τ を gc と nap に限っていない．台本の `summarize` は `S5_<τ>_ldaO_vs_raw` と `S5_<τ>_removed_emotion` を gc，nap10，nap50 についてだけ criteria.csv に書く（`for tau in ("gc", "nap10", "nap50"):` の中）．pca64 と pca256 の値は diffs.csv の `pca64_ldaO_vs_raw`・`pca256_ldaO_vs_raw` の行にあるので，結果の表を作るときに diffs.csv から読む．再計算は要らない．

## 再計算が要るか

要らない．単位の出力は 40 個そろい，鍵の欠けも重複も無い．集計の 5 ファイルは作り直すとバイト単位で一致するので，集計のやり直しも要らない．記録 2 の pca の S5 は，diffs.csv の既存の行から読めばよい．
