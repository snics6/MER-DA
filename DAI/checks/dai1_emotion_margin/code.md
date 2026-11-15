# 関門 1：実装と出力（計算の後の解析 dai1_emotion_margin，2026-11-16，コミット ded3c45＋未追跡の台本）

## 判定
判定：直してから通す．数値の作り（先生と自分の候補の上の再現率，名前と保存の鍵の対応，主の先生の積，感情ごとの差と区間）は 4 コーパスの全単位で照合 1〜5 と独立の検算に合ったが，問い (b) の対象の「UAR では先生が当たるのに下がった 36 点」のうち 24 点（ソースの識別器 src_* が 23 点，start_AVT が 1 点）を，作り直せるのに先生の集合から外しているので，先生の集合を広げ，候補の上の取り違えの表と単位ごとの値を足してから本番を流す．

対象：`tools/dai1_emotion_margin.py`（未追跡．sha256 09eaf70b33d6b2f9f2600f8ed96b1a7cf6bb68f2a0090822c273fad17939a627．HEAD は ded3c45．検査の途中で HEAD が 25ab740 から ded3c45 に進んだが，exp・measure・tools の部品は変わっていない）．入力：`results/dai1/eval/units/*`（単位のコミット ba98974），`results/dai14/units/*.npz`，`results/dai22/units/*.npz`，`results/dai1/stats/desc/teacher_types.csv`・`recall.csv`・`outside_families.csv`，`results/dai1/stats/families_flagged.csv`・`teacher_margin.csv`，`results/dai1/select/frozen.json`．
試しでは行の数・照合の差の最大・所要・エラーだけを見た．margin_e・gain_e・相関・数えの値は画面に出していない．

## A．設計と台本の対応表

設計の正本は台本の冒頭の説明（`:1-38`）と依頼の文である（`docs/exploratory-analyses.md` の節はまだ無い）．

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 候補の上の K 本を除いた印 | `:137-140`（`~isin(sample_pool, sample_K{K})`，数が `len(pool) − 4K` であることを確かめる） | 合う．K = 5・25 は実験 14・22 の `nk__K{K}` と全単位一致（`:151`）．`exp/dai1.py:449` の `setdiff1d(arange(n), kpos[K])` と同じ集合 |
| 先生の名前 → 保存の鍵 | `:84-91`（single_m → none_m，pair_XY → none_XY，fused_AVT → none_AVT，ほかはそのまま），`:71-81` | 合う．`exp/dai1.py:474-478` の tts の作りと同じ．流した 4 コーパスで照合 1 の差の最大 2.8e-14（528 組すべて） |
| poe_none＝none の A・V・T の確率の対数の和 | `:79` | 合う（`dai1.poe` は確率を 1e-12 で下から切ってから対数を取る．台本は切らない．今回の単位では判定が全部一致し，照合 1 の差は 0 の水準．記録 [I-3]） |
| 主の先生＝ret_A・ret_V・ret_T の確率の積（指数 (1,1,1)） | `:80`，`:166`（`_teacher_stats` の teacher = one） | 合う．固定のファイルで 4 コーパス × K = 5・10・25 の main はすべて one．`dai1.product(tk,(1,1,1))` と判定が全単位一致 |
| 主の先生の gain＝族 1（prop − none） | `:187`，`:194`（K = 5・25 は families_flagged，K = 10 は outside_families） | 合う．固定のファイルで 4 × 4 × 3 = 48 条件の prop は全部 joint，teacher = one，ρ = 1，a = 1 で，tt の行と同じ形の問 |
| 自分＝none[K]（固定の引き戻しの強さの W・b） | `:122`，`:141`（`dai3.Saved.none` → `dai1.none_wb`） | 合う．テストの判定が集計の NONE の行の予測と全 600 単位×組で一致（`none_test_equal_units`），K = 5・25 の候補の上の判定が実験 14・22 と 0 発話違い |
| 再現率 | `:94-99` | 合う．sklearn の `recall_score` と差の最大 1.4e-14（検算 (3)） |
| テストの上の感情ごとの差と区間 | `:189-192`（`emo_of`，`ST.effect_stats`，`_fmt(scale=4)`） | 合う．`exp/dai1_desc.py:811-814`（t_recall の族の行）と同じ部品．保存した予測から独立に（分割の中で種を平均し分割を平均）作り直した gain_e と差の最大 3.6e-15（AFEW，36 点．検算 (2)） |
| 区間の乱数の列 | `:189`（`ctx.rng("teacher_types")`） | 説明のとおり．MOSEI の区間は recall.csv と別の列なので lo・hi が最大 0.215 点違う（照合 5 は MOSEI の lo・hi を外している．`:210`） |
| no_own_modality | `:221` | 合う．teacher_types.csv の列と 120 組で食い違い 0（AFEW．検算 (4)） |
| 除く先生（noret_sel，start_AVT，src_*，AVT の fused_AVT） | `:68`，`:130` | **説明の理由と合わない**．start_AVT と src_* は `dai1.Unit` の `start`・`src`（`exp/dai1.py:179-182`）から作り直せる．noret_sel も `dai1.product(tk, wsel[K], 0, src="none")` で作り直せる（指摘 [L4]） |
| K = 5・10・25，単位 5 分割 × 10 種 | `:63`，`:107` | 合う．全行 n_units = 50 |
| 要約の数え | `:237-263` | 説明のとおりに動く．ただし [C]・[D] |
| 出力の上書きをしない | `:273-275` | 合う |

組の数：生徒 A・V・T は先生 10（single 2，pair 3，fused，poe，ret 3）＋主の先生 1＝11，AVT は 10（single 3，pair 3，poe，ret 3）＋1＝11．4 コーパス × 4 生徒 × 3 K × 11 × 4 感情＝2112 行．出力も 2112 行で重複 0．teacher_types.csv の B = none の先生（A・V・T は 21，AVT は 25）に比べ，src_*（A・V・T は 9，AVT は 12），start_AVT，noret_sel を計算しない．その理由は「作り直せるもの」と書かれているが，事実と合わない（指摘 [L4]）．

## A．小さく流した結果

- コマンド：`CUDA_VISIBLE_DEVICES="" ./.venv/bin/python tools/dai1_emotion_margin.py --out-dir checks/dai1_emotion_margin/code_run/afew --targets AFEW`（56 秒）と `--out-dir checks/dai1_emotion_margin/code_run/rest --targets IEMOCAP MOSEI MELD`（9 分 10 秒．MOSEI の塊の重みの引き直しが大半）．GPU は使っていない
- 出力：`checks/dai1_emotion_margin/code_run/{afew,rest}/emotion_margin*.csv`．ログ `code_run/afew.log`・`rest.log` に警告・例外なし
- 照合（4 コーパスとも．`emotion_margin_check.csv`）：
  - 照合 1：先生の候補の上の UAR と uar_pool の差の最大 2.8e-14，1e-6 を超えた単位 0（528 組）
  - 照合 2：自分の候補の上の判定の実験 14・22 との食い違い 0 発話（32 組 × 50 単位），テストの判定と NONE の行の一致 50/50（48 組）
  - 照合 3：4 感情の gain_e の平均と UAR の差の差の最大 7.1e-15（528 組）
  - 照合 4：margin_e の平均と teacher_margin.csv の差の最大 7.1e-15（352 組．K = 5・25 の全組が 1 行に当たった）
  - 照合 5：主の先生の gain_e と recall.csv の family1 の差の最大 1.8e-15，AFEW・IEMOCAP・MELD の lo・hi の差の最大 3.6e-15
  - 集計の予測の照合（`Unit.pred`）：none 600，tt 6000，joint 600 を照合し，欠けと食い違い 0
- 退化の検算（`code_run/verify.py`，AFEW）：
  - (1) 学び直しの行を none にした（A と B を同じ問にした）とき，4 感情の差の最大 0，区間の幅 0
  - (5) teacher_key("fused_AVT")（自分と同じ none_AVT）を先生にしたときの margin_e の最大 0（2 単位 × K = 5・25）
  - (2)(3)(4) は上の対応表のとおり
- 出力の点検：NaN 0，Inf 0，再現率は全部 0〜100，gain_e は全行で lo ≤ effect ≤ hi，同じ (T, 生徒, K) の中で 2 つの先生が同じ値になる組 0，margin_e と gain_e の idxmax の同点 0，要約に NaN 0

照合 1〜5 の意味：照合 1・4 は「先生の判定を作り直す」部分（鍵の対応，積，K 本の除き方）を，GPU で計算した保存の値に照らす．照合 2 は自分の判定と K 本の印を，別の実験の保存に照らす．照合 3 は感情ごとの差の部品（emo_of と 4 倍）と，学び直しの行の指定（tt の名前，ρ = 1，a = 1）を UAR の差に照らす．照合 5 は区間の作りまで recall.csv に照らす．どれも意味のある確かめである．弱い点は 2 つある．照合 3 は，テストに感情が欠ける単位があっても通る（欠けると各感情が 4/3 倍になるが平均は合う）．今回は `exp/dai1_desc.py:788` の assert が評価の回の集計で全単位を通っているので起きていない．照合 4 は teacher_margin.csv に行が無いと黙って飛ばす（今回は全組が当たった）．記録 [I]．

## A．本番の規模の見積もり

4 コーパスで CPU 約 10 分（AFEW 1 分，IEMOCAP・MOSEI・MELD 9 分）．GPU は使わない（`CUDA_VISIBLE_DEVICES=""`．GPU で流すなら作り直しの部分だけが速くなり，MOSEI の区間は numpy なので変わらない）．指摘 [L4] で先生を 21〜25 に広げると，区間の計算が約 2 倍になり約 20 分．

## B．予想される指摘と，出力から答えられるか

| 表・図 | 予想される指摘 | もとにした項目か例 | (a)／(b)／(c) | (b) なら足すもの | 足す費用 |
|---|---|---|---|---|---|
| 問い (b) の数え | 「UAR では先生が当たるのに下がった 36 点」の全部で感情ごとに見たか | 台本の冒頭 `:5-7`，9.3 節 L4 | (b) | src_*・start_AVT・noret_sel を先生に足す（指摘 [L4]）．今の集合は 36 点のうち 12 点しか含まない | 約 10 分増 |
| 点 × 感情の表 | 感情ごとの取り違えの向き（先生がどの感情をどの感情と取り違えるか，生徒の誤りがどの感情に流れたか） | 依頼の例，review-affect の感情ごとの要求 | 先生の候補の上は (b)．学び直しのテストの上は単位の p3 npz の予測から CPU で出せる（主の先生の行は `results/dai1/stats/desc/` の confusion の表にある） | 単位の平均の 4 × 4 の取り違えの表を，先生の候補の上，自分の候補の上，学び直しと none のテストの上で書く（長い表．T, 生徒, K, 先生, 真, 予測, 数） | 約 3 万行，2 MB，数十秒 |
| 点 × 感情の表 | 再現率でなく適合率（疑似ラベル e のうち本当に e の割合）や，偽の e の量で説明できるか | 9.3 節 D（量の定義） | (b) | 上の取り違えの表から出せる．感情ごとの候補の本数も残す | 上に含む |
| 点 × 感情の表 | 疑似ラベルは柔らかい確率で，感情ごとの総量をそろえて使う（`exp/dai1.py:483` の soft_rows）．判定の再現率で測ってよいか | 9.3 節 D | (b) | 真の感情の確率の平均を感情ごとに足す（`teacher_softrec_pool_e`）．作り直した対数確率から出せる | なし |
| 点 × 感情の表 | margin_e の区間（単位の揺れ）．点の平均どうしでなく単位ごとの関係 | 前の段の検査の [R]，tac-round1-status | (b) | 単位ごとの長い表（T, f, r, 生徒, K, 先生, 感情, t_rec, s_rec, 候補の本数, テストの再現率の差）を同時に書く | 約 21 万行（[L4] の後），15 MB，数秒 |
| 点 × 感情の表 | margin_e と gain_e の関係は，自分の候補の上の再現率と自分のテストの上の再現率が同じ識別器で連動することから出ていないか（自分が弱い感情ほど margin_e も gain_e も大きくなる） | 9.3 節 C（別の説明） | (a) | `teacher_rec_pool` と `self_rec_pool` が別の列で残る．gain_e を `teacher_rec_pool` の点内の偏差と `−self_rec_pool` の点内の偏差のそれぞれと順位相関で比べる | なし |
| 要約 | コーパスをまたいだ順位相関は，コーパスの水準の違いで出る．コーパスの中で見よ | 前の段の検査の [C]，9.3 節 C | (a) | `emotion_margin.csv` を (T, 生徒, K) で分けて同じ数えを出せる．ただし要約の台本に入れるべき（指摘 [C]） | 数秒 |
| 要約 | top_emotion_same は偶然でいくつになるか | 9.3 節 S | (a) | 4 感情なので偶然の一致は 1/4．同点が無いことを確かめた（0 組） | なし |
| 主の先生の表 | 主の先生の感情ごとの差の表 | 依頼の例 | (a) | `teacher == "main_q"` の行．gain_e は recall.csv の family1 と 1e-9 で一致（照合 5） | なし |
| K = 10 | K = 10 の UAR の margin | 依頼の例 | (a) | `K == 10` の行の `margin_uar`．teacher_margin.csv には K = 10 が無いが，この表の先生の値は uar_pool と 2.8e-14 で一致した | なし |
| 要約 | hurt の点で負の margin の感情の gain_e の和と正の側の和は，感情の数が違うので比べられない | 9.3 節 D | (a) | `emotion_margin.csv` から行の数と平均を出せる．要約に数を足すのがよい（指摘 [D]） | なし |
| 全体 | 点どうし（同じ先生を 4 生徒が受け取る，K で同じ識別器）は独立でない | 9.3 節 S | (a)（書き方） | 台本の冒頭 `:37` のとおり検定ではない．K をまたいで数えを足さない | なし |
| 全体 | 別のエンコーダ・別のコーパスでも同じか | 9.3 節 L | (c) | 実験 1 の範囲の外．結果の文で 4 コーパス・この特徴に限ると書く | — |

(b) のうち，学び直しを要するものは無い．すべて保存した単位と重みから CPU で 20 分以内に出せるので，「止める」にした行は無い．ただし [L4] は問い (b) の答えの対象を変えるので，結果を見る前に直す．

## 指摘

### [L4] 問い (b) の 36 点のうち 24 点の先生を，作り直せるのに外している
- 重さ：直してから進む
- 該当箇所：`tools/dai1_emotion_margin.py:17,24`（「作り直せるもの」「noret_sel，start_AVT，src_* は除く」），`:68`（TEACHABLE），`:130`
- 問題：冒頭の問い (b) は「先生のほうが UAR で当たるのに gain の区間が 0 を下回った点が 36」を対象にする．teacher_margin.csv（除外の印の無い行）で margin > 0 かつ gain_hi < 0 の 36 点を先生の種類で数えると，src_* が 23，single_ が 7，pair_ が 3，ret_ が 2，start_ が 1 である．この台本は src_* と start_AVT を計算しないので，36 点のうち 12 点しか見ない．src_* と start_AVT の重みは `dai1.Unit` が読み込む（`exp/dai1.py:179-182` の `self.start`・`self.src`）．`dai3.Saved` の `sv.u.start["AVT"]` と `sv.u.src[kind]` から，`exp/dai1.py:479-481` と同じ式で候補の上の判定を作れる．noret_sel も `dai1.product(tk, sv.wsel[K], 0, src="none")` で作れる．「作り直せない」ことは除く理由にならない．要約の hurt_* は，台本の説明が指す 36 点ではない集合の数えになる
- 計算の後で直せるか：直せる（学び直しは要らない．約 10 分の流し直し）．ただし結果を見てから先生の集合を変えると，先生の集合を結果で選んだと読まれる
- 求める対応：pool_logits に `start_AVT`（`sv.u.start["AVT"]`），`src_<c>_<kind>`（`sv.u.src[kind]` の各 (c, W, b)），`noret_sel`（指数 `sv.wsel[K]`．`dai1.product` と同じく確率を 1e-12 で切る）を足し，TEACHABLE を tt の全部（AVT の fused_AVT を除く）にする．teacher_margin.csv と同じく，指数が (1,1,1) の (T, K) では noret_sel に `same_as_poe_none` の印を付けて要約から外す．照合 1・3・4 はそのまま広がる．最後に (T, 生徒, K) ごとの先生の数が 22（A・V・T）と 26（AVT）であることを確かめて止める

### [C] 要約の順位相関 spearman_e がコーパスをまたぐ
- 重さ：直してから進む
- 該当箇所：`tools/dai1_emotion_margin.py:243`（(K, 生徒)，全部，K ごと．コーパスで分けない），`:250`
- 問題：spearman_e は 4 コーパスと点をまたいで行を混ぜるので，コーパスと点の水準（自分の強さ，gain の水準）の違いだけで値が出る．問い (a)「1 つの点の中で」に答えるのは spearman_dev と top_emotion_same であり，spearman_e ではない．前の段の台本は検査の [C] を受けて (T, 生徒, K) の中の数えを足したが，この台本には無い
- 計算の後で直せるか：直せる（`emotion_margin.csv` から数秒）
- 求める対応：要約に (T, 生徒, K) と (T, K) の行を足す．結果の文では問い (a) に spearman_dev と top_emotion_same（偶然で 1/4）を使い，spearman_e は点の間の関係として別に書く

### [R] 単位ごとの値，候補の上の取り違えの表，候補の本数を書き出していない
- 重さ：直してから進む
- 該当箇所：`tools/dai1_emotion_margin.py:171-172,184-185,213`（単位ごとの t_rec・s_rec を平均して捨てる），`recalls`（`:94-99`．取り違えの表と本数を捨てる）
- 問題：margin_e は候補の感情ごとの本数で割る再現率で，本数の少ない感情では単位で大きく揺れる．margin_e の区間，単位ごとの関係，取り違えの向き，適合率，柔らかいラベルの真の感情の確率を後で求められると，台本を書き換えて流し直すことになる（学び直しは要らない）．結果を見てから足すと，足した解析を結果で選んだと読まれる
- 計算の後で直せるか：直せる
- 求める対応：同時に次を書く．(1) `emotion_margin_units.csv`：T, f, r, 生徒, K, 先生, 感情, t_rec, s_rec, 候補の本数, 先生の真の感情の確率の平均．(2) `emotion_margin_confusion.csv`：単位の平均の 4 × 4（先生の候補の上，自分の候補の上）．テストの上の学び直しと none の取り違えも `Unit.pred` から同じ表に足す

### [D] hurt_gain_e_sum_* は行の数の違う和を並べる
- 重さ：記録する
- 該当箇所：`tools/dai1_emotion_margin.py:261-262`
- 問題：負の margin の感情と正の margin の感情の行の数が違うので，2 つの和の大小は行の数で決まりうる．hurt の点の数え（`:247`）は K = 10 を含み，先生の集合も teacher_margin.csv と違うので，冒頭の「36」と同じ数にならない
- 計算の後で直せるか：直せる
- 求める対応：行の数と平均を並べて書く．結果の文で hurt の点の数を冒頭の 36 と並べるときは，K と先生の集合を書く

### [I] 黙って飛ばす分岐と，弱い照合
- 重さ：記録する
- 該当箇所：`:216`（teacher_margin.csv に行が無いと照合 4 を飛ばす），`:189-192`（テストに感情が欠ける単位の確かめが無い），`:79-80`（確率を 1e-12 で切らない）
- 問題：今回は照合 4 が 352 組の全部で当たり，テストの感情の欠けは `exp/dai1_desc.py:788` の assert が評価の回で通り，切りの有無で判定は全単位一致した．台本の側では確かめていない
- 計算の後で直せるか：直せる
- 求める対応：照合 4 は K = 5・25 で当たった組の数が見込みと同じことを確かめて止める．t_recall と同じ assert（全単位のテストに 4 感情がある）を入れる．積と poe は `dai1.product`・`dai1.poe` と同じく 1e-12 で切る

### [I9] 記録の不足
- 重さ：記録する
- 該当箇所：`:292-295`
- 問題：台本は未追跡で，`dai1.git_commit()` の dirty は `exp splits measure features data/metadata` だけを見るので，出力の commit 列（ded3c45）と dirty（偽）からこの台本の版を特定できない．`units_commit="ba98974"` は単位の保存から読まずに書いている．DAI の取り決め（`results/<実験>/provenance.json`）にも書かない
- 計算の後で直せるか：直せる
- 求める対応：本番の前に台本をコミットする．台本と入力の csv・frozen.json の sha256，単位の p3 の commit を provenance か列に残す

## 結果を書くときの注意
- margin_e は K 本を除いた候補の上の判定の再現率の差，gain_e はテストの上の学び直しの再現率の差である．測る発話が違う
- 疑似ラベルは柔らかい確率で，感情ごとの総量をそろえて使う．margin_e は判定（argmax）で測る
- 主の先生の疑似ラベルは none_AVT から学び直した 3 つの識別器の積で，AVT の生徒の主の先生は自分を間接に含む
- MOSEI の gain_e の区間は recall.csv と別の乱数の列で，recall.csv の区間と最大 0.215 点違う

## 目録に無い種類の不備
- 「後の段の解析の対象が，前の段の解析で数えた点の集合と違う」（今回の 36 点と，台本が計算する 12 点）．9.3 節 L4 の下に「前の段の結果の部分集合を問いにする解析は，その部分集合の全部を計算の範囲に含めるか，含めない点の数と理由を書く」を足す案

## 検算の台本
`checks/dai1_emotion_margin/code_run/verify.py`（AFEW の退化，gain_e の独立の作り直し，sklearn の再現率，no_own_modality）．
