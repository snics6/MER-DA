# 関門 1 の再検査：実装と出力（計算の後の解析 dai1_emotion_margin，2026-11-16，コミット e3f225e）

## 判定
判定：通す．前回の指摘 [L4]・[C]・[R]・[D]・[I]・[I9] はすべて台本に入った．4 コーパスを 3 回に分けて流し，照合 1〜6 と追加の検算（足した先生を `exp/dai1.py` の式で作り直す，gain_e を保存した予測から独立に作り直す，hurt の点の集合を teacher_margin.csv に照らす）はすべて合った．

対象：`tools/dai1_emotion_margin.py`（コミット e3f225e，sha256 fe1a383e…7e72，作業木に差分なし）．前回の報告は `checks/dai1_emotion_margin/code.md`．
試しでは行の数・照合の差の最大・真偽・所要・エラーだけを見た．margin_e・gain_e・相関・数えの値は画面に出していない（hurt_points は冒頭の 36 と等しいかの真偽だけを出した）．

## 前回の指摘への対応

| 前回の指摘 | 直した箇所 | 確かめた結果 |
|---|---|---|
| [L4] src_*・start_AVT・noret_sel を外していた | `:94-116`（`source_logprobs`，`pool_logprobs` に noret_sel），`:164`（tt の全部，AVT の fused_AVT だけ除く），`:170`（w[K] が (1,1,1) の K の noret_sel に same_as_poe_none の印），照合 6 `:165-167`，照合 4 の集合と印 `:285-289,308-312` | (T, 生徒, K) ごとの先生の数は A・V・T が 22，AVT が 25（tt の 25 から fused_AVT を除き主の先生を足す）．48 組の全部で teacher_types.csv の行の数と一致．印は AFEW の 3 K × 4 生徒の 12 組だけに付き，K = 5・25 の 8 組で teacher_margin.csv の印と一致．足した先生の候補の上の判定と真の感情の確率は，`dai1.build_tk`・`dai1.poe`・`dai1.product`・`dai1.probs` で作り直した値と差の最大 1.4e-14 と 6.7e-16（AFEW の 2 単位 × K = 5・25．`code_run_v2/verify_v2.py` (1)）．4 コーパスをつないだ要約の "5+25" の行の hurt_points は 36 と等しく，hurt の点の集合も teacher_margin.csv の 36 点と同じ（`code_run_v2/combine.py`） |
| [C] 要約がコーパスをまたぐ | `:329-331`（(T, K, 生徒)，(T, K)，"5+25" の行を足した） | 4 コーパスで要約は 77 行，NaN の欄 0（hurt の平均の欄は行が無いときだけ NaN になる作り） |
| [R] 単位ごとの値・取り違えの表・本数 | `:228-230`（`emotion_margin_units.csv.gz`），`:173,205,223,246,259,318-319`（`emotion_margin_confusion.csv.gz`），`:129-139`（本数，真の感情の確率） | AFEW で単位の表は 54,600 行，重複 0，NaN 0．単位の表の平均は点の表の t_rec・s_rec・本数と差の最大 2.8e-14．取り違えの表の候補の上の行の和は候補の本数の平均と 2.8e-14，テストの上（学び直しと none）の行の和はテストの本数の単位の平均と 7.1e-15 で一致 |
| [D] hurt の和 | `:338,350-351`（行の数と平均） | 説明の `:49` と同じ定義 |
| [I] 黙って飛ばす分岐と弱い照合 | `:155-157`（テストの 4 感情），`:286-287`（照合 4 の行が無いと止まる），`:89-91`（1e-12 で切る），`:294-307`（no_own_modality を teacher_types.csv と照らす），`:182-183`（単位の指数と固定のファイル），`:197-198`（K 本の印と kpos） | 4 コーパスで止まらずに通った |
| [I9] 記録 | `:142-146,391-394`（provenance．台本の sha256，git の状態，単位のコミットを p3 から読む，入力 6 本の sha256，版） | AFEW の provenance：commit e3f225e，script_status clean，units_commit ba98974583…，入力 6 本の sha256 あり |

## A．小さく流した結果

- コマンド（GPU は使わない）：`CUDA_VISIBLE_DEVICES="" ./.venv/bin/python tools/dai1_emotion_margin.py --out-dir checks/dai1_emotion_margin/code_run_v2/<afew|mosei|iem_meld> --targets <AFEW | MOSEI | IEMOCAP MELD>`．所要は AFEW 1 分 14 秒，MOSEI 9 分 19 秒，IEMOCAP と MELD で 6 分 19 秒．ログ `code_run_v2_*.log` に警告・例外なし
- 行：コーパスごとに 1092 行（273 点 × 4 感情），4 コーパスで 4368 行，(T, 生徒, K, 先生, 感情) の重複 0，NaN 0，Inf 0，gain_e は全行で lo ≤ effect ≤ hi，印の無い先生どうしで同じ値になる組 0，idxmax の同点 0
- 照合（4 コーパスの差の最大）：照合 1 は 2.8e-14（1e-6 を超えた単位 0），照合 2 は全単位一致，照合 3 は 7.1e-15，照合 4 は 1.4e-14（182 組 × 4 コーパス），照合 5 は effect が 1.8e-15，lo・hi が AFEW・IEMOCAP・MELD で 3.6e-15（MOSEI は乱数の列が違い最大 0.137 点で，説明の `:17` のとおり照合の外），照合 6 は 48 組の全部で一致，集計の予測の照合（none 600，tt 13,050，joint 600）は欠けと食い違い 0
- 足した先生の gain_e を保存した予測から独立に作り直した値との差の最大 1.1e-14（AFEW の 141 点．`verify_v2.py` (2)）

## A．本番の規模の見積もり

4 コーパスを 1 回で流すと CPU で約 17 分（上の 3 回の和）．GPU は使わない．出力は `results/dai1/stats/` の下に 6 つ（単位の表は gzip で 1 コーパスあたり約 0.7 MB，4 コーパスで約 3 MB）．既にあれば止まる．

## B．予想される指摘と，出力から答えられるか（前回の表からの変化）

| 予想される指摘 | 前回 | 今回 |
|---|---|---|
| 36 点の全部で感情ごとに見たか | (b) | (a)：要約の "5+25" の行．点の集合は teacher_margin.csv と同じ |
| 感情ごとの取り違えの向き，適合率 | (b) | (a)：`emotion_margin_confusion.csv.gz`（who = teacher_pool，self_pool，retrained_test，none_test） |
| 柔らかいラベルの真の感情の確率 | (b) | (a)：`teacher_trueprob_pool`，`self_trueprob_pool` |
| margin_e の区間，単位ごとの関係 | (b) | (a)：`emotion_margin_units.csv.gz`．区間は gain と同じく分割と種の分散から作れる．単位ごとの gain_e は単位の表に無いが，`results/dai1/eval/units/` の予測から CPU で数十秒で出せる（記録 1） |
| コーパスの中の関係 | (a)（要約に無い） | (a)：要約の (T, K, 生徒) の行 |
| K = 10，主の先生の表 | (a) | (a)（変わらず） |

学び直しを要する行は無い．

## 指摘

### [記録 1] 単位の表にテストの上の感情ごとの差が無い
- 重さ：記録する
- 該当箇所：`tools/dai1_emotion_margin.py:228-230`
- 問題：単位ごとの margin_e と gain_e の関係を見るには，単位ごとのテストの上の再現率の差が要る．単位の表は候補の上の量だけを持つ
- 計算の後で直せるか：直せる（学び直しは要らない．`Unit.pred` の予測から出せる）
- 求める対応：要るときに別の台本で出す．本番の前に足すなら，`:258-259` の取り違えの加算と同じ場所で単位ごとの再現率の差を単位の表に足す

### [記録 2] 取り違えの表の count_mean の尺度
- 重さ：記録する
- 該当箇所：`tools/dai1_emotion_margin.py:318`（`M[i, j] / len(units)`）
- 問題：count_mean は 50 単位の和を 50 で割った「1 単位あたりの平均」で，`exp/dai1_desc.py:805` の confusion の表（分割について和，種について平均．5 倍の尺度）と尺度が違う．行で割った割合は同じになる
- 求める対応：結果の文で両方の表を並べるときは，割合で比べるか尺度の違いを書く

### [記録 3] "5+25" の行と 36 点の一致は照合に入っていない
- 重さ：記録する
- 該当箇所：`tools/dai1_emotion_margin.py:49`
- 問題：説明は「"5+25" の行が冒頭の 36 点に当たる」と書くが，台本は確かめない．4 コーパスで流したときに合うことは検算で確かめた．`--targets` で一部だけを流すと合わない
- 求める対応：4 コーパスを流したときだけ，"5+25" の hurt の点の集合が teacher_margin.csv の margin > 0 かつ gain_hi < 0 の点と同じことを照合に足すとよい

## 結果を書くときの注意（前回からの追加）
- noret_sel は AFEW の K = 5・10・25 で poe_none と同じ先生で，要約から外れている．ほかの 3 コーパスでは外れていない
- 要約の (T, K, 生徒) の行は 22 点か 25 点の上の数えで，コーパスの中の比べに使う．"all" の行はコーパスと点の水準をまたぐ

## 検算の台本
`checks/dai1_emotion_margin/code_run_v2/verify_v2.py`（AFEW：足した先生の作り直し，gain_e の独立の作り直し，hurt の数，取り違えの表の行の和），`checks/dai1_emotion_margin/code_run_v2/combine.py`（4 コーパスをつないだ要約と 36 点の集合）．
