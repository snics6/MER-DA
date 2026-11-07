# 関門 1：実装と出力の再検査 3（DAI-6，2026-11-07，コミット b0f858a と 2affd25）

検査したもの：`git diff 62f4e9c b0f858a -- DAI/exp/dai6_probe.py DAI/docs/実験6_下調べ_設計.md`（査読想定の 2 体の第 1 回の指摘を入れた差分．`run_unit` と `summarize` と `--selftest`）と，`git diff b0f858a 2affd25`（再検査の指摘を入れた差分．`summarize`・補助の関数・`--selftest`・設計だけ．`run_unit` は b0f858a と同じことを差分の行番号（714 行以降だけ）で確かめた）．前の再検査は `DAI/checks/DAI-6/code_v3.md`（99e1bd5，判定「通す」）．作業ツリーの `DAI/exp` と `DAI/docs` に未コミットの変更は無い．

## 判定

直してから進む．本番の計算（`run_unit`）はコミット 2affd25（`run_unit` は b0f858a と同じ）で起動してよい．足した量は検算と合い，それまでの出力も誤差 5e−15 以内で再現した．ただし `summarize` の「話者と感情を分けられない」の印は塊の η² の規則 (a) だけを見ていて，IEMOCAP では感情だけで決まる変量がマーカーの規則 (c) のしきい値を越えうる（重相関の上限 0.338 ≥ 0.30）．このため，下の直す点 1 を `summarize` に入れてから集計する．

### 直す点（番号つき）

1. 【直してから進む．`summarize` の前に】`diagnostics.csv` の `inseparable_a`（`DAI/exp/dai6_probe.py:872`）は `emo_eta2_{塊}_min − 偶然 ≥ 0.10` だけで決まる．印の規則 (b)〜(d)（長さ・マーカー・compression_ratio との順位相関の絶対値 ≥ 0.30）は見ていない．`units.csv` の `kind = emotion_only_range` の行で，IEMOCAP 分割 0 の `emo_R_marker` は 0.338 だった．感情だけで決まる変量のうちにマーカーの印が付くものがあるので，IEMOCAP ではマーカーの印が感情の方向も除きうる．
   求める対応：`emo_R_*` のどれかが 0.30 以上の単位にも「感情と分けられない印がある」を付け，どの規則で付いたかを列に残す（例：`inseparable_rule = a / c`）．設計 3.5 の該当の文にも (b)〜(d) の扱いを 1 文足す．`units.csv` に必要な量はすでにあるので，`run_unit` は変えなくてよい．
2. 【記録する】IEMOCAP の筋書きの番号は `_impro(\d+)` で取る（`:298`，`:948`）．このため `impro05`・`impro05a`・`impro05b` は「05」に，`impro08`・`impro08a`・`impro08b` は「08」にまとまり，筋書きは 8 個になる（`DAI/splits/IEMOCAP.csv` の `sample_id` で確かめた）．設計 3.5 に「a・b の版は同じ筋書きとして数える」と書く．
3. 【記録する】印の規則 (a) の帰無の 95% 点は並べ替え 200 回の推定値なので，乱数の使い方が変わると値も変わる．第 3 版と第 4 版では `variate_rows` が乱数を引く順が変わり（残りの帰無を足した），AFEW 分割 0 の `eta2_cluster_null_q95` は最大 0.0101 動いた．印が変わった変量は 74 個のうち 0 個だったが，74 個のうち 11 個は塊の η² が帰無の 95% 点から 0.01 以内にある．結果の文で r_clean を書くときは，この揺れがあることを添える．
4. 【記録する】`summarize` は `de` に列を何度も足すので，pandas の `PerformanceWarning`（DataFrame is highly fragmented）が 5 回出る（`:811-816`）．値には影響しない．`de = de.copy()` を 1 行足せば消える．

## A．差分と設計の対応

| 設計の項目（第 4・5 版） | 台本の箇所 | 合っているか |
|---|---|---|
| 3.4：4 感情を同じ重みで足した感情の間の部分（`between_eq`）とその帰無 | `_parts`（`:385-404`），`decomp_row` の並べ替えの中で同じ関数を呼ぶ | 合う．感情の割合が等しい人工データで `fro_between` 0.415927 と `fro_between_eq` 0.415927 が一致 |
| 3.4.1 A1：「棄却できない／棄却される」と検出の目安 | `:811-812` | 合う |
| 3.4.1 A1′：組・側ごとに上位 3 個と LDA の 3 次元の主角を，単位ごとの参照の分布（100 回，前提が成り立つ世界）の 5% 点と比べる．言葉は組ごとに 3 次元どうし．主成分の値と 3/p を並べる．種と回数を記録 | `:717-755`（`_s3_lda_cos2`・`a1prime`），`:817-837` | 合う．`assumptions_lda.csv` に 2 単位 × 2 組 × 2 側 = 8 行．`n_ref` 100 と `ref_seed` がある．参照は推定した感情の平均と感情の中の共分散から，モダリティを独立にして作る |
| 3.4.1 A2 と同じ重みの版 A2_eq | `:813-814` | 合う |
| 3.5：感情の平均に置き換えた変量（emo_）と残り（res_）の印の量，IEMOCAP の筋書きの η²，感情ごとの平均 | `:321-350`（`_marks`・`variate_rows`） | 合う．AFEW 74 個，IEMOCAP 364 個の変量のすべてに `emo_*`・`res_*` の列がある．IEMOCAP に `eta2_scenario`・`emo_eta2_scenario`・`res_eta2_scenario` がある |
| 3.5：感情だけで決まる変量の塊の η² の最小と最大，長さなどとの重相関の上限 | `:353-383`（`emotion_only_range`） | 合う．一般化固有値の式の範囲が，でたらめの感情の重み 2,000 個の η² の範囲と 4 桁まで一致した（AFEW の塊 [0.1268, 0.2488]，IEMOCAP の塊 [0.0054, 0.0297]，話者 [0.0294, 0.0390]）．各変量の `emo_eta2_*` はすべてこの範囲に入った．各変量の `emo_rho_*` の絶対値はすべて重相関の上限以下だった（例：IEMOCAP のマーカー 0.3323 ≤ 0.3381） |
| 3.5：話者と感情を分けられない印，5 分割のうち 3 つ以上で除いた一致の数 | `:872`，`:877-891` | 一部．規則 (a) だけで判定する（直す点 1） |
| 3.5：偶然の一致の確率，どちらかが 4 つとも同じなら一致は何も言わない | `:698-707`（`_hypergeom_p`），`:878-890` | 合う．`_hypergeom_p(4, 2, 2) = 1/6`，`(4, 1, 1) = 0.25`，`(2, 2, 2) = 0.833` |
| 3.5：`q2_shares.csv` の残りの印と `n_mark_not_res` | `:856-862` | 合う |
| 4 節：ブートストラップの分割の重み = 引かれた回数 ÷ 塊の数 | `:688-689` | 合う．重みがすべて 1 の回が点推定と 63.2557949959 で一致した．本当の抜き直しの平均 63.120 は点推定の近くにある |
| 4 節：`ratio_pca_p`，cv_uar 版，IEMOCAP を除いた数と `proceed_excl_IEMOCAP`，揺れを反対にした場合 | `:771-809` | 合う．`criteria.csv` に 32 列がある．`proceed_excl_IEMOCAP` は IEMOCAP 以外が 3 つそろうときだけ真になりうる |
| 3.5 の記述：融合と単独の生徒の感情ごとの再現率・混同・部分集合（IEMOCAP のマーカーの層，MELD の顔，AFEW の書き起こしの疑い，MOSEI のチャンネル）と全テスト | `:892-940` | 合う．`per_emotion.csv` は 2 コーパス × 4 生徒 × 3 行 × 4 感情 = 96 行．`subsets.csv` に AFEW の `asr_ok`（151/158 本），IEMOCAP の `marker_marker`（356 本）・`marker_nomarker`（375 本）．MOSEI は `mosei_utterance_manifest.csv` の `speaker_source` の値 `youtube_channel`（6,722 本）と `video_id_fallback`（16,134 本）を確かめた |
| IEMOCAP の筋書きの中の 2 感情の対（融合と言葉．保存した W・b と基底から点数を作り直す） | `:942-976` | 合う．作り直した判定が保存の判定と 99.9% 以上一致することを台本の assert が確かめ，通った |
| 計算機と版の記録 | `main` の `provenance` | 合う（`argv`・`host`・`gpu`・`torch`・`cuda`・`numpy`・`pandas`） |
| `--selftest` (7)・(8) | `:1042-1070` | 合う．通った（40 秒）：無関係な組で r ≥ 1 の割合 0.033，A1′ の「離れている」は前提が成り立つとき 10 回中 1，共有が強いとき 10 回中 7 |

## A．小さく流した結果

- 単位の計算（コミット b0f858a．`run_unit` は 2affd25 と同じ）：
  - `MERDA_GPU_FRACTION=0.3 ./.venv/bin/python -W default DAI/exp/dai6_probe.py --out DAI/checks/DAI-6/code_run_v4/out --corpora AFEW --folds 0 --n-perm 1000`：167 秒．
  - 同じ出力先に `--corpora IEMOCAP --folds 0 --n-perm 50`：478 秒．
  - どちらも警告は出なかった（ログ `DAI/checks/DAI-6/code_run_v4/afew_f0.log`・`iemocap_f0.log`）．
- 第 3 版の出力（`DAI/checks/DAI-6/code_run_v2/afew_f0/`，b47568f）との照合（`DAI/checks/DAI-6/code_run_v4/check_v4.py`，ログ `check_v4.log`）：`seqtest`・`subspace_uar`・`anchor_outside`・`k0_proj`・`stability` の共通の数値の列の差は最大 5e−15．`variates` の帰無を使わない列（塊の η²，感情の η²，順位相関）は差 2e−15 以下．`decomp` の値と帰無の 95% 点は差 4e−15 以下．`variates` の `eta2_cluster_null_q95` だけが最大 0.0101 動いた（直す点 3）．
- 集計（コミット 2affd25）：AFEW と IEMOCAP の分割 0 の出力の写し（`DAI/checks/DAI-6/code_run_v4/sum/`）で `--summarize --allow-partial` を流した．13 秒で 9 つの表（`criteria`・`assumptions`・`assumptions_lda`・`q2_shares`・`diagnostics`・`per_emotion`・`confusion`・`subsets`・`scenario_pairs`）を書いた．どの表にも NaN と Inf は無い．`criteria.csv` の `commit` は 2affd25，`unit_commits` は b0f858a で，2 つのコミットが残る（ML の査読 m10 のとおり）．
- 所要の見積もり：A1′ の参照の分布は 2 単位で数秒しかかからず，20 単位でも集計は数分の見込みである．単位の計算では，`variate_rows` が残りの帰無（200 回）を変量ごとに足した．IEMOCAP 分割 0 の 364 個の変量で数十秒の増え方で，前回の見積もり（20 単位 約 28〜30 時間）は変わらない．
- 見た値：IEMOCAP と AFEW の分割 0 の `emotion_only_range` の行（感情の件数を含む），`assumptions_lda.csv` の `below_ref_q05` の 8 個，`subsets.csv` の部分集合の本数，`check_v4.log` の範囲の値．成績の値（UAR）は開いていない．本体が `docs/seen-data.md` に記録する．

## B．予想される指摘と，出力から答えられるか

`code_v2.md` の表に，次の行を足す．

| 表・図 | 予想される指摘 | (a)／(b)／(c) | 根拠 |
|---|---|---|---|
| Q2 の表 | 感情を除いた残りでも話者・長さが方向を占めるか | (a) | `variates.csv` の `res_*`，`q2_shares.csv` |
| Q2 の表 | マーカーと感情を分けられない単位はどれか | (a)（直す点 1 を入れた後） | `units.csv` の `emo_R_marker` |
| 仮定の表 | A1′ の参照の分布の回数を増やしたら，正規でない雑音なら | (b)（回数）／(c)（雑音の形） | `N_REF` は定数（`:717`）．集計だけなので数分で流し直せる．雑音の形は設計 3.4.1 に正規と書いてある |
| 仮定の表 | 感情の割合の偏りを除いたら | (a) | `decomp.csv` の `*_between_eq*`，`assumptions.csv` の `A2_eq` |
| 4 節の判定 | IEMOCAP を除くと進む判定は変わるか | (a) | `criteria.csv` の `n_pass_excl_IEMOCAP`・`proceed_excl_IEMOCAP` |
| 記述 | 筋書きの中の対で S は全部の次元とどれだけ違うか（言葉の生徒も） | (a) | `scenario_pairs.csv` |

## 指摘

### [D1] 「話者と感情を分けられない」の印が塊の規則だけを見る
- 重さ：直してから進む（`summarize` の前に．本番の計算は起動してよい）
- 該当箇所：`DAI/exp/dai6_probe.py:872`，設計 3.5
- 問題：上の直す点 1．`insep = bool(... er.loc[(c, f), f"emo_eta2_{nm}_min"] - er.loc[(c, f), f"emo_eta2_{nm}_chance"] >= 0.10)` は印の規則 (a) だけを使う．IEMOCAP 分割 0 では，感情だけで決まる変量とマーカーの順位相関の上限が 0.338 で，規則 (c) のしきい値 0.30 を越える．
- 計算の後で直せるか：直せる（`units.csv` に量がある）．
- 求める対応：直す点 1 のとおり．

### [記録] 筋書きの番号のまとめ方，帰無の推定の揺れ，pandas の警告
- 重さ：記録する
- 該当箇所：`:298`，`:948`，`:321-350`，`:811-816`
- 問題と対応：直す点 2〜4．

## 当てて指摘の無かった項目

差分について，I1（設計の第 4・5 版と台本），I3（足した量のうち感情のラベルを使うもの（emo_・res_・感情ごとの平均・LDA・A1′）は `summarize` か `variates.csv` の測定だけに使い，部分空間と分類器と選択には入らない），I7（第 3 版の値の再現，`emotion_only_range` の式の検算，`between_eq` の検算，重みがすべて 1 のブートストラップ），I9（A1′ の種と回数，計算機と版の記録），I11，I12（集計に単位のコミットと集計のコミットの両方を残す），S2・S3（ブートストラップの重み），R12（A1′ に参照の分布，A1 に検出の目安）．

## 目録に無い種類の不備

- D20（案）：ラベル無しの印（話者・長さ・マーカーなど）で方向を除く規則を作ったときは，感情だけで決まる変量がそれぞれの印の規則でどこまで印を付けられるか（塊の η² の最小，付帯の量との重相関の上限）を，規則ごとに単位ごとに出し，印が感情の方向も除きうる単位を規則ごとに示す（DAI-6 の gate-code の再検査 3．IEMOCAP ではマーカーとの重相関の上限が 0.338 で，しきい値 0.30 を越えた）．
