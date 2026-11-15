# 感情認識の査読者の再検査 2：DAI-6（2026-11-07，コミット cc07f13．HEAD は 28bafdc）
判定：OK

検査したもの：`git diff b0f858a cc07f13 -- DAI/exp/dai6_probe.py DAI/docs/実験6_下調べ_設計.md`（設計 第 5 版）と，`git diff cc07f13 28bafdc -- DAI/docs/実験6_下調べ_設計.md`（A1′ の行の注だけの変更）．前回の報告は `DAI/checks/DAI-6/review_affect_v2.md`（判定 OK．軽微な指摘 b1〜b6）である．作業ツリーの `DAI/exp` と `DAI/docs` に未コミットの変更は無い．

b1〜b6 はすべて入った．台本の差分は `_lda_basis` より後の新しい関数（`_s3_lda_cos2`・`a1prime`），`summarize`，`selftest` だけで，各単位の計算（`run_unit`）は b0f858a から変わっていない．`--selftest` は通った．

## b1〜b6 の判定

| 前回の指摘 | 入った箇所 | 判定 |
|---|---|---|
| b1 IEMOCAP を除いたときの判定の書き方 | 設計 4 節「4 つの規則で『進む』になり，IEMOCAP を除くと『進まない』になるときは，『進む判定は IEMOCAP の合否に依り，IEMOCAP の合否は筋書きの話題による当たりを分けられない』と書く」．`criteria.csv` の `proceed_excl_IEMOCAP`（IEMOCAP 以外の 3 つがそろい，そのうち 2 つ以上で基準を満たすとき真） | 入った |
| b2 筋書きの中の対に言葉の生徒 | `scenario_pairs.csv` に `student` の列．生徒 AVT と T の 3 行（元の全部の次元，S，同じ次元の主成分）．T の基底は `S_i_T`，同じ次元の主成分は S と同じ次元で作る（`run_unit` の作り方と同じ） | 入った |
| b3 感情ごとの表と部分集合の表の範囲 | `per_emotion.csv`・`confusion.csv` を 4 生徒に広げた．`subsets.csv` に全テストの行（`subset = all`）と 4 生徒の行を足した | 入った |
| b4 印の無い変量の書き方と，印が消える変量の数 | 設計 3.5 節「印の無い変量は，測った 4 種類の印で説明されない方向であり，感情の方向とは限らない（音声と言葉には語の内容が両方に入りうる．Pasad ほか 2024．MELD の対話の共有も測っていない）」．`q2_shares.csv` の `n_mark_not_res` | 入った（下の記録 1） |
| b5 仮定の表の注と同じ重みの版 | 設計 3.4.1 節の注「仮定の表の判定は，候補の感情の割合のもとでの判定である」．A2 の行に同じ重みの版．`assumptions.csv` の `A2_eq`・`sv_between_eq_3`・`null_q95_sv_between_eq_3` | 入った |
| b6 5 節の出力の一覧 | 設計 5 節に `assumptions_lda.csv`，`q2_shares.csv`，`per_emotion.csv`・`confusion.csv`，`subsets.csv`，`scenario_pairs.csv`，`units.csv` の `emotion_only_range` の行 | 入った |

## ML の査読と台本の検査の直しのうち，感情認識に関わる点
- **A1′ の参照を感情の中だけの並べ替えに替えた点**：参照は感情ごとの平均と候補の感情の割合をそのまま残す．したがって，感情の割合の偏り（MOSEI の Angry 7.7% など）で少ない感情の方向が弱くなる分は，観測と参照の両方に同じように入る．前回の A3 で指摘した「割合の偏りだけで『離れている』と読む」危険は，この参照で除かれる．Fisher の判別の部分空間（`_lda_basis`）は 4 感情の平均を同じ重みで使う．問題は無い．
- **「話者と感情を分けられない」の印を規則 (b)〜(d) に広げた点**：感情だけで決まる変量とマーカーの重相関は，IEMOCAP の分割 0 で 0.338 である（前回の再検査で確かめた）．このため IEMOCAP の多くの分割にも規則 (c) の名前でこの印が付き，Q5 の一致の数の「除いた数」は AFEW と MELD だけで数えることになる見込みである．設計 3.5 節は含めた数と除いた数の両方を書くので，報告の誤りにはならない．

## 記録（判定に影響しない）
1. b4 で前回求めた「8 節の限界に測らない共有の要因を書く」は，3.5 節の文として入った．8 節の限界の節には同じ文が無い．論文の限界の節を書くときに，3.5 節のこの文（音声と言葉の語の内容，MELD の対話）を写せばよい．
2. `subsets.csv` の UAR は 5 つのテストを合わせた値で，`subset = all` の行と比べる．4 節の判定の分割の平均とは比べない（前回の b3 のとおり．設計 5 節の書き方で足りる）．

## 確かめたこと
- **`--selftest`**：`MERDA_GPU_FRACTION=0.15 ./.venv/bin/python DAI/exp/dai6_probe.py --selftest` が通った．出力は「無関係な組で r ≥ 1 の割合 0.033，A1′ の『離れている』：前提が成り立つとき 20 回中 1，共有が強いとき 10 回中 10」だった．
- **差分の範囲**：`git diff b0f858a cc07f13 -- DAI/exp/dai6_probe.py` の変更の塊はすべて 714 行より後（`_lda_basis` の後の関数，`summarize`，`selftest`）で，`ProbeUnit`・`variate_rows`・`emotion_only_range`・`run_unit` に変更は無い．
- **本検査が新しく見たデータ**：無い．台本と設計の差分を読み，人工データの `--selftest` を流しただけである．

判定：OK
