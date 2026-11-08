# ML の査読者の再検査：DAI-11 の下調べ（2026-11-08．第 3 回）
判定：OK

第 2 回（`review_ml_v2.md`）の軽微 8 件（m-a〜m-h）は，設計の第 4 版（`DAI/docs/実験11_下調べ_設計.md`）と台本の第 3 版（`DAI/exp/dai11_probe.py`．sha256 `70f6fad3ae4edb8767a88f6c5c32e35e7ecddf601c0aba0cec961aa13ba684c4`）で解けている．新しく入れた部分（塊の並べ替えと Freedman–Lane の部分相関，受け取る分類器ごとの印，`pool_projbal` と D8 の (i)(ii)，学び直した版 − `none_loco`，有限の確かめ，`fuse1` と DAI-1 の `ret` の W の照合，第 2 エンコーダの W・b の保存，集計の段のコミットと対の照合）に重大な問題は無い．実データの値は見ていない．

| 指摘 | 確かめた箇所 |
|---|---|
| m-a 選びの偏りの向き | 設計 3.2 節（対称なのは `enc2none − none` だけ．学び直した版 − `none` は学び直した側に不利）．台本 `PAIRS` の `(<出所>, "noneloco")` 5 組，`RULES` の `R1loco` |
| m-b D8 の基準 | 設計 3.4 節の (i)(ii)．台本 `pool_projbal`（496 行．交差の予測の softmax を感情ごとの総量でそろえてから argmax），`d8_agree_none_projbal`・`d8_agree_student_projbal`・`d8_agree_student_teacherbal`（509〜511 行），集計の D8(i)・D8(ii) の割合（755〜758 行） |
| m-c 有限の確かめ | 段 A の `fit`（289 行），段 B の `enc2` の `fit`（425 行），学び直しの `fit`（457 行），交差の予測（482 行）．`finite_p` を `conv` に残す |
| m-d 切り分けの部分集合と印 | `other_mod(student, teacher)`（AVT は 0），部分集合 `AVT_excl_other_excl`（主の版．744 行），部分相関は印が塊の中で変わる行だけ（`keep`），Freedman–Lane（残りを塊で並べ替えて当てはめに足す．628〜636 行），塊は (c, m)（主）と c（副）で K をまたいで同じ並べ替え（`block_plan`・`permute_by_plan`．自己検査 832〜847 行で多重集合の保存と K をまたぐ同じ並べ替え，AVT が部分相関から外れることを確かめる）．中心を引いた `uar_t` は Δ と同じなので外した（正しい） |
| m-e K = 0 の単位 | 設計 4.3 節の注（独立な単位は分割の 5 個．`diffs.csv` の `n_units` の読み方） |
| m-f `fuse1` と `ret` の W の照合 | 台本 469〜473 行（`W_TOL` = 1e-2 で止める．`mism` に残す） |
| m-g IEMOCAP の marker | 設計 5 節（顔の記述にだけ使う） |
| m-h 列名 | `grid_oracle_gap_per_TkindK`（329 行）と設計 5 節の注 |

軽微（計算を止めない）：
- [m-i] `r3_rows` の行が 4 未満のときの戻り（600 行）に `subset=""` が入っており，呼ぶ側の `dict(subset=sub_name, **row)` と鍵が重なって TypeError になる．120 単位の本計算では最小の部分集合（`student=AVT`）でも 40 行あるので届かないが，`--allow-partial` の試走で届きうる．戻りから `subset=""` を外す（1 語）．
- [m-j] `W_TOL` = 1e-2 の照合で止まった場合は，標準化か引き戻しか出発点が DAI-1 と違う印であり，許しを広げて通さない．止まったら差の大きい (m, K) を報告する．
- 起動は，この報告が見た台本の sha256（上）を `DAI11_SHA` に渡す．台本を 1 文字でも変えたら再検査の対象である．
