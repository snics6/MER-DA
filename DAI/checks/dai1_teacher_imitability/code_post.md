# 計算の後の点検：実装と出力（計算の後の解析 dai1_teacher_imitability，2026-11-16，コミット 8fe5022）

## 判定
判定：通す．差分は第 2 版の検査の求めどおりで，本番の 2 つの出力は第 2 版の検査で直した写し（`code_run_v2/patched.py`）の出力と，名前を替えた中央値の 3 列を除いて差 0.0 で一致し，欠けも重複も無い．

点検した差分：`git diff 1dba48f 8fe5022 -- tools/dai1_teacher_imitability.py`（作業木に差分なし）．点検した出力：`results/dai1/stats/teacher_imitability.csv`，`results/dai1/stats/teacher_imitability_summary.csv`（どちらも 2026-11-16 02:05 に書かれた）．前の報告は `code.md`（2a0d326）と `code_v2.md`（1dba48f）．

## (1) 差分の検査

| 第 2 版の指摘 | 8fe5022 の直し | 合っているか |
|---|---|---|
| 1．:62〜63 の `r.T` で落ちる | `frozen[r["T"]]`，`f"{r['T']}_f{f}_r{rr}.npz"` | 合う．ほかに `r.T` も `x.T` も残っていない（`r.K`，`r.teacher`，`r.student` は属性の名前と重ならない列で，値は正しく読まれる．本番が 48 点を書いたことで確かめた） |
| 2．中央値は候補の全体の D で，ほかは D_nk_R1 | 3 つの中央値を `D_nk_R1` に替え，列の名前を `pos_hurt_Dnk_median`，`pos_gain_ci_pos_Dnk_median`，`pos_other_Dnk_median` にした | 合う．候補の全体の D の中央値は出さなくなった（D の順位相関 spearman_*_D の 4 列は並べる量として残る） |
| 2．docstring | 主の量（D_nk_R1），並べるだけの量（候補の全体の R・D と D_nk_R0），増えた入力（実験 14 の単位，assumptions.csv），frac_low の 12 点と除いた版，gain_resid の直線，D_nk に区間を付けないことを書き足した | 合う |
| 3・4・5（記録の項） | 区間を付けないことと gain_resid の当て方を docstring に書いた．来歴の上書き，pos_other_n，pivot の集計は直していない | 記録の項のまま（下の「残る記録」） |

差分は上の 3 か所だけで，計算の作りは 1dba48f と同じである．

## (2) 本番の出力と patched.py の出力の照合

照合の台本：`checks/dai1_teacher_imitability/code_run_v2/post.py`．画面に出したのは行の数，列の名前，差の最大，真偽，NaN の数だけで，相関・中央値・数えの値は出していない．

- `teacher_imitability.csv`：本番 48 行，写し 48 行．列は同じ 27 列．鍵 (T, K, student, teacher) で 48 行すべてが結びつく．commit と dirty を除くすべての列で，数値の列の差の最大は 0.0，文字と真偽の列は全部の行で同じ，NaN の位置も同じ．
- `teacher_imitability_summary.csv`：本番 6 行，写し 6 行．鍵 (K, subset) で 6 行すべてが結びつく．違う列は名前を替えた中央値の 3 列だけ（本番 `*_Dnk_median`，写し `*_D_median`）．それ以外の列（n，n_margin_pos，n_pairs，pairs_*，spearman_* の 6 列，pos_hurt_n，pos_gain_ci_pos_n）は差の最大 0.0．
- 中央値の 3 列：本番の点の表の D_nk_R1，margin，gain_lo，gain_hi，frac_low から，6 行 × 3 列を作り直すと，差の最大は 1.8 × 10⁻¹⁵ で，NaN は 0 個．
- 来歴：両方のファイルの commit 列は 8fe5022（点検した台本のコミット），dirty は偽．

## (3) 欠けと重複

- 点の表：48 点．(コーパス, K) ごとに 6 点で 8 組，(生徒, 先生) ごとに 8 点で 6 組．鍵の重複 0．数値の列に NaN と Inf は無い（NaN は文字の列 excluded の空欄だけ．teacher_margin.csv で single_m の点は除外の印が空であることによる）．frac_low は 12 点（第 1 版の報告で数えた 6 組 × 2 生徒と同じ）．
- 要約：6 行（K = 5，25，all × all_points，frac_ok）．n は 24，24，48，18，18，36，n_pairs は 12，12，24，7，7，14 で，第 2 版の検査で見込んだ数と同じ．NaN と Inf は無い．
- ログ：この台本は標準出力に 1 行を書くだけで，ログのファイルは無い．本体の実行の出力は渡されていないので，例外と警告は確かめていない（写しを流したときの警告は `frozen.json` を閉じない ResourceWarning の 1 件だけだった）．

## 残る記録（結果に影響しない）

- 2 つの出力は `results/*` の `.gitignore`（`DAI/.gitignore:12`）にかかり，git に入っていない．入力の `teacher_margin.csv` と `teacher_margin_summary.csv` は `git add -f` で入っている．DAI の取り決め（集計と記録は git に入れる）に合わせるなら，同じく `git add -f` で入れる．
- `commit` 列は teacher_margin.csv の commit（dd309bb）を上書きしたままで，実験 14 の commit（fbd6c07）と入力の sha256 は出力に無い（`code.md` の指摘 5）．結果の文の冒頭に，入力の 3 つ（teacher_margin.csv の dd309bb，実験 14 の descriptive.csv・assumptions.csv・単位の fbd6c07，実験 1 の単位の ba98974）を書く．
- 区間が 0 を含む区分の点の数（pos_other_n）は要約に無い．n_margin_pos − pos_hurt_n − pos_gain_ci_pos_n で出せる．
- 書くときの注意は `code.md` と `code_v2.md` のとおり（D_nk に区間が無い，frac_ok の 7 組に MOSEI の組が無く AFEW は 1 組だけ，K = 5 と K = 25 の組は同じ単位を共有する，学び直しは確率の柔らかいラベルで R は argmax の硬いラベル，AFEW の言葉は Whisper の書き起こし，要点 6 の 36 点のうち 48 点に入るのは一部）．

## 目録に無い種類の不備

なし（`code_v2.md` に書いた「pandas の行の属性で列を読む」の案のほかに，新しい種類は見つからなかった）．
