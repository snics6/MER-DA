# 関門 1：実装と出力の再検査 5（DAI-6，2026-11-07，コミット 4219446）

検査したもの：`git diff 28bafdc 4219446 -- DAI/exp/dai6_probe.py`（`summarize` の 3 か所だけ．`run_unit` は b0f858a から変わらない）．前の再検査は `DAI/checks/DAI-6/code_v5.md`（判定「直してから進む」）．作業ツリーの `DAI/exp` に未コミットの変更は無い．

## 判定

通す．`code_v5.md` の直す点 1 と記録 2 は入り，`diagnostics.csv` の列の名前の衝突は無くなった．集計のほかの値は前回と同じである．本番の計算と `--summarize` をこのコミットで流してよい．

## 確かめたこと

- 差分：コーパスごとの印を `agg["inseparable_corpus"]` にし，`sub = agg[~agg["inseparable_corpus"]]` も同じ名前を使う．印の無い単位の `inseparable_rule` は `"-"` にした．重複していた `de = de.copy()` を消した．
- 試走：`DAI/checks/DAI-6/code_run_v4/out/`（b0f858a で流した AFEW と IEMOCAP の分割 0）の写し `DAI/checks/DAI-6/code_run_v6/sum/` で `--summarize --allow-partial` を流した．警告は出ずに終わった（ログ `DAI/checks/DAI-6/code_run_v6/summarize.log`）．
- `diagnostics.csv` の列は `inseparable_a`・`inseparable`・`inseparable_rule`・`n_insep`・`inseparable_corpus` を含み，`_x`・`_y` の付いた列は無い．NaN は 0 個（AFEW の `inseparable_rule` は `-`，IEMOCAP は `c`）．前回の `inseparable_x` と今回の `inseparable` が一致し，前回の `inseparable_y` と今回の `inseparable_corpus` が一致した．
- ほかの 8 つの表（`criteria`・`assumptions`・`assumptions_lda`・`q2_shares`・`per_emotion`・`confusion`・`subsets`・`scenario_pairs`）は，形が同じで，数値の列の差は 0.0 だった．
- `dg` と `agg` の列の名前を並べ，`c` のほかに同じ名前の列が無いことを確かめた．
- 新しく見た成績の値は無い．

## 指摘

指摘なし．当てた項目：I1（`code_v5.md` の直す点 1 と記録 2），I7（前回の集計の値の再現），I19 の案（`merge` の列の名前の衝突）．
