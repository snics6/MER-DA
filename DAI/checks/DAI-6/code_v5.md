# 関門 1：実装と出力の再検査 4（DAI-6，2026-11-07，コミット ffcb980 と cc07f13）

検査したもの：`git diff 2affd25 cc07f13 -- DAI/exp/dai6_probe.py DAI/docs/実験6_下調べ_設計.md`．前の再検査は `DAI/checks/DAI-6/code_v4.md`（判定「直してから進む」）．`git diff b0f858a cc07f13` の塊は 714 行以降だけなので，`run_unit` は b0f858a と同じである．作業ツリーの `DAI/exp` と `DAI/docs` に未コミットの変更は無い．

## 判定

直してから進む．前回の直す点 1 と記録 2〜4 は入り，A1′ の参照を感情の中の並べ替えに替えた形も正しく動く．本番の計算（`run_unit`）は起動してよい．ただし `diagnostics.csv` で，分割ごとの印 `inseparable` とコーパスごとの印 `inseparable` が同じ名前で `merge` され，`inseparable_x`・`inseparable_y` に名前が変わって書き出される．このため，下の直す点 1 を直してから `--summarize` を流す．

### 直す点（番号つき）

1. 【直してから進む．`summarize` の前に】`DAI/exp/dai6_probe.py` の診断の集計では，`drows` に分割ごとの `"inseparable": bool(rules)` を入れ，`agg["inseparable"] = agg["n_insep"] >= 3`（コーパスごと）も作る．その後 `dg = dg.merge(agg, on="c", how="left")` で 2 つを合わせる．同じ名前の列が両方にあるので，pandas が `inseparable_x`（分割ごと）と `inseparable_y`（コーパスごと）に名前を変える．AFEW と IEMOCAP の分割 0 で流すと，`diagnostics.csv` に `inseparable` の列は無く，IEMOCAP の行は `inseparable_x = True`，`inseparable_y = False` になった．集計の中の `sub = agg[~agg["inseparable"]]` は `merge` の前に作るので，`n_agree_excl_inseparable` の値は正しい．ただし，表を読む人と後の台本は，列の名前からどちらがどちらかを知ることができない．
   求める対応：コーパスごとの列を `inseparable_corpus` などの別の名前にする．あわせて，印が付かない単位の `inseparable_rule` は空の文字列で書かれ，読み直すと NaN になるので，`"-"` などを入れる．
2. 【記録する】`de = pd.read_csv(...).copy()` の直後にもう一度 `de = de.copy()` がある（重複．値には影響しない）．

## A．差分と設計の対応

| 設計の項目 | 台本の箇所 | 合っているか |
|---|---|---|
| 3.4.1 A1′：b の側の行を感情の中だけで並べ替えた参照（100 回．A1 の帰無と同じ並べ替え），種と回数を記録 | `a1prime`（`perms = make_perms(len(y), n_ref, y, seed)` と `_s3_lda_cos2(Za, Zb[P], y)`） | 合う．感情の中で並べ替えると，b の側の感情ごとの平均と感情の中の共分散は変わらない．そのため，b の側の Fisher の判別の部分空間も変わらず，参照の揺れは正準相関の推定の揺れと感情の中の対応だけから出る．`make_perms` の assert が感情を保つことを確かめる．`assumptions_lda.csv` の 8 行に `n_ref` 100 と `ref_seed` がある |
| 3.5：「話者と感情を分けられない」の印を規則 (a)〜(d) で付け，規則の名前を残す | `rules` の作り方，`inseparable_rule` | 合う．IEMOCAP 分割 0 で `inseparable_rule = c`（マーカー），AFEW 分割 0 は印なし．列の名前の衝突は直す点 1 |
| 3.5：筋書きの a・b の版，帰無の 95% 点の揺れ | 設計 3.5 の文 | 合う |
| pandas の警告を消す | `de.copy()` | 合う．集計のログに警告は出なかった（重複は直す点 2） |
| `--selftest` の (8) | n = 2,200，p = 48，感情の方向 1.2 倍 | 合う．通った（42 秒）：前提が成り立つとき「離れている」は 20 回中 1，共有が強いとき 10 回中 10 |

## A．小さく流した結果

- `DAI/checks/DAI-6/code_run_v4/out/`（b0f858a で流した AFEW と IEMOCAP の分割 0）の写し `DAI/checks/DAI-6/code_run_v5/sum/` で，`MERDA_GPU_FRACTION=0.3 ./.venv/bin/python -W default DAI/exp/dai6_probe.py --out DAI/checks/DAI-6/code_run_v5/sum --summarize --allow-partial` を流した．10 秒で 9 つの表を書いた．警告は出なかった（ログ `DAI/checks/DAI-6/code_run_v5/summarize.log`）．
- 前回（2affd25）の集計（`DAI/checks/DAI-6/code_run_v4/sum/`）と比べた．`criteria`・`assumptions`・`q2_shares`・`per_emotion`・`confusion`・`subsets`・`scenario_pairs` は，共通の数値の列の差が 0.0 だった．`assumptions_lda.csv` で変わったのは，参照の作り方を替えたことによる `below_ref_q05` 1 個（AFEW の V–T の a の側）と，参照の平均・5% 点である．`diagnostics.csv` は列の名前が変わった（直す点 1）．
- NaN は `diagnostics.csv` の `inseparable_rule` の 1 個（印が付かない AFEW の行の空の文字列）だけで，Inf は無い．
- `--selftest` は通った（ログ `DAI/checks/DAI-6/code_run_v5/selftest.log`）．
- 見た値：`assumptions_lda.csv` の `below_ref_q05` 8 個，`diagnostics.csv` の 2 行（r_clean，安定，印）．成績の値（UAR）は開いていない．本体が `docs/seen-data.md` に記録する．

## B．予想される指摘と，出力から答えられるか

`code_v4.md` の表から変わらない．A1′ の参照の回数を増やす指摘は，集計だけで流し直せる（AFEW 1 単位 4.4 秒，IEMOCAP 1 単位 8.8 秒．本体の測定）．

## 指摘

### [I1] `diagnostics.csv` の印の列が `merge` で名前を変えられる
- 重さ：直してから進む（`summarize` の前に．本番の計算は起動してよい）
- 該当箇所：`DAI/exp/dai6_probe.py` の `drows.append({... "inseparable": bool(rules), ...})`，`agg["inseparable"] = agg["n_insep"] >= 3`，`dg = dg.merge(agg, on="c", how="left")`
- 問題：上の直す点 1．書き出した表に `inseparable` の列が無いので，検算の台本が `KeyError: "['inseparable'] not in index"` で止まった．
- 計算の後で直せるか：直せる（集計だけ）．
- 求める対応：直す点 1 のとおり．

### [記録] `de.copy()` の重複
- 重さ：記録する
- 該当箇所：`summarize` の `de = pd.read_csv(out / "decomp.csv", low_memory=False).copy()` と次の行
- 求める対応：1 行を消す．

## 当てて指摘の無かった項目

差分について，I1（設計 3.4.1・3.5 と台本．列の名前を除く），I3（A1′ の参照は候補のラベルを測定だけに使う），I7（前回の集計の値の再現．差 0.0），I9（参照の種と回数），I11，R12（A1′ の参照の分布）．

## 目録に無い種類の不備

- I19（案）：集計の表を `merge` でつなぐときは，両方に同じ名前の列が無いかを確かめる（pandas は黙って `_x`・`_y` を付ける）．分割ごととコーパスごとのように粒度の違う同じ意味の印には，粒度を表す別の名前を付ける（DAI-6 の gate-code の再検査 4）．
