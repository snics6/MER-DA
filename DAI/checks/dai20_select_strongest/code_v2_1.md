# 関門 1：実装と出力（計算の後の解析 dai20_select_strongest 第 2.1 版，2026-11-17，コミット 0ee5d4e）

## 判定
判定：通す．ret の raw の判定の照合の行は検査の写し `sel_tolerant.py` と同じ条件になり，足した感情ごとの件数の列は集計の `per_emotion.csv` と一致し，4 コーパスの全部が最後まで流れて第 2 版の写しと同じ出力を出した（依頼の語では「OK」）．

対象：`tools/dai20_select_strongest.py` の 08bf5b9 → 0ee5d4e の差分（`git diff 08bf5b9 0ee5d4e -- DAI/tools/dai20_select_strongest.py`）．差分は docstring の 1 行，照合の行（133〜138 行），感情ごとの列（257〜258 行），provenance の 1 項目（270 行）だけである．UAR の水準と差の値は画面にも報告にも出していない．

## 差分の照合

| 直した点 | 台本の箇所 | 合っているか |
|---|---|---|
| ret の raw の判定の照合で，float32 の丸めによる同点を一致と数える | 133〜138 行：`hit = raw[m] == Pt[j].argmax(1)`，`tie_ok = Pt[j][np.arange(len(raw[m])), raw[m]] == Pt[j].max(1)`，`if not (hit \| tie_ok).all(): raise` | 合っている．検査の写し `checks/dai20_select_strongest/code_run_v2/sel_tolerant.py` の 133 行と同じ条件である．判定を変えず，止めるかどうかだけを決める |
| 最大と等しいだけの発話の数を記録する | 138 行 `n_argmax_tie += int((~hit & tie_ok).sum())`，270 行 `ret_raw_argmax_float32_ties` | 合っている．argmax と一致した発話は数えず，最大と並ぶだけの発話を数える．4 コーパスで流すと 1 になった．第 2 版の検査で見つけた 1 発話（MELD_ALL_f3_r7，K = 5，声）と同じ数である |
| 感情ごとの出力に `n_fr`，`n_units`，`n` を足す | 257〜258 行 | 合っている．`dd` の値は `P12._finish` の後の (塊の名前, 正解の数 / m, 件数 / m) で，この台本では 1 つの (分割, 種) に 1 回しか足さないので m = 1 になる．したがって `x[2][:, i].sum()` を単位について足した値は，集計の `n`（全部の単位の合計．`dai20_eval.py:526〜527`，`558`）と同じ量である．`n_units = len(dd)` と `n_fr = r_["n_fr"]` も集計と同じ作り |

## 流した結果

- コマンド：`./.venv/bin/python tools/dai20_select_strongest.py --out-dir /tmp/claude-1000/-home-snics-MER-DA/1c397eb5-7560-4b25-97d9-b9ef70549ea7/scratchpad/gate_select_strongest/v21full`（コミット済みの台本をそのまま，4 コーパス．CPU，3 分 34 秒）．エラーも警告も出なかった
- 自己検査：(1) 集計の作り直しのずれの最大 7.1e-15，(2) 選んだ分類器は 600 行で一致，(3) avg の adj の判定の作り直しで違う発話 0．`ret_raw_argmax_float32_ties` は 1
- provenance：`commit` は 0ee5d4e，`dirty` は False，`script_sha256` の頭 a3370e598adee28e は `git show HEAD:DAI/tools/dai20_select_strongest.py` の sha256 の頭と一致した（第 2 版の検査の記録の指摘 2 の確かめ方）
- 検算の台本：`checks/dai20_select_strongest/code_run_v2/verify_v21.py`
  - 足した列：prop・prop@raw・prod の 144 行で，`n`・`n_fr`・`n_units` は集計の `per_emotion.csv` と違う行が 0．同じ (コーパス, K, 感情) で版によって `n` が違う組は 0（どの版もテストの発話は同じ）．`n_units` と `n_fr` はすべて 50
  - 出力の差分：差，水準，選びの 3 つのファイルは，第 2 版の検査で流した写しの出力（`…/gate_select_strongest/v2tol`）と `commit`・`dirty` の列を除いて完全に一致した．感情ごとのファイルも，前からある列は完全に一致した

## 指摘
指摘なし．当てた項目：I（照合の条件，件数の作り，記録），R（感情ごとの件数の列）．第 2 版の記録の指摘のうち，「dirty が tools/ を数えない」は上の sha256 の照合で確かめられた．「sel_test・sel_orc は参照の版」は結果の文で書くことで，台本の直しは要らない．

判定：OK
