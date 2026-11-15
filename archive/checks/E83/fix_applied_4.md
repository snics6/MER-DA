# 統計の再検査 2 の指摘の直し（E83，2026-11-03）

対象：`checks/E83/stats_recheck2.md`（直してから進む 1，記録 2）．設計は `docs/exploratory-analyses.md` の E83 3・4 節，台本は `exp/analyze_pairs_e83.py`（1,357 行）．E85 の節と作業木の `exp/analyze_projection_e85.py` は触っていない．コミットはしていない．`checks/E83/code_recheck3.md` はこの時点（直しの後）でまだ届いていない．

守ったこと：(A)(B) は流していない．(C) の試走（`--out <scratchpad>/e83_fix4 --stages c --targets MELD --boot 20 --smoke`．29 秒，RSS 0.74 GB，警告なし，13,950 行，J_dir の NaN 0．出力は消した）と，`p3a_table` を results/ を読まない人工の配置で動かした検算．`python -m py_compile` は通る．

| 指摘 | どこをどう直したか |
|---|---|
| 1 [S2・S10・D14]（直してから進む）：印を除く版では (iii) の対象が印（手法の結果）で決まる | 台本：`elig_and_keys`（892〜901 行．効果の出うるセルと 6 対すべてが効果の出うるセルの単位），`p3a_eval` に `m_ref`（印を含む行．909〜942 行）．対象の単位・到達しうる最小の p（`p_min_attainable`）・`judgeable` は印を含む行で決め，使う行（印を除く版）の p は対象のうち印の付いた行を含まない単位で計算する（`full_units_used = full_units_used & full_units_ref`，918 行）．新しい列：`n_units_full_elig_used`，`full_elig_units_by_corpus_used`，`n_full_elig_dropped_by_flag`，`p_min_attainable_after_flag`，`judgeable_after_flag`（986〜991 行）．`P3a_verdict` に `not_judgeable_by_flag`（印を含む行では判定できるが印で対象が減って判定できない）を足した（972 行）．`cond_p` は両方の judgeable が真のとき（959 行）．`p3a_table` は印を含む行 `m_all` を `m_ref` に渡す（1012〜1016 行）．設計 10106 行（「手法の結果に依らない」を印を含む行についての文に直し，印を除く版の扱いと列を書いた．not_judgeable_by_flag の文），10118 行（文の規則 (2) に not_judgeable_by_flag の書き方），4 節 10139 行（印で対象から抜ける偏りの向きは決まらない）．人工の配置で gate-stats の例を再現：3 コーパスで平静–怒り，印なし → p ＝ 0.00463；A の喜び–悲しみに印 → 印を除く版は対象 3 → 2，p ＝ 0.0278，落ちた 1，判定できる；A と B に印 → 印を除く版は対象 1，`p_min_attainable_after_flag` ＝ 0.167，`not_judgeable_by_flag`（印を含む版は judgeable のまま）；効果の出うるセルが平静–怒りだけ → `not_judgeable` |
| 2 [S2]（記録）：顔の P3a は判定できない見込み，顔の最小の型 | 設計 10108 行：計算の前の見込み（E13 4 節の映像の喜び–悲しみの損 −3.9・−2.4，平静–喜び −1.8．感情を共有しない対の性質．判定できないときは平静–怒りの「減らした」を効果の出うるセルの中でコーパス・K・世界ごとに並べる記述にとどめる）．10111 行：到達しうる最小の p は音声 4^−c・顔 6^−c，顔の最小の型（(iii) は 2 コーパスで 0.028．(i) には平静–怒り以外の塊が 2 つ以上要り，対象の単位にあれば (iii) の p を上げる） |
| 3 [S2・S4]（記録）：当たったときの文を対象の単位に限る | 設計 10118 行：当たりの文を「6 対すべてで none の向きの損が δ_d,対 以上だった単位（コーパスごとの数）で…偏った」とし，対象の外の「減らした」の数（`n_pos_elig_not_full`）とそのうち怒りを含む対の数（新しい列 `n_pos_elig_not_full_selected`，台本 988 行）を並べる．音声の f 全体の偏りとは書かない |

## 入れなかったもの

- `docs/methods.md` 9.3 節の目録（S15 の追記・S16・S17）は入れていない（範囲は設計と台本．前回の S15・S16 も未収録なので本体がまとめて振る）．
- gate-stats の検出力の表（手順 4）は設計に写していない（判定の規則ではない）．

## git diff --stat（HEAD 1b775e2．コミットはしていない）

```
 docs/exploratory-analyses.md | 10 +++---
 exp/analyze_pairs_e83.py     | 84 ++++++++++++++++++++++++++++----------------
 2 files changed, 59 insertions(+), 35 deletions(-)
```
