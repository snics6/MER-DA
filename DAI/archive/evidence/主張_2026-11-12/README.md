# 主張の根拠の控え（2026-11-13 作成）

採用した主張の骨格 `DAI/docs/主張_採用した説明_2026-11-13.md`（と正本 `DAI/docs/主張_2026-11-12.md`）の数を導いた実験・台本・結果を，ここから辿れるようにした．

## 1. 守り方

| 何を | どこに | 作り方 |
|---|---|---|
| 結果の要約と記録（小さいファイル 136 本．gzip） | このフォルダの `results/`（git に入る） | `DAI/evidence/make_evidence.py` |
| 結果の全ファイル（3,080 本，約 26 GB）の sha256 と大きさ | `sha256_results.tsv`（git に入る） | 同上 |
| 結果の全ファイルの丸ごとの写し | `/mnt/c/Users/snics/MER-DA_backup/results_2026-11-13/`（WSL の外の Windows 側．git に入らない） | rsync（`logs/backup_2026-11-13.log`）．`sha256_results.tsv` と照合する |
| 台本と設計と結果の文と検査の報告 | git（github.com/snics6/MER-DA，非公開）．下の表のコミット | — |

`results/` は `.gitignore` で git に入らない．元の置き場は `results/dai1`，`results/dai3`，`results/dai11_probe`，`results/dai14`．

## 2. 文ごとの根拠

| 主張の文（`主張_採用した説明_2026-11-13.md`） | 結果のファイル（抜き出しの条件） | 計算したコミット | 台本 | 設計と結果の文 | 照合 |
|---|---|---|---|---|---|
| 1 節：本番の上がり幅は声で 4 コーパス +2.4〜+4.1 点 | `results/dai1/stats/families_flagged.csv`（family = 1，K = 5，student = A．effect） | ba98974 | `DAI/exp/dai1.py`（評価の回），`DAI/exp/dai1_stats.py` | `DAI/docs/実験1_設計.md`，`DAI/docs/結果の文の下書き_2026-11-05.md` | `DAI/checks/DAI-eval/claim_v3.md` |
| 2 節：条件 1（余地） | `results/dai3/eval/expl4_corpus_restored.csv`（K = 5，subset = all，quantity = room．列 A・V・T_student）．融合は `results/dai3/eval/decomp.csv`（version four，AVT\|none\|5 の sum） | ba98974 | `DAI/exp/dai3.py` | `DAI/docs/実験3_設計.md` | 記述の出力（結果の文にしていない） |
| 2 節：条件 2・3 の値（モダリティごとの理由） | 下調べ：`results/dai11_probe/req_vs_gain.csv`（出所 other，30 単位の平均の delta_uar と proj_gain）．評価の回の版：`results/dai14/summary/conditions.csv`（K = 5） | 下調べ 67e3553，評価の回の版 fbd6c07 | `DAI/exp/dai11_probe.py`，`DAI/exp/table_dai11_cells.py`，`DAI/exp/dai14.py` | `DAI/docs/実験11_下調べ_設計.md`・`実験11_下調べ_結果の文の下書き_2026-11-09.md`，`DAI/docs/実験14_設計.md` | 下調べの 3 条件の表の値は照合の前（主張 5 節に書いた）．評価の回の版は `DAI/checks/DAI-14/claim_v7.md` |
| 3 節：声 − 顔の差（言葉の判断の再現）IEMOCAP +16.7，MELD +14.9，MOSEI +15.5 | `results/dai14/summary/predictions.csv`（K = 5，readout = R1，control = cl，subset = all，prediction = P1） | fbd6c07 | `DAI/exp/dai14.py` | `DAI/docs/実験14_設計.md`（第 5 版），`DAI/docs/実験14_結果の文の下書き_2026-11-12.md`（第 7 版） | `DAI/checks/DAI-14/claim_v7.md` |
| 3 節：感情と話者を差し引いた差 IEMOCAP +17.3，MELD +13.9，MOSEI +10.2 | 同上（prediction = P4a）．数えは `criteria.csv`（primary = True） | fbd6c07 | 同上 | 同上 | 同上 |
| 3 節：顔の判断の再現は基準を満たさない | 同上（prediction = P2・P4b，`criteria.csv` の hit） | fbd6c07 | 同上 | 同上 | 同上 |
| 3 節：言葉の特徴も声の判断をまねできた（記述） | `results/dai14/summary/descriptive.csv`（K = 5，readout = R1，subset = all，quantity = D_shufcl，student = T，target = single_A） | fbd6c07 | 同上 | 同上 | 同上 |
| 5 節：評価の回の真似できる分（声）0.4〜3.7，IEMOCAP の区間 −3.8〜4.5 | `results/dai14/summary/conditions.csv`（K = 5，student = A，quantity = imitable_R0） | fbd6c07 | 同上 | 同上 | 同上 |

## 3. 共通の部品とデータ

- 共通の台本：`DAI/exp/dai_data.py`（読み込み，分割，塊），`DAI/exp/dai_solver.py`（GPU の L-BFGS），`DAI/exp/dai1_select.py`，`measure/data.py`（`load_group`），`measure/split.py`．
- 分割：`DAI/splits/<コーパス>.csv`．固定のファイル：`DAI/frozen/`．
- データ：特徴は `data/cache/*.npz`（git に入らない．`run/build_data.sh` と `features/` で作り直す），発話一覧は `data/metadata/`（git に入る）．作り方は `data/README.md`．
- 実験 14 の本番は 3 回に分けて流した（照合で 2 回止まった．設計 10.1 節）．3 回目は fbd6c07 の作業木から流した．計算の後の点検は `DAI/checks/DAI-14/code_post.md`．

## 4. 作り直し

```
.venv/bin/python DAI/evidence/make_evidence.py   # 控えと sha256 の一覧を作り直す
```
