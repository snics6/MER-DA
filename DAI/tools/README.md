# 道具

| 台本 | 中身 |
|---|---|
| `md_tables_dai16_17.py`，`latex_tables_dai16_17.py`，`md_tables_dai18.py`，`md_tables_dai20.py`，`md_tables_dai21.py` | 結果の表（`docs/結果/` と `docs/説明/`）を結果ファイルから書き出す |
| `pdf_figs_proposal.py` | 説明の資料の図と表（`docs/説明/説明_提案_図表_*.tex`）を書き出す |
| `dai18_posthoc_selcounts.py`，`dai18_posthoc_voice40.py`，`dai20_posthoc_intercept.py`，`dai21_posthoc_cremad_sentence.py` | 計算の後の解析（出力は `results/<実験>/posthoc/`） |
| `extract_dai18_text.py`，`extract_dai20_text.py`，`extract_dai21_text.py`，`rebuild_explain_pdf_2026-11-14.py` | 結果の文の照合のときに作った道具．**git の履歴を整理の前の置き場で読むので，書き換えていない．**MER-DA の整理の前のコミット（`8fd53fd` まで）の作業木でだけ動く |
| `reorg_2026-11-15.py` | 2026-11-15 の整理を行った台本（1 回だけ流した．記録として残す） |
| `remap_eval_gate.py` | 評価の段の守り `frozen/eval_gate.json` を整理の後の置き場と sha256 で作り直した台本 |
| `collect_results.py` | git に入れる結果を選び，`results/MANIFEST.tsv` と `results/MANIFEST_outside_git.tsv` を書く |
