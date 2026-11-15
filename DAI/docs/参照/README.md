# 読むだけの資料（2026-11-15）

DAI の設計・台本・検査の報告・論文の原稿が引く資料のうち，今は書き足さないものを置く．どれも書き換えない．

| 場所 | 中身 | 読む役 |
|---|---|---|
| `DAI_paper/` | 旧版の論文の執筆フォルダ（2026-10-31〜11-14）．MER-DA の直下から中身を変えずに移した（`git mv`）．MER-DA の直下の `DAI_paper` はここへのリンク | |
| `DAI_paper/docs/venue.md`，`DAI_paper/docs/sources/` | 投稿要項の要約と原文 | 査読想定の役（`dai-review-ml`・`dai-review-affect`），`dai-editor`，論文のセッション |
| `DAI_paper/参照論文/` | 参照論文の要点のノートと PDF（PDF は出版社に著作権があるので git に入れない．一覧は同じフォルダの `README.md`） | 論文のセッション，`paper-citations` |
| `DAI_paper/inventory.md` | 実験の棚卸し | `prior-work-diff` |
| `DAI_paper/reviews/` | 前回の模擬査読と比較の設計の版 | 査読想定の役，`gate-code` |
| `DAI_paper/steps/`，`DAI_paper/overview/` | 論文の前半の設計（写しは `docs/設計/前半/`）と旧版の概要 | （記録） |
| `TAC_paper/` | IEEE TAC 向けの旧版の原稿の主張 `CLAIM.md` と模擬査読 `reviews/round1/`（MER-DA の `archive/paper/` から写した．原稿の全体はそちら） | `gate-code`，`dai-review-affect`，`thm-*` |
| `TAC_reports/` | 旧版の報告（MER-DA の `archive/reports/` から写した） | `thm-*` |
| `MER-DA_CLAUDE.md` | MER-DA の `CLAUDE.md` の写し（データの取り決め，実験の関門，進め方）．正本は MER-DA の側 | DAI だけを取り出したとき |

2026-11-15 の朝から夕方までここに置いていた MER-DA の文書の写し 6 本（`methods.md`・`seen-data.md`・`related-work.md`・`exploratory-analyses.md`・`results.md`・`researchvision.md`）は，同じ日に正本を `docs/` の直下に移したので消した（写しの中身は正本の古い版で，`researchvision.md` の更新記録の 1 行だけが違った）．
