# DAI/paper — Discover Artificial Intelligence に出す論文の原稿（2026-11-14〜）

このフォルダで，Discover Artificial Intelligence（DAI）に出す解析論文の原稿を LaTeX で書く（2026-11-14 研究者の指示「DAI/にpaperフォルダを作りそこで作業しよう」）．
親の `../CLAUDE.md`（DAI の設定・決定・検査役）と `../../CLAUDE.md`（MER-DA 全体の取り決め．DAI の中の写しは `../docs/参照/MER-DA_CLAUDE.md`）はそのまま当てはまる．
旧版の作業場 `../docs/参照/DAI_paper/`（要項 `docs/venue.md`，参照論文 `参照論文/`，棚卸し．2026-11-15 に MER-DA の直下から移した）と TAC 向けの旧版の主張と模擬査読 `../docs/参照/TAC_paper/`（原稿の全体は MER-DA の `archive/paper/`）は読むだけにする．原稿はこのフォルダにだけ書く．

## 中身の正本（原稿はこれに合わせる）

| 対象 | 正本 |
|---|---|
| 採用した主張の骨格（捨てない） | `../docs/主張/主張_採用した説明_2026-11-13.md`，主張の文・根拠の数・弱いところは `../docs/主張/主張_2026-11-12.md` |
| 節ごとに使う実験・結果・照合 | `../docs/実験の地図.md` |
| 論文の骨格の相談の記録 | `../research_notes/解析論文のストーリー_2026-11-08/骨格の案_2026-11-09.md` |
| 各実験の結果の文 | `../docs/結果/実験*_結果の文の下書き_*.md`（`gate-claim` を通した版だけを使う） |
| 投稿要項 | `../docs/参照/DAI_paper/docs/venue.md` |
| 先行研究と位置づけ | `../docs/related-work.md`，`../research_notes/` の調べ直し |

## 書き方の取り決め

- **文章を書く・直す前に，スキル `mer-paper-style`（`../../.claude/skills/mer-paper-style/SKILL.md`）を読む**（2026-11-15）．

- 本文は英語，学術論文の語調で書く．研究者への説明は日本語で平易に書く（`~/.claude/CLAUDE.md`）．
- **本文の数値は，`gate-claim` を通した結果の文か結果ファイルからだけ取る．**出所の無い数値を書かない．
- 計算中の実験の数値は `TBA` にするか，下調べの値を仮に置く（2026-11-14 研究者の指示）．仮の値には `\PROV{}`（下調べの値の印）を付け，出所をコメントに書き，最終版で必ず置き換える．
- 書き方は「データ，モデル，実験，結果」の順に事実を書く．評価の手順（どのデータを何に使ったか）は実際のとおりに書き，ラベルや断り書きで評価の性格を言い換えない．
- 引き継ぎの正本：実験のセッションからの伝言 `../docs/記録/論文のセッションへの伝言_2026-11-14.md`．`../docs/設計/実験21_設計.md` は執筆では読まない（研究者の指示）．
- 要旨と序論で成果を述べるときは具体的な数値で書く．程度を形容詞・副詞で表さない．
- 査読（想定の査読を含む）への対応では，手元の結果から言えることの追記，書き方の明確化，主張を弱める・範囲を限る・削る，の 3 種類だけを行う．無い結果を書かない．
- 原稿の段の査読想定は `dai-editor`・`dai-review-ml`・`dai-review-affect` に行わせ，報告は `../reviews/<回>/` に置く．

## ファイル

| 場所 | 中身 |
|---|---|
| `main.tex` | 原稿の本体（`sn-jnl`，`sn-nature` の参考文献様式） |
| `sections/` | 節ごとの `.tex`（`main.tex` から `\input` する） |
| `figures/` | 図．図を作る台本と図の数値の出所も同じ場所に置く |
| `references.bib` | 参考文献（2026-11-15 に 50 項目．選んだ理由は `notes/引用の候補_2026-11-14.md` の末尾の「削る案」．DOI のある項目は Crossref の BibTeX，ほかは ACL Anthology・NeurIPS と PMLR の論文集・JMLR・arXiv の API で確かめた） |
| `references/` | 書き方の手本と読む論文の PDF（git の外）．一覧は `references/README.md` |
| `notes/` | 書き方の読み取り（Ryumina 2025，DAI の手本 3 本）と引用の候補 |
| `sn-jnl.cls`，`sn-nature.bst`，`sn-jnl-user-manual.pdf` | Springer Nature の雛形．2026-11-14 に `~/papers/common/styles/` から 1 回だけ写した（以後は読まない）．`sn-nature.bst` の `format.in.ed.booktitle` を 2026-11-15 に 1 か所直した（editor の無い `@inproceedings` で booktitle が出ずに BibTeX が止まる誤り．直した行に注記あり） |

組み方：`make`（`latexmk -pdf`）．PDF と中間生成物は git に入れない．

## 未決

- 節の構成（骨格を研究者と 1 節ずつ確かめてから `main.tex` に入れる）．
- 参考文献の様式．要項は NLM（Vancouver 方式）を推し，DOI を完全な URL で書くよう求める．手元の `sn-nature.bst` は DOI を出さない（出す行が注釈で消してある）．投稿前に誌の雛形の版と様式を確かめる．
