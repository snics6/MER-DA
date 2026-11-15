# paper/ — TAC 論文の執筆フォルダ

MER-DA の結果を基に IEEE Transactions on Affective Computing（TAC）へ出す統合論文（分析＋手法）を書く場所．研究ディレクトリ MER-DA の直下に置く（2026-10-27 研究者の決定）．執筆の方針（成果は数値で述べる／査読対応で結果を作らない／査読ループと語数の圧縮の回し方／主張契約）は `~/papers/CLAUDE.md` が正本で，査読役・リバッタル・AC・claim-drift の定義は `~/papers/.claude/agents/` を使う．

## 正本

| 目的 | ファイル |
|---|---|
| 主張契約（1 文・土台・反証条件・最小の証拠 3 つ・主張しないこと） | `CLAIM.md`（暫定承認．編集には研究者の承認が要る） |
| 章立て・題名・TAC の方向性・数理の書き方の注意 | `../docs/paper-claim.md` 6〜11 節 |
| 投稿先の要項と投稿前の点検表 | `docs/venue-tac.md` |
| 先行研究と差の書き方（関連研究の 7 段落の構成） | `../reports/TAC 論文 先行研究の精読と穴.md`，`../docs/related-work.md` 1.13・1.14 節 |
| 各主張の最新の状態と結果の場所 | `../docs/results.md`（3・4 節） |
| 数値台帳 | `results/numbers.json`（`results/README.md`） |
| 実験要求 | `requests.md` |

## 破ってはいけない規則

1. **本文の実験由来の数値は `\NUM{key}` でしか書かない．**key の実体は `results/numbers.json`（値・出所・検算日）．`make numbers` で `numbers.tex` を生成し，生成物もコミットする．値は `../docs/results.md` から写さず，結果ファイルから取り直す
2. **`CLAIM.md` に無い主張を本文に書かない．**「初めて」と書けるのは `../docs/paper-claim.md` 6.2 節の 4 点だけで，序論の貢献の文に限る
3. **足りない結果は `requests.md` に起票し，MER-DA 側で作る．**長時間の処理はこのフォルダから起動しない．退避（返らないときに主張をどこまで弱めるか）を同時に書く
4. **参考文献は原典で書誌を確かめてから入れ，確かめた水準をコメントで添える**（`references.bib` の冒頭の規則）
5. **査読役に渡すのは `main.tex`・`sections/*.tex`・`references.bib`・補足資料だけ．**`CLAIM.md`・`reviews/`・MER-DA の結果や文書は渡さない

## ビルドと検査

```bash
make            # numbers.tex を生成してビルド
make check      # 改稿のたびに走らせる（ビルド・頁数・数値の出所・査読ループ）
```

IEEEtran（V1.8b）のクラスと bst はこのフォルダに同梱する（TeX Live に無い）．設定は `journal, compsoc`．投稿時の PDF は 11.5 頁を目安にする（12 頁の上限は参考文献・略歴込みで最終組版で数える）．

参考文献の正本は `references.bib`（確かめた水準のコメントつき）．組版には `scripts/ieee_bib.py` が作る `references_ieee.bib`（会議論文の編者・出版社・叢書を外す．誌名・会議名は正式名のまま）を使い，本文は `bstctl.bib` の設定で URL を出さない．`references_ieee.bib` は生成物なので手で直さない（make が作り直す）．

2026-10-27 に本文を 15 頁から 12 頁に縮めた．補足資料へ移したもの：源の価値の散布図（S3），ラベル無しの選び方の図（S5），比較の棒グラフ（S6），K ごとの曲線（S7），エントロピー最小化（S8），微調整の図と詳細（S9，S7 節）．本文から補足の図・節は番号を直書きで引くので，補足の図を足したり並べ替えたりしたら `supplementary.aux` で番号を確かめて本文を直す．
