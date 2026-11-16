# DAI/slides/ — DAI の研究の発表資料（Beamer，LuaLaTeX，日本語と英語）

DAI に出す研究（`DAI/CLAUDE.md`）の途中経過や発表の資料を作る場所（2026-11-05 研究者の指示「DAI の下に slides というフォルダを作り，そこにスライド（PDF）を作ります．日本語と英語で」）．
資料は 1 つにつき 1 フォルダ（`<日付>-<用途>/`）で，中に `main_ja.tex`・`main_en.tex`・`deck.tex`・`parts/`・`figures/`・`scripts/`・`.latexmkrc` を置く．

## 自己完結

- 見た目の設定 `common/`（`preamble.tex`・`theme.tex`・`macros.tex`・`merda.tex`）は 2026-11-05 に MER-DA の直下の `slides/common/`（今は MER-DA の `archive/slides/common/`）から 1 回だけ写した．以後は同期しない．直すならここを直す
- MER-DA の旧版の `slides/`（今は MER-DA の `archive/slides/`）の資料（TAC 向け）とは語も台帳も共有しない．DAI の語は `docs/設計/` の設計の文書に合わせる

## 組み方

```bash
cd DAI/slides
make DECK=2026-11-05-途中経過 data   # 結果ファイルから figures/*.csv と figures/numbers.tex を書き出す（数秒）
make DECK=2026-11-05-途中経過        # 日本語版と英語版の PDF（<資料>/<資料>_ja.pdf，_en.pdf）
make DECK=2026-11-05-途中経過 png    # 全ページの PNG（目視用．build/png/）
```

**組んだら全ページを画像にして目視する．**重なりとはみ出しはログに出ない．

## 決まり

1. **日本語と英語は同じ原稿**．文は `\T{日本語}{English}` で書き，`main_ja.tex`・`main_en.tex` が `\iflangja` を切り替えて同じ `deck.tex` を読む．図は共通
2. **本文の数値は手で書かない．**`scripts/export_figdata.py` が結果ファイルから `figures/numbers.tex` を書き，本文は `\N{key}` で引く（key は英字だけ．数字は zero / five / twofive，`fam1` → `famone` のように置き換わる）．無い key は組むときに止まる
3. **図の値は `figures/*.csv`** から pgfplots で描く．CSV の先頭の `#` 行に出どころを書く
4. 模式図（設定，手法の流れ，位置と向き，重みづけ，LOO）には `\notreal`（「模式図．実験の値ではない」）を付ける
5. 上限（テストで選んだ指数など）は手法の性能として出さない
6. 語は `DAI/docs/` の文書に合わせる（提案，温度つき蒸留，K 本だけ，出発点，引き戻し，教師と生徒（初出で定義），塊，単位，族，位置と向きの損）．実験番号や台本の記号（`joint`・`none`・`ret_m`）は本文に出さず，補足の対応表だけに置く
7. 途中の値（本番の台本の出力がまだ無いもの）は `\interim` の印を付け，出どころの行に「途中」と書く．本番の出力がそろったら台本を直して組み直す
8. 説明は大学 1 年生に伝わる水準で書く（グローバルの CLAUDE.md）．AI 臭い文（対句，短文の連続，比喩の動詞）を書かない

## 資料の一覧

| 資料 | 中身 |
|---|---|
| `2026-11-05-途中経過/` | 論文の骨組み（6 項目）に沿った途中経過：目的と設定，手法と比較，結果（実験 1・2），なぜ効くか（実験 3 と探索），普通の蒸留の弱点と実験 4，いま進めていること（LOO の信頼度と積／AVT の選択），まとめ．補足に詳細の表と規則 |
| `2026-11-16-共同研究者向け説明/` | 共同研究者に研究の全体を説明する資料（日本語だけ）．第 2 版（2026-11-16 午後．28 枚）：研究の全体の図（設定，前半と後半）を足し，前半を「解析の道具と素朴なやり方（なぜ声か）」，後半を「答えのまとめ方（提案）：どのコーパスで効き，なぜか，既存の方法はどこで低いか」に組み替えた．新しい図（`figures/fig_overview_study_*_ja`・`fig_fusion_map_ja`・`fig_corpus_types_*_ja`）は `DAI/paper/scripts/make_figs_overview.py`・`make_fig_corpus_types.py` が書く．ほかの図は原稿の図（`paper/figures/`）．**決まり 2 の例外**：原稿の説明なので，数値は照合を通した原稿の文と結果の文から写し，各ページに出所を書いた（写し間違いの点検は `check_numbers.md`・`check_numbers_v2.md`）．組み方は `make DECK=2026-11-16-共同研究者向け説明 LANGS=ja` |
