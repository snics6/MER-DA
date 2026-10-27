MAIN = main
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all watch clean check numbers

all: numbers $(MAIN).pdf

# 本文に書く実験由来の数値は results/numbers.json から生成する（手で書かない）．
numbers:
	@python3 scripts/gen_numbers_tex.py

$(MAIN).pdf: $(MAIN).tex numbers.tex $(wildcard sections/*.tex) references.bib
	$(LATEXMK) $(MAIN).tex

watch:
	$(LATEXMK) -pvc $(MAIN).tex

clean:
	$(LATEXMK) -C $(MAIN).tex
	rm -f *.bbl numbers.tex sections/*.aux

# 改稿のたびに走らせる検査．
#   1. 数値台帳の生成（必須項目を欠く key があればここで落ちる）
#   2. ビルド（LaTeX エラー・未解決参照・はみ出し・頁数．TAC は参考文献・略歴込みで 12 頁）
#   3. Abstract / Results / Discussion / Conclusion の数値が台帳（\NUM{key}）を経由しているか
#   4. 査読ループの起動漏れ（reviews/.loop-active がある間だけ働く）
check:
	@python3 scripts/gen_numbers_tex.py
	@$(LATEXMK) $(MAIN).tex > /dev/null 2>&1 || true
	@echo "--- ビルド ---"
	@echo "main:  errors=$$(grep -cE '^! |:[0-9]+: .*[Ee]rror' main.log) undefined=$$(grep -icE 'undefined (reference|citation)' main.log) overfull=$$(python3 scripts/count_overfull.py main.log) pages=$$(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $$2}')"
	@echo "--- 数値の出所 ---"
	@python3 scripts/check_numbers.py
	@echo "--- 査読ループ ---"
	@python3 scripts/loop_guard.py
