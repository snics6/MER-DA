MAIN = main
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

PY = ../.venv/bin/python

.PHONY: all watch clean check numbers ledger

all: numbers $(MAIN).pdf

# 台帳 results/numbers.json は MER-DA の結果ファイルから scripts/ledger/ のコードで作る（手で書かない）．
# 値か出所が変わった key は一覧に出る．本文を読み直す．
ledger:
	@$(PY) scripts/build_ledger.py

# 台帳から numbers.tex（\NUM{key} の定義）を生成する．
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
#   0. 台帳が結果ファイルと一致するか（結果を作り直したのに台帳が古いままなら落ちる）
#   1. numbers.tex の生成（必須項目を欠く key があればここで落ちる）
#   2. ビルド（LaTeX エラー・未解決参照・はみ出し・頁数．TAC は参考文献・略歴込みで 12 頁）
#   3. 要旨と，序論・Part I・Part II・比較・考察・結論の節の数値が台帳（\NUM{key}）を経由しているか
#   4. 査読ループの起動漏れ（reviews/.loop-active がある間だけ働く）
check:
	@echo "--- 台帳と結果ファイル ---"
	@$(PY) scripts/build_ledger.py --check
	@python3 scripts/gen_numbers_tex.py
	@$(LATEXMK) $(MAIN).tex > /dev/null 2>&1 || true
	@echo "--- ビルド ---"
	@echo "main:  errors=$$(grep -cE '^! |:[0-9]+: .*[Ee]rror' main.log) undefined=$$(grep -icE 'undefined (reference|citation)' main.log) overfull=$$(python3 scripts/count_overfull.py main.log) pages=$$(pdfinfo main.pdf 2>/dev/null | awk '/Pages/{print $$2}')"
	@echo "--- 数値の出所 ---"
	@python3 scripts/check_numbers.py
	@echo "--- 査読ループ ---"
	@python3 scripts/loop_guard.py
