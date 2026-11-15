# MER-DA/DAI/slides — DAI の資料（Beamer，LuaLaTeX）の組み方．日本語版と英語版を同じ原稿から組む
#
#   make                              名前順で最後の資料を組む（<資料>/<資料>_ja.pdf と _en.pdf）
#   make DECK=2026-11-05-途中経過      資料を指定して組む
#   make data                         作図用の CSV と本文の数（figures/numbers.tex）を結果ファイルから書き出す
#   make png                          全ページを PNG にする（目視の確認用．<資料>/build/png/）
#   make list / make clean
#
# 数値は手で書かず，scripts/export_figdata.py が書く figures/numbers.tex から \N{key} で引く．

DECKS := $(sort $(patsubst %/.latexmkrc,%,$(wildcard */.latexmkrc)))
ifeq ($(origin DECK),command line)
else
DECK := $(lastword $(DECKS))
endif
PY := ../../.venv/bin/python
LATEXMK := latexmk -lualatex -interaction=nonstopmode -file-line-error -outdir=build
LANGS := ja en

.PHONY: pdf data list png clean

pdf:
	@echo "==> 組む: $(DECK)"
	@for l in $(LANGS); do \
	  (cd "$(DECK)" && $(LATEXMK) main_$$l.tex >/dev/null 2>&1 || (grep -nE '^!|:[0-9]+: ' build/main_$$l.log | head -20; exit 1)) || exit 1; \
	  cp "$(DECK)/build/main_$$l.pdf" "$(DECK)/$(DECK)_$$l.pdf"; \
	  echo "    $(DECK)/$(DECK)_$$l.pdf（$$(pdfinfo "$(DECK)/$(DECK)_$$l.pdf" | awk '/Pages/{print $$2}') ページ） errors=$$(grep -c '^!' "$(DECK)/build/main_$$l.log") undefined=$$(grep -ciE 'undefined (reference|control)|Reference .* undefined' "$(DECK)/build/main_$$l.log") overfull=$$(grep -c 'Overfull' "$(DECK)/build/main_$$l.log")"; \
	done

data:
	@cd "$(DECK)" && for f in scripts/export_*.py; do $(PY) $$f > /dev/null || exit 1; echo "    $$f：済み"; done

list:
	@for d in $(DECKS); do if [ "$$d" = "$(DECK)" ]; then echo "* $$d"; else echo "  $$d"; fi; done

png: pdf
	@rm -rf "$(DECK)/build/png" && mkdir -p "$(DECK)/build/png"
	@for l in $(LANGS); do pdftoppm -png -r 70 "$(DECK)/$(DECK)_$$l.pdf" "$(DECK)/build/png/$$l"; done
	@echo "    $(DECK)/build/png/"

clean:
	@cd "$(DECK)" && for l in $(LANGS); do latexmk -C -outdir=build main_$$l.tex >/dev/null 2>&1 || true; done
	@rm -rf "$(DECK)/build"
