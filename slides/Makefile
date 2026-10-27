# MER-DA/slides — 資料（Beamer，LuaLaTeX）の組み方
#
#   make                         名前順で最後の資料を PDF に組む（<資料>/<資料>.pdf）
#   make DECK=2026-10-27-TAC論文  資料を指定して組む
#   make data                    作図用の CSV を結果ファイルから書き出す（<資料>/scripts/export_*.py を名前順に流し，付録の出所の表を作る）
#   make list                    資料の一覧
#   make png                     全ページを PNG にする（目視の確認用．<資料>/build/png/）
#   make clean                   中間生成物を消す
#
# 数値は論文の台帳から取る：各資料の main.tex が ../../paper/numbers.tex を読み込み，\NUM{key} で書く．
# 台帳を作り直したら，先に paper/ で make numbers を流す．

DECKS := $(sort $(patsubst %/.latexmkrc,%,$(wildcard */.latexmkrc)))
ifeq ($(origin DECK),command line)
else
DECK := $(lastword $(DECKS))
endif
PY := ../../.venv/bin/python
LATEXMK := latexmk -lualatex -interaction=nonstopmode -file-line-error -outdir=build

.PHONY: pdf data list png clean

pdf:
	@echo "==> 組む: $(DECK)"
	@cd "$(DECK)" && $(LATEXMK) main.tex >/dev/null 2>&1 || (grep -nE '^!|:[0-9]+: ' build/main.log | head -20; exit 1)
	@cp "$(DECK)/build/main.pdf" "$(DECK)/$(DECK).pdf"
	@echo "    $(DECK)/$(DECK).pdf（$$(pdfinfo "$(DECK)/$(DECK).pdf" | awk '/Pages/{print $$2}') ページ）"
	@echo "    errors=$$(grep -c '^!' "$(DECK)/build/main.log") undefined=$$(grep -ciE 'undefined (reference|control)|Reference .* undefined' "$(DECK)/build/main.log") overfull=$$(grep -c 'Overfull' "$(DECK)/build/main.log")"

data:
	@cd "$(DECK)" && for f in scripts/export_*.py; do $(PY) $$f > /dev/null || exit 1; echo "    $$f：済み"; done
	@cd "$(DECK)" && if [ -f tools/key_sources.py ]; then python3 tools/key_sources.py; fi

list:
	@for d in $(DECKS); do if [ "$$d" = "$(DECK)" ]; then echo "* $$d"; else echo "  $$d"; fi; done

png: pdf
	@rm -rf "$(DECK)/build/png" && mkdir -p "$(DECK)/build/png"
	@pdftoppm -png -r 80 "$(DECK)/$(DECK).pdf" "$(DECK)/build/png/p"
	@echo "    $(DECK)/build/png/"

clean:
	@cd "$(DECK)" && latexmk -C -outdir=build main.tex >/dev/null 2>&1 || true
	@rm -rf "$(DECK)/build"
