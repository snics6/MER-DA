MAIN = main
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all watch clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEXMK) $(MAIN).tex

watch:
	$(LATEXMK) -pvc $(MAIN).tex

clean:
	$(LATEXMK) -C $(MAIN).tex
	rm -f *.bbl
