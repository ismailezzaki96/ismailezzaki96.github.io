.PHONY: main clean FORCE

main: presentation.pdf

%.pdf: FORCE
	latexmk -pdflatex='lualatex -interaction nonstopmode' -pdf $(patsubst %.pdf,%.tex,$@)
	okular  presentation.pdf
clean:
	latexmk -pdf -C
