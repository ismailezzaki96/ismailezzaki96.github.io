# LaTeX Makefile
LATEXFILES = 	*.acn\
	*.alg\
	*.aux\
	*.bbl\
	*.bcf\
	*.blg\
	*.dvi\
	*.fdb_latexmk\
	*.glg\
	*.glo\
	*.gls\
	*.idx\
	*.ilg\
	*.ind\
	*.ist\
	*.loa\
	*.lof\
	.log\
	*.log\
	*.lot\
	*.maf\
	*.mtc\
	*.mtc*\
	*.nav\
	*.nlo\
	*.out\
	*.pdfsync\
	*.ps\
	*.run.xml\
	*.snm\
	*.synctex.gz\
	*.toc\
	*.vrb\
	*.xdy\
	*.tdo\
	*.fls\

filename  := "main"


cv_pdf:
	rm -f cv.pdf
	lualatex  -synctex=1 -interaction=nonstopmode cv.tex
	rm -f $(LATEXFILES)

cv_fr_pdf:
	rm -f cv_fr.pdf
	lualatex  -synctex=1 -interaction=nonstopmode cv_fr.tex
	rm -f $(LATEXFILES)

stat_pdf:
	rm -f stat.pdf
	lualatex  -synctex=1 -interaction=nonstopmode stat.tex
	rm -f $(LATEXFILES)
	
resume_pdf:
	rm -f letter of motivation.pdf
	lualatex  -synctex=1 -interaction=nonstopmode "letter of motivation.tex"
	rm -f $(LATEXFILES)
show:
	okular  ${filename}.pdf
clean:
	rm -f $(LATEXFILES)
docx:
	pandoc ${filename}.tex   -o ${filename}.docx

