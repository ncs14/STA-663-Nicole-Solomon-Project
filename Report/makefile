STA_663_Final_Project.pdf: STA_663_Final_Project.ipynb
	ipython nbconvert --to latex STA_663_Final_Project.ipynb
	pdflatex STA_663_Final_Project
	pdflatex STA_663_Final_Project
	pdflatex STA_663_Final_Project

.PHONY: all clean allclean test

all: STA_663_Final_Project.pdf 

clean:
	rm -rf *aux *log *pytxcode tests/__pycache__ tests/*pyc

allclean:
	make clean
	rm -f *pdf

tests:
	py.test