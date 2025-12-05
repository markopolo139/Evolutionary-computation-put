PROBLEMS := TSPA TSPB
MEASURES := nodes edges
TYPES := best1000 bestExternal avg

# Generate list of all expected CSVs and PNGs
CSVS := $(foreach prob,$(PROBLEMS),$(foreach measure,$(MEASURES),$(foreach type,$(TYPES),$(prob)_$(measure)_$(type).csv)))
TARGETS := $(CSVS:.csv=.png)

.PHONY: all clean clean-pdf

all: $(TARGETS) main.pdf

# Generic rule for plotting
%.png: %.csv plot.gnuplot
	gnuplot -e "INPUT='$<'; OUTPUT='$@'; TITLE='$*'" plot.gnuplot

# Dependency management for CSV generation
# We use stamp files to ensure main is run only once per instance to generate all related CSVs
TSPA_%.csv: TSPA.stamp
	@:

TSPB_%.csv: TSPB.stamp
	@:

TSPA.stamp: main TSPA.csv
	./main TSPA
	touch TSPA.stamp

TSPB.stamp: main TSPB.csv
	./main TSPB
	touch TSPB.stamp

main: main.cpp
	$(CXX) -std=c++23 -O3 -march=native -DDONT_PRINT_LATEX -o $@ $<

# Rule to build the PDF report
main.pdf: main.tex algorithms.tex TSPA.tex TSPB.tex $(TARGETS)
	pdflatex main.tex
	pdflatex main.tex
	pdflatex main.tex

clean-pdf:
	rm -f *.aux *.log *.out *.toc *.nav *.snm *.fdb_latexmk *.fls *.synctex.gz *.dvi *.ps *.bbl *.blg main.pdf

clean: clean-pdf
	rm -f main $(TARGETS) $(CSVS) *.stamp *_points_*.csv *_solution_*.csv