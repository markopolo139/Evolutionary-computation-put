PROBLEMS := TSPA TSPB
METHODS  := msls ils_10 ils_11 ils_12 ils_13 ils_14 ils_15 ils_16 ils_17 ils_18 ils_19 ils_20

TARGETS := $(foreach prob,$(PROBLEMS),$(patsubst %,$(prob)_solution_%.png,$(METHODS)))

.PHONY: all clean

all: $(TARGETS)

define PROBLEM_RULES
$(1)_solution_%.png: $(1)_points_%.csv $(1)_solution_%.csv plot_template.gnuplot
	gnuplot -e "INSTANCE='$(1)'" -e "METHOD='$$*'" plot_template.gnuplot

$(foreach method,$(METHODS),$(1)_points_$(method).csv) $(foreach method,$(METHODS),$(1)_solution_$(method).csv): main $(1).csv
	./main $(1)
endef

$(foreach prob,$(PROBLEMS),$(eval $(call PROBLEM_RULES,$(prob))))

main: main.cpp
	$(CXX) -std=c++23 -O3 -march=native -DDONT_PRINT_LATEX -o $@ $<

clean:
	rm -v main $(TARGETS) $(foreach prob,$(PROBLEMS),$(patsubst %,$(prob)_points_%.csv,$(METHODS))) $(foreach prob,$(PROBLEMS),$(patsubst %,$(prob)_solution_%.csv,$(METHODS)))

TEX_SRCS := $(wildcard *.tex)
main.pdf: $(TEX_SRCS) $(TARGETS)
	pdflatex main.tex
