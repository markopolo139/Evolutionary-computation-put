PROBLEMS := TSPA TSPB
METHODS  := nn_back nn_insert gc random

TARGETS := $(foreach prob,$(PROBLEMS),$(patsubst %,$(prob)_solution_%.png,$(METHODS)))

.PHONY: all clean

all: $(TARGETS)

define PROBLEM_RULES
$(1)_solution_%.png: $(1)_points_%.csv $(1)_solution_%.csv plot_template.gnuplot
	gnuplot -e "INSTANCE='$(1)'" -e "METHOD='$$*'" plot_template.gnuplot

$(1)_points_%.csv $(1)_solution_%.csv: main $(1).csv
	./main $(1)

endef

# Instantiate the template rules for each problem
$(foreach prob,$(PROBLEMS),$(eval $(call PROBLEM_RULES,$(prob))))

# Rule to compile the main executable
main: main.cpp
	$(CXX) -std=c++23 -O3 -march=native -o $@ $<

# --- Housekeeping ---
clean:
	rm -v main $(TARGETS) $(foreach prob,$(PROBLEMS),$(patsubst %,$(prob)_points_%.csv,$(METHODS))) $(foreach prob,$(PROBLEMS),$(patsubst %,$(prob)_solution_%.csv,$(METHODS)))
