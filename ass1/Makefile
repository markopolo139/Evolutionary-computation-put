.PHONY: all
all: solution_gc.png solution_nn_insert.png solution_nn_back.png solution_random.png

solution_%.png: points_%.csv solution_%.csv %.gnuplot
	gnuplot $*.gnuplot

points_%.csv solution_%.csv: main TSPA.csv
	./main

main: main.cpp
	$(CXX) -std=c++23 -O3 -march=native -o $@ $<
