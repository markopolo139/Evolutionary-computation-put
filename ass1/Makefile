.PHONY: all
all: solution_gc.png solution_nn_insert.png solution_nn_back.png solution_random.png

cpp/ass1: cpp/ass1.cpp cpp/Makefile
	make -C cpp

solution_%.png: points_%.csv solution_%.csv %.gnuplot
	gnuplot $*.gnuplot

points_%.csv solution_%.csv: cpp/ass1 TSPA.csv
	make -C cpp run