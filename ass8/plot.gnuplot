# This script requires variables to be passed via the command line:
#   INPUT: The input CSV file
#   OUTPUT: The output PNG file
#   TITLE: The chart title
#
# Example usage from command line:
# gnuplot -e "INPUT='TSPA_nodes_avg.csv'; OUTPUT='TSPA_nodes_avg.png'; TITLE='TSPA Nodes Average'" plot.gnuplot

set terminal pngcairo size 800,600 enhanced font 'Verdana,10'
set output OUTPUT

set datafile separator ","

# Calculate statistics to get correlation
stats INPUT using 1:2 every ::1 name "STATS" nooutput

set title sprintf('%s\nCorrelation: %.4f', TITLE, STATS_correlation)
set xlabel 'Objective Function Value'
set ylabel 'Similarity'
set grid
set key outside

# Skip the header line
plot INPUT using 1:2 every ::1 with points pt 7 ps 0.8 lc rgb "blue" title 'Local Optima'