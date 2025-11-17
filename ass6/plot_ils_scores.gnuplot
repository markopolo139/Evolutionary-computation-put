set terminal pngcairo size 1000,450 enhanced font 'Verdana,10'
set output 'ils_scores.png'

set xlabel 'Perturbation Strength'
set ylabel 'Avg Score (TSPA)'
set y2label 'Avg Score (TSPB)'

set key outside
set grid

set datafile separator ","

set ytics nomirror
set y2tics

plot 'ils_scores.csv' using 1:2 with linespoints title 'TSPA' lc rgb "blue", \
     '' using 1:3 with linespoints title 'TSPB' lc rgb "red" axes x1y2


