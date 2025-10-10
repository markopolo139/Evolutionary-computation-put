set terminal pngcairo size 1000,800 enhanced font 'Verdana,10'
set output 'solution_nn_insert.png'

set title 'Nearest neighbor considering adding the node at all possible position'
set xlabel 'X'
set ylabel 'Y'
set key outside
set size ratio -1
set grid

set datafile separator ","

stats 'points_nn_insert.csv' using 4 nooutput
cost_min = STATS_min
cost_max = STATS_max

scale(v) = 0.6 + 1.8 * ((v - cost_min) / (cost_max - cost_min))   # size range [0.6, 2.4]

set palette defined (0 "white", 1 "black")
set cblabel 'Cost'

plot \
    'points_nn_insert.csv' using 2:3:($4 - cost_min)/(cost_max - cost_min):(scale($4)) \
        with points pt 7 ps variable lc palette title 'All Nodes', \
    'solution_nn_insert.csv' using 1:2 with lines lw 2 lc rgb "blue" title 'Path', \
    'points_nn_insert.csv' using ($5==1 ? $2 : 1/0):($5==1 ? $3 : 1/0):($4 - cost_min)/(cost_max - cost_min):(scale($4)) \
        with points pt 7 ps variable lc rgb "red" title 'Selected Nodes'
