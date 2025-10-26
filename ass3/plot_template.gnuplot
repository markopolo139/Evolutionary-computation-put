# This script requires two variables to be passed via the command line:
#   INSTANCE: The name of the problem instance (e.g., 'TSPA')
#   METHOD:   The name of the solution method (e.g., 'nn_back')
#
# Example usage from command line:
# gnuplot -e "INSTANCE='TSPA'" -e "METHOD='nn_back'" plot_template.gnuplot

points_file   = INSTANCE . '_points_' . METHOD . '.csv'
solution_file = INSTANCE . '_solution_' . METHOD . '.csv'
output_file   = INSTANCE . '_solution_' . METHOD . '.png'
# plot_title    = 'Solution for ' . INSTANCE . ' using method: ' . METHOD

set terminal pngcairo size 1000,450 enhanced font 'Verdana,10'
set output output_file

# set title plot_title
set xlabel 'X'
set ylabel 'Y'
set key outside
set size ratio -1
set grid

set datafile separator ","

stats points_file using 4 nooutput
cost_min = STATS_min
cost_max = STATS_max

scale(v) = (cost_max - cost_min > 0) ? (0.6 + 1.8 * ((v - cost_min) / (cost_max - cost_min))) : 1.5

set palette defined (0 "white", 1 "black")
set cblabel 'Cost'

plot \
    points_file using 2:3:($4 - cost_min)/(cost_max - cost_min):(scale($4)) \
        with points pt 7 ps variable lc palette title 'All Nodes', \
    solution_file using 1:2 with lines lw 2 lc rgb "blue" title 'Path', \
    points_file using ($5==1 ? $2 : 1/0):($5==1 ? $3 : 1/0):($4 - cost_min)/(cost_max - cost_min):(scale($4)) \
        with points pt 7 ps variable lc rgb "red" title 'Selected Nodes'
