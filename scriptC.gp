set terminal png size 1600,900
set view map
set pm3d at b corners2color c4

set palette model RGB
set palette rgbformulae 7,5,15
set palette negative
set cbrange [0:1]
set size ratio -1

set xlabel "x, m^{-6}"
set zlabel "y, m^{-6}"
