# Spitzer_4.5Micron_MasterMap
code to generate the master intrapixel sensitivity maps for the 4.5 micron spitzer channel


This code assumes you have run p1-p5 of POET (https://github.com/kevin218/POET) for all of your calibration data.
It bins, rescales, and fits a 3D spline to the 2D map data to create a smoothed master intrapixel sensitivity map for use in POET
See **insert paper here** for detailed description and example use of the map
