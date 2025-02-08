# -*- coding: utf-8 -*-
"""
Script to test the different functions of maze_generator.py, maze_solver.py
and maze_imshow.py.
"""

import numpy as np
import maze_imshow as mi

#=============================================================================

# int : size of the maze, between 2 and +inf (or at least as much as your
# computer can handle it)
size = 51

# str : algorithm used to create the maze ['fusion', 'randwalk',
# 'kurskal', 'ticking_f', 'ticking_s', 'jumper', 'hunter', 'grower',
# 'Eller', 'sidewinder', 'bintree', 'bintree_sn', 'bintree_ce',
# 'bintree_rc', 'bintree_sp']
creation = 'bintree_sp'

# bool : if some walls are broke to create loop and thus a not having an
# unique solution to the maze.
complexit = False

# list of str : from ('pre_reduc' or None) and ('right_hand',
# 'right_hand_single', 'left_hand', 'left_hand_single', or 'straight')
resolve = ['None', 'None']

# from 'all' or 'maze' or 'empty' or None.
plot = 'all'

# bool : print (or not) the time for the execution of the tasks.
timing = False

maze = mi.full_maze(size, creation, complexit, resolve, plot, timing)
