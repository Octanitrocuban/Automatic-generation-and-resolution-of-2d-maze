# -*- coding: utf-8 -*-
"""
Script to test the different functions of maze_generator, maze_solver.py and
maze_imshow.py.
"""
import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

import maze_imshow as mi
import maze_solvers as ms
import maze_generators as mg

#=============================================================================

to_do = ['maze_test']

if 'maze_test' in to_do:
	# int : size of the maze, between 2 and +inf (or at least as much as your
	# computer can handle it)
	size = 51

	# str : algorithm used to create the maze ['fusion', 'randwalk', 'kurskal'
	# or 'ticking']
	creation = 'randwalk'

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

	mi.full_maze(size, creation, complexit, resolve, plot, timing)


if 'maze_stats' in to_do:
	# Sizes of the mazes
	widths = np.arange(11, 82, 10)

	# Number of creation of the same type and size of maze
	n_runs = 1000

	# to store the connectivity stats
	empty = np.zeros((len(widths), n_runs, 4, 4))
	empty_runed = np.load('connections.npy', allow_pickle=True)
	empty[:, :, :4] = empty_runed

	# to store the speed of creation of maze
	timer = np.zeros((len(widths), n_runs, 4))
	timer_runed = np.load('timer.npy', allow_pickle=True)
	timer[:, :, :4] = timer_runed

	for i in range(len(widths)):
		for r in tqdm(range(n_runs)):
			t0 = time()
			fusion = mg.maze_formation(mg.create_maze_base(widths[i]))
			timer[i, r, 0] = time()-t0

			t0 = time()
			rand_walk = mg.make_maze_exhaustif(
							mg.create_maze_base_boolean(widths[i]))

			timer[i, r, 1] = time()-t0

			t0 = time()
			min_stree = mg.kurskal_maze(widths[i])
			timer[i, r, 2] = time()-t0

			t0 = time()
			ticktack = mg.ticking_maze(widths[i])
			timer[i, r, 3] = time()-t0

			empty[i, r, 0] = mi.caracterisation(fusion)
			empty[i, r, 1] = mi.caracterisation(rand_walk)
			empty[i, r, 2] = mi.caracterisation(min_stree)
			empty[i, r, 3] = mi.caracterisation(ticktack)

	empty = np.round(empty, 12).astype('float32')
	timer = np.round(timer, 12).astype('float32')

	# marker for the median
	symb = '-'
	# size of the x and y ticks
	tksz = 18
	# size of the labels and titles
	lbsz = 20
	# size of the marker
	mksz = 6
	# positions of the size of the maze on x axis
	positions_loc = range(len(widths))

	# figures for the connectivity
	for nod in range(4):
		plt.figure(figsize=(20, 10))
		plt.title('Distribution of nodes with '+str(nod+1)+' connection',
				  fontsize=lbsz)

		plt.grid(True, zorder=1)
		plt.plot(positions_loc, np.median(empty[:, :, 0, nod], axis=1), symb,
				 color='b', label='fusion', markersize=mksz)

		parts = plt.violinplot(empty[:, :, 0, nod].T, positions_loc)
		for l in parts:
			if type(parts[l]) == list:
				for pieace in parts[l]:
					pieace.set_color('b')
					pieace.set_zorder(3)

			else:
				parts[l].set_color('b')

		plt.plot(positions_loc, np.median(empty[:, :, 1, nod], axis=1), symb,
				 color='g', label='random walk', markersize=mksz)

		parts = plt.violinplot(empty[:, :, 1, nod].T, positions_loc)
		for l in parts:
			if type(parts[l]) == list:
				for pieace in parts[l]:
					pieace.set_color('g')
					pieace.set_zorder(3)

			else:
				parts[l].set_color('g')

		plt.plot(positions_loc, np.median(empty[:, :, 2, nod], axis=1), symb,
				 color='r', label='kurskal', markersize=mksz)

		parts = plt.violinplot(empty[:, :, 2, nod].T, positions_loc)
		for l in parts:
			if type(parts[l]) == list:
				for pieace in parts[l]:
					pieace.set_color('r')
					pieace.set_zorder(3)

			else:
				parts[l].set_color('r')

		plt.plot(positions_loc, np.median(empty[:, :, 3, nod], axis=1), symb,
				 color='orange', label='ticking', markersize=mksz)

		parts = plt.violinplot(empty[:, :, 3, nod].T, positions_loc)
		for l in parts:
			if type(parts[l]) == list:
				for pieace in parts[l]:
					pieace.set_color('orange')
					pieace.set_zorder(3)

			else:
				parts[l].set_color('orange')

		plt.xticks(positions_loc, widths, fontsize=tksz)
		plt.yticks(fontsize=tksz)
		leg = plt.legend(title='Methods (median)', fontsize=lbsz,
						 title_fontsize=lbsz, markerscale=4, ncol=4)

		for lin in leg.get_lines():
			lin.set_linewidth(3)

		plt.xlabel('Width of the maze', fontsize=lbsz)
		plt.ylabel('PDF (nodes/tot nodes)', fontsize=lbsz)
		plt.savefig('distribution_of_connections_'+str(nod+1)+'.png',
					bbox_inches='tight')

		plt.close()
		plt.show()


	# figure for the time took to create the mazes
	plt.figure(figsize=(20, 10))
	plt.title('Time consumption', fontsize=lbsz)

	plt.grid(True, zorder=1)
	plt.plot(positions_loc, np.median(timer[:, :, 0], axis=1), symb,
			 color='b', label='fusion', markersize=mksz)

	parts = plt.violinplot(timer[:, :, 0].T, positions_loc)
	for l in parts:
		if type(parts[l]) == list:
			for pieace in parts[l]:
				pieace.set_color('b')
				pieace.set_zorder(3)

		else:
			parts[l].set_color('b')

	plt.plot(positions_loc, np.median(timer[:, :, 1], axis=1), symb,
			 color='g', label='random walk', markersize=mksz)

	parts = plt.violinplot(timer[:, :, 1].T, positions_loc)
	for l in parts:
		if type(parts[l]) == list:
			for pieace in parts[l]:
				pieace.set_color('g')
				pieace.set_zorder(3)

		else:
			parts[l].set_color('g')

	plt.plot(positions_loc, np.median(timer[:, :, 2], axis=1), symb,
			 color='r', label='kurskal', markersize=mksz)

	parts = plt.violinplot(timer[:, :, 2].T, positions_loc)
	for l in parts:
		if type(parts[l]) == list:
			for pieace in parts[l]:
				pieace.set_color('r')
				pieace.set_zorder(3)

		else:
			parts[l].set_color('r')

	plt.plot(positions_loc, np.median(timer[:, :, 3], axis=1), symb,
			 color='orange', label='ticking', markersize=mksz)

	parts = plt.violinplot(timer[:, :, 3].T, positions_loc)
	for l in parts:
		if type(parts[l]) == list:
			for pieace in parts[l]:
				pieace.set_color('orange')
				pieace.set_zorder(3)

		else:
			parts[l].set_color('orange')

	plt.xticks(positions_loc, widths, fontsize=tksz)
	plt.yticks(fontsize=tksz)
	leg = plt.legend(title='Methods (median)', fontsize=lbsz,
					 title_fontsize=lbsz, markerscale=4, ncol=4,
					 loc='upper left')

	for lin in leg.get_lines():
		lin.set_linewidth(3)

	plt.xlabel('Width of the maze', fontsize=lbsz)
	plt.ylabel('Time (seconds)', fontsize=lbsz)
	plt.savefig('time_contruction_methods.png', bbox_inches='tight')
	plt.close()
	plt.show()

	np.save('connections.npy', empty)
	np.save('timer.npy', timer)
