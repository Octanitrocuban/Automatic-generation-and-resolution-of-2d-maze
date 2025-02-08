# -*- coding: utf-8 -*-
"""
Script to create the mazes for the dataset and compute maze's caracteristics.
"""

import numpy as np
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt

import maze_imshow as mi
import maze_solvers as ms
import maze_generators as mg
import maze_statistics as mc

#=============================================================================

to_do = ['']

if 'maze_generate_dataset' in to_do:
	# Sizes of the mazes [11, 21, 31, ..., 91, 101]
	widths = np.arange(11, 102, 10)

	# Number of creation of the same type and size of maze
	n_runs = 1000

	# to store the connectivity stats (n sizes, n_runs, type, caracte)
	empty = np.zeros((len(widths), n_runs, 5, 4))
	empty = {}

	# to store the speed of creation of maze
	timer = np.zeros((len(widths), n_runs, 5))
	timer = {}
	to_build = ['fusion', 'randwalk', 'kurskal', 'ticking_f1', 'ticking_f10',
				'ticking_f100', 'ticking_s1', 'ticking_s10', 'ticking_s100',
				'jumper', 'hunter', 'grower_1', 'grower_2', 'grower_3',
				'grower_4', 'grower_5', 'grower_6', 'grower_7', 'grower_8',
				'grower_9', 'grower_10', 'Eller', 'sidewinder', 'bintree',
				'bintree_sn', 'bintree_ce', 'bintree_rc', 'bintree_sp']

	for i in to_build:
		empty[i] = {}
		timer[i] = {}
		for j in widths:
			empty[i][str(j)] = np.zeros((n_runs, j, j), dtype='int8')
			timer[i][str(j)] = np.zeros(n_runs, dtype='float32')

	for i in range(len(widths)):
		i_idx = str(widths[i])
		for r in tqdm(range(n_runs), desc=i_idx):
			t0 = time()
			fusion = mg.fusion(mg.create_maze_base(widths[i]))
			timer['fusion'][i_idx][r] = time()-t0

			t0 = time()
			rand_walk = mg.make_maze_exhaustif(mg.create_maze_base_boolean(widths[i]))
			timer['randwalk'][i_idx][r] = time()-t0

			t0 = time()
			min_stree = mg.kurskal_maze(widths[i])
			timer['kurskal'][i_idx][r] = time()-t0

			t0 = time()
			tickt_f1 = mg.ticking_maze(widths[i], 'fork', 1)
			timer['ticking_f1'][i_idx][r] = time()-t0

			t0 = time()
			tickt_f10 = mg.ticking_maze(widths[i], 'fork', 10)
			timer['ticking_f10'][i_idx][r] = time()-t0

			t0 = time()
			tickt_f100 = mg.ticking_maze(widths[i], 'fork', 100)
			timer['ticking_f100'][i_idx][r] = time()-t0

			t0 = time()
			tickt_s1 = mg.ticking_maze(widths[i], 'snake', 1)
			timer['ticking_s1'][i_idx][r] = time()-t0

			t0 = time()
			tickt_s10 = mg.ticking_maze(widths[i], 'snake', 10)
			timer['ticking_s10'][i_idx][r] = time()-t0

			t0 = time()
			tickt_s100 = mg.ticking_maze(widths[i], 'snake', 100)
			timer['ticking_s100'][i_idx][r] = time()-t0

			t0 = time()
			jumper = mg.jumping_explorer(mg.create_maze_base_boolean(widths[i]))
			timer['jumper'][i_idx][r] = time()-t0

			t0 = time()
			hunter = mg.hunt_and_kill(mg.create_maze_base_boolean(widths[i]))
			timer['hunter'][i_idx][r] = time()-t0

			t0 = time()
			grower_1 = mg.growing_tree(mg.create_maze_base_boolean(widths[i]), 1)
			timer['grower_1'][i_idx][r] = time()-t0

			t0 = time()
			grower_2 = mg.growing_tree(mg.create_maze_base_boolean(widths[i]), 2)
			timer['grower_2'][i_idx][r] = time()-t0

			t0 = time()
			grower_3 = mg.growing_tree(mg.create_maze_base_boolean(widths[i]), 3)
			timer['grower_3'][i_idx][r] = time()-t0

			t0 = time()
			grower_4 = mg.growing_tree(mg.create_maze_base_boolean(widths[i]), 4)
			timer['grower_4'][i_idx][r] = time()-t0

			t0 = time()
			grower_5 = mg.growing_tree(mg.create_maze_base_boolean(widths[i]), 5)
			timer['grower_5'][i_idx][r] = time()-t0

			t0 = time()
			grower_6 = mg.growing_tree(mg.create_maze_base_boolean(widths[i]), 6)
			timer['grower_6'][i_idx][r] = time()-t0

			t0 = time()
			grower_7 = mg.growing_tree(mg.create_maze_base_boolean(widths[i]), 7)
			timer['grower_7'][i_idx][r] = time()-t0

			t0 = time()
			grower_8 = mg.growing_tree(mg.create_maze_base_boolean(widths[i]), 8)
			timer['grower_8'][i_idx][r] = time()-t0

			t0 = time()
			grower_9 = mg.growing_tree(mg.create_maze_base_boolean(widths[i]), 9)
			timer['grower_9'][i_idx][r] = time()-t0

			t0 = time()
			grower_10 = mg.growing_tree(mg.create_maze_base_boolean(widths[i]), 10)
			timer['grower_10'][i_idx][r] = time()-t0

			t0 = time()
			eller = mg.Eller(mg.create_maze_base(widths[i]))
			timer['Eller'][i_idx][r] = time()-t0

			t0 = time()
			sidwd = mg.sidewinder(mg.create_maze_base(widths[i]))
			timer['sidewinder'][i_idx][r] = time()-t0

			t0 = time()
			bint = mg.binary_tree(mg.create_maze_base_boolean(widths[i]),
								  'none')
			timer['bintree'][i_idx][r] = time()-t0

			t0 = time()
			bint_sn = mg.binary_tree(mg.create_maze_base_boolean(widths[i]),
									 'snake')
			timer['bintree_sn'][i_idx][r] = time()-t0

			t0 = time()
			bint_ce = mg.binary_tree(mg.create_maze_base_boolean(widths[i]),
									 'center')
			timer['bintree_ce'][i_idx][r] = time()-t0

			t0 = time()
			bint_rc = mg.binary_tree(mg.create_maze_base_boolean(widths[i]),
									 'randcent')
			timer['bintree_rc'][i_idx][r] = time()-t0

			t0 = time()
			bint_sp = mg.binary_tree(mg.create_maze_base_boolean(widths[i]),
									 'spiral')
			timer['bintree_sp'][i_idx][r] = time()-t0

			empty['fusion'][i_idx][r] = fusion
			empty['randwalk'][i_idx][r] = rand_walk
			empty['kurskal'][i_idx][r] = min_stree
			empty['ticking_f1'][i_idx][r] = tickt_f1
			empty['ticking_f10'][i_idx][r] = tickt_f10
			empty['ticking_f100'][i_idx][r] = tickt_f100
			empty['ticking_s1'][i_idx][r] = tickt_s1
			empty['ticking_s10'][i_idx][r] = tickt_s10
			empty['ticking_s100'][i_idx][r] = tickt_s100
			empty['jumper'][i_idx][r] = jumper
			empty['hunter'][i_idx][r] = hunter
			empty['grower_1'][i_idx][r] = grower_1
			empty['grower_2'][i_idx][r] = grower_2
			empty['grower_3'][i_idx][r] = grower_3
			empty['grower_4'][i_idx][r] = grower_4
			empty['grower_5'][i_idx][r] = grower_5
			empty['grower_6'][i_idx][r] = grower_6
			empty['grower_7'][i_idx][r] = grower_7
			empty['grower_8'][i_idx][r] = grower_8
			empty['grower_9'][i_idx][r] = grower_9
			empty['grower_10'][i_idx][r] = grower_10
			empty['Eller'][i_idx][r] = eller
			empty['sidewinder'][i_idx][r] = sidwd
			empty['bintree'][i_idx][r] = bint
			empty['bintree_sn'][i_idx][r] = bint_sn
			empty['bintree_ce'][i_idx][r] = bint_ce
			empty['bintree_rc'][i_idx][r] = bint_rc
			empty['bintree_sp'][i_idx][r] = bint_sp

	#empty = np.round(empty, 12).astype('float32')
	#timer = np.round(timer, 12).astype('float32')
	np.save('mazes.npy', np.array([empty]))
	np.save('timer.npy', np.array([timer]))

if 'time_stats' in to_do:

	# Faire tableaux
	# Sizes |    11
	# ------+-------------
	#       | min
	# algo  | median +- std
	#       | avg
	#       | max
	# ------+-------

	# Sizes of the mazes [11, 21, 31, ..., 91, 101]
	widths = np.arange(11, 102, 10)

	timer = np.load('timer.npy', allow_pickle=True)[0]

	times_stats = {}
	mz_keys = list(empty.keys())
	for i in range(len(mz_keys)):
		times_stats[mz_keys[i]] = []
		for j in range(len(widths)):
			times_stats[mz_keys[i]].append([np.median(
				timer[mz_keys[i]][str(widths[j])]),
				np.std(timer[mz_keys[i]][str(widths[j])])])

		times_stats[mz_keys[i]] = np.array(times_stats[mz_keys[i]])

	np.save('timers_stats.npy', np.array([times_stats]))

if 'maze_caract' in to_do:
	# Faire tableaux
	# Sizes |    11
	# ------+-------------
	#       | min
	# algo  | median +- std
	#       | avg
	#       | max
	# ------+-------

	# Sizes of the mazes [11, 21, 31, ..., 91, 101]
	widths = np.arange(11, 102, 10)
	empty = np.load('mazes.npy', allow_pickle=True)[0]
	mazes_stats = {}
	mz_keys = list(empty.keys())
	for i in range(len(mz_keys)):
		mazes_stats[mz_keys[i]] = {}
		for j in range(len(widths)):
			nx = np.arange(1, widths[j], 2)
			g_nodes = np.meshgrid(nx, nx)
			g_nodes = np.array([np.ravel(g_nodes[1]), np.ravel(g_nodes[0])]).T
			maze_p = np.zeros((widths[j], widths[j]), dtype='uint32')
			mazes_stats[mz_keys[i]][str(widths[j])] = {}
			distri_num_c = np.zeros((1000, 4))
			distri_direc = np.zeros((1000, 4))
			other_st = np.zeros((1000, 5))
			for r in tqdm(range(1000)):
				# distribution
				distri_num_c[r] = mc.caracterisation_nco(
					empty[mz_keys[i]][str(widths[j])][r], g_nodes)

				# len_p, num_turn, max_dx, max_dy, path
				mx_grd, len_p, n_turn, mx_dx, mx_dy, maze_p = mc.path_opti_stats(
					empty[mz_keys[i]][str(widths[j])][r], maze_p)

				other_st[r, 0] = mx_grd
				other_st[r, 1] = len_p
				other_st[r, 2] = n_turn
				other_st[r, 3] = mx_dx
				other_st[r, 4] = mx_dy
				# distribution
				distri_direc[r] = mc.caracterisation_orient(
					empty[mz_keys[i]][str(widths[j])][r], g_nodes)

			maze_p = maze_p / 1000
			w_norm = np.arange(widths[j]//2)[:, np.newaxis]+np.arange(widths[j]//2)
			w_norm = (np.abs(w_norm-widths[j]//2+1)+1)/(widths[j]//2)
			maze_p = np.ravel(maze_p[1::2, 1::2] / w_norm)
			mazes_stats[mz_keys[i]][str(widths[j])]['distrib_n_co'] = distri_num_c
			mazes_stats[mz_keys[i]][str(widths[j])]['max_gradient'] = other_st[:, 0]
			mazes_stats[mz_keys[i]][str(widths[j])]['len_opt_path'] = other_st[:, 1]
			mazes_stats[mz_keys[i]][str(widths[j])]['n_turn_op'] = other_st[:, 2]
			mazes_stats[mz_keys[i]][str(widths[j])]['max_dx'] = other_st[:, 3]
			mazes_stats[mz_keys[i]][str(widths[j])]['max_dy'] = other_st[:, 4]
			mazes_stats[mz_keys[i]][str(widths[j])]['distrib_direction'] = distri_direc
			mazes_stats[mz_keys[i]][str(widths[j])]['maze_proba'] = maze_p

	np.save('maze_caracter_dict.npy', np.array([mazes_stats]))

if 'maze_caract_arr' in to_do:
	widths = np.arange(11, 102, 10)
	maze_caracter_arr = {}
	maze_caracter_dict = np.load('maze_caracter_dict.npy', allow_pickle=True)[0]
	mz_keys = list(maze_caracter_dict.keys())
	for i in range(len(mz_keys)):
		maze_caracter_arr[mz_keys[i]] = {}
		for j in range(len(widths)):
			maze_caracter_arr[mz_keys[i]][str(widths[j])] = np.zeros((1000, 13))

			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 0] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['distrib_n_co'][:, 0]
			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 1] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['distrib_n_co'][:, 1]
			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 2] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['distrib_n_co'][:, 2]
			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 3] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['distrib_n_co'][:, 3]
			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 4] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['max_gradient']
			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 5] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['len_opt_path']
			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 6] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['n_turn_op']
			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 7] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['max_dx']
			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 8] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['max_dy']
			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 9] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['distrib_direction'][:, 0]
			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 10] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['distrib_direction'][:, 1]
			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 11] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['distrib_direction'][:, 2]
			maze_caracter_arr[mz_keys[i]][str(widths[j])][:, 12] = maze_caracter_dict[mz_keys[i]][str(widths[j])]['distrib_direction'][:, 3]

	np.save('maze_caracter_arr.npy', np.array([maze_caracter_arr]))
