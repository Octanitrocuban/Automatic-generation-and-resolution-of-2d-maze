# -*- coding: utf-8 -*-
"""
Script to compute anlyze on the generated datasets.
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

to_do = []


if 'plot_time' in to_do:
	to_plot = ['t_evol_general']

	if 't_general_101' in to_plot:
		timing = np.load('../data/timer.npy', allow_pickle=True)[0]
		labels = ['Fusion', 'Random\nwalking', 'Kurskal',
				  'Origin shift\n(fork 10)', 'Origin shift\n(snake 10)',
				  'Jumping\nexplorer', 'Hunt and\nkill', 'Growing\ntree (4)',
				  'Eller', 'Sidewinder', 'Binary\ntree',
				  'Binary tree\n(snake)', 'Binary tree\n(spiral)']

		clees = ['fusion', 'randwalk', 'kurskal', 'ticking_f10',
				 'ticking_s10', 'jumper', 'hunter', 'grower_4', 'Eller',
				 'sidewinder', 'bintree', 'bintree_sn', 'bintree_sp']

		mi.distribution_caract_1_size(timing, clees, labels,
									'Time distribution generate 101*101 maze',
									  '101', 'Time (seconds)',
									  savep='time_gene_101.png',
									  ylims=[-0.01, 0.4])

	if 't_all_grower_101' in to_plot:
		timing = np.load('../data/timer.npy', allow_pickle=True)[0]
		labels = ['Growing\ntree (1)', 'Growing\ntree (2)',
				  'Growing\ntree (3)', 'Growing\ntree (4)',
				  'Growing\ntree (5)', 'Growing\ntree (6)',
				  'Growing\ntree (7)', 'Growing\ntree (8)',
				  'Growing\ntree (9)', 'Growing\ntree (10)']

		clees = ['grower_1', 'grower_2', 'grower_3', 'grower_4', 'grower_5',
				 'grower_6', 'grower_7', 'grower_8', 'grower_9', 'grower_10']

		mi.distribution_caract_1_size(timing, clees, labels,
					'Time distribution generate 101*101 growing tree maze',
									  '101', 'Time (seconds)',
									  savep='time_all_grower_101.png',
									  ylims=[0.03, 0.05])

	if 't_all_oshit_101' in to_plot:
		timing = np.load('../data/timer.npy', allow_pickle=True)[0]
		labels = ['Origin shift\n(fork 1)', 'Origin shift\n(fork 10)',
				  'Origin shift\n(fork 100)', 'Origin shift\n(snake 1)',
				  'Origin shift\n(snake 10)', 'Origin shift\n(snake 100)',]

		clees = ['ticking_f1', 'ticking_f10', 'ticking_f100',
				 'ticking_s1', 'ticking_s10', 'ticking_s100']

		mi.distribution_caract_1_size(timing, clees, labels,
					'Time distribution generate 101*101 origin shift maze',
									  '101', 'Time (seconds)',
									  savep='time_all_oshift_101.png',
									  ylims=[0.05, 0.40])

	if 't_all_bint_101' in to_plot:
		timing = np.load('../data/timer.npy', allow_pickle=True)[0]
		labels = ['Binary\ntree', 'Binary tree\n(center)',
				  'Binary tree\n(random cener)', 'Binary tree\n(snake)',
				  'Binary tree\n(spiral)']

		clees = ['bintree', 'bintree_ce', 'bintree_rc', 'bintree_sn',
				 'bintree_sp']

		mi.distribution_caract_1_size(timing, clees, labels,
					'Time distribution generate 101*101 binary tree maze',
									  '101', 'Time (seconds)',
									  savep='time_all_bint_101.png',
									  ylims=[0.000, 0.015])

	if 't_evol_general' in to_plot:
		t_stats = np.load('../data/timers_stats.npy', allow_pickle=True)[0]
		widths = np.arange(11, 102, 10)

		labels = ['Fusion', 'Random walking', 'Kurskal',
				  'Origin shift (fork 10)', 'Jumping explorer',
				  'Hunt and kill', 'Growing tree (4)', 'Eller', 'Sidewinder',
				  'Binary tree']

		clees = ['fusion', 'randwalk', 'kurskal', 'ticking_f10', 'jumper',
				 'hunter', 'grower_4', 'Eller', 'sidewinder', 'bintree']

		mi.plot_one_stat(t_stats, 0, widths, clees, labels,
						 'Time median for maze generation', 'Time (second)',
						 ylog=False, savep='t_evol_med_gener.png')

		mi.plot_one_stat(t_stats, 0, widths, clees, labels,
						 'Time median for maze generation', 'Time (second)',
						 ylog=True, savep='log_t_evol_med_gener.png')

	if 't_evol_oshift' in to_plot:
		t_stats = np.load('../data/timers_stats.npy', allow_pickle=True)[0]
		widths = np.arange(11, 102, 10)

		labels = ['Origin shift (fork 1)', 'Origin shift (fork 10)',
				  'Origin shift (fork 100)', 'Origin shift (snake 1)',
				  'Origin shift (snake 10)', 'Origin shift (snake 100)']

		clees = ['ticking_f1', 'ticking_f10', 'ticking_f100', 'ticking_f1',
				 'ticking_f10', 'ticking_f100']

		mi.plot_one_stat(t_stats, 0, widths, clees, labels,
						 'Time median for maze generation', 'Time (second)',
						 ylog=False, savep='t_evol_med_oshift.png')

		mi.plot_one_stat(t_stats, 0, widths, clees, labels,
						 'Time median for maze generation', 'Time (second)',
						 ylog=True, savep='log_t_evol_med_oshift.png')

	if 't_evol_grower' in to_plot:
		t_stats = np.load('../data/timers_stats.npy', allow_pickle=True)[0]
		widths = np.arange(11, 102, 10)

		labels = ['Growing tree (1)', 'Growing tree (2)', 'Growing tree (3)',
				  'Growing tree (4)', 'Growing tree (5)', 'Growing tree (6)',
				  'Growing tree (7)', 'Growing tree (8)', 'Growing tree (9)',
				  'Growing tree (10)']

		clees = ['grower_1', 'grower_2', 'grower_3', 'grower_4', 'grower_5',
				 'grower_6', 'grower_7', 'grower_8', 'grower_9', 'grower_10']

		mi.plot_one_stat(t_stats, 0, widths, clees, labels,
						 'Time median for maze generation', 'Time (second)',
						 ylog=False, savep='t_evol_med_grower.png')

		mi.plot_one_stat(t_stats, 0, widths, clees, labels,
						 'Time median for maze generation', 'Time (second)',
						 ylog=True, savep='log_t_evol_med_grower.png')

	if 't_evol_bintree' in to_plot:
		t_stats = np.load('../data/timers_stats.npy', allow_pickle=True)[0]
		widths = np.arange(11, 102, 10)

		labels = ['Binary tree', 'Binary tree\n(center)',
				  'Binary tree\n(rand center)', 'Binary tree\n(snake)',
				  'Binary tree\n(spiral)']

		clees = ['bintree', 'bintree_ce', 'bintree_rc', 'bintree_sn',
				 'bintree_sp']

		mi.plot_one_stat(t_stats, 0, widths, clees, labels,
						 'Time median for maze generation', 'Time (second)',
						 ylog=False, savep='t_evol_med_bint.png')

		mi.plot_one_stat(t_stats, 0, widths, clees, labels,
						 'Time median for maze generation', 'Time (second)',
						 ylog=True, savep='log_t_evol_med_bint.png')

if 'plot_maze_stats' in to_do:
	# '../data/maze_caracter_arr.npy' index : 
	#	0, 1, 2, 3 : number of co (1, 2, 3, 4)  -|
	#	4 : max gradient                        -
	#	5 : optimal path length                 +
	#	6 : number of turns in the optimal path +
	#	7 : max dx                              -|
	#	8 : max dy                              -
	#	9, 10, 11, 12 : directions distribution +
	
	# '../data/stats_maze.npy' index :
	#	8 : optimal path median length          +

	to_plot = []

	if 'len_opt_di_gen' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Fusion', 'Random\nwalking', 'Kurskal',
				  'Origin shift\n(fork 10)', 'Origin shift\n(snake 10)',
				  'Jumping\nexplorer', 'Hunt and\nkill', 'Growing\ntree (4)',
				  'Eller', 'Sidewinder', 'Binary\ntree',
				  'Binary tree\n(snake)', 'Binary tree\n(spiral)']

		clees = ['fusion', 'randwalk', 'kurskal', 'ticking_f10',
				 'ticking_s10', 'jumper', 'hunter', 'grower_4', 'Eller',
				 'sidewinder', 'bintree', 'bintree_sn', 'bintree_sp']

		mi.distribution_caract_1_size(mz_sts, clees, labels,
							'Optimal path length distribution, 101*101 maze',
								'101', 'Length (number of cell)',
								index=5, savep='len_op_dstb_gen.png')

	if 'len_opt_di_oshift' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Origin shift\n(fork 1)', 'Origin shift\n(fork 10)',
				  'Origin shift\n(fork 100)', 'Origin shift\n(snake 1)',
				  'Origin shift\n(snake 10)', 'Origin shift\n(snake 100)']

		clees = ['ticking_f1', 'ticking_f10', 'ticking_f100',
				 'ticking_s1', 'ticking_s10', 'ticking_s100']

		mi.distribution_caract_1_size(mz_sts, clees, labels,
				'Optimal path length distribution, 101*101 origin shift maze',
									  '101', 'Length (number of cell)',
									  index=5, savep='len_op_dstb_oshift.png')

	if 'len_opt_di_grower' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Growing\ntree (1)', 'Growing\ntree (2)', 'Growing\ntree (3)',
				  'Growing\ntree (4)', 'Growing\ntree (5)', 'Growing\ntree (6)',
				  'Growing\ntree (7)', 'Growing\ntree (8)', 'Growing\ntree (9)',
				  'Growing\ntree (10)']

		clees = ['grower_1', 'grower_2', 'grower_3', 'grower_4', 'grower_5',
				 'grower_6', 'grower_7', 'grower_8', 'grower_9', 'grower_10']

		mi.distribution_caract_1_size(mz_sts, clees, labels,
				'Optimal path length distribution, 101*101 growing tree maze',
									  '101', 'Length (number of cell)',
									  index=5, savep='len_op_dstb_grower.png')

	if 'len_opt_di_bint' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Binary tree', 'Binary tree\n(center)',
				  'Binary tree\n(rand center)', 'Binary tree\n(snake)',
				  'Binary tree\n(spiral)']

		clees = ['bintree', 'bintree_ce', 'bintree_rc', 'bintree_sn',
				 'bintree_sp']

		mi.distribution_caract_1_size(mz_sts, clees, labels,
							'Optimal path length distribution, 101*101 maze',
									  '101', 'Length (number of cell)',
									  index=5, savep='len_op_dstb_bint.png')

	if 'nturn_opt_di_gen' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Fusion', 'Random\nwalking', 'Kurskal',
				  'Origin shift\n(fork 10)', 'Origin shift\n(snake 10)',
				  'Jumping\nexplorer', 'Hunt and\nkill', 'Growing\ntree (4)',
				  'Eller', 'Sidewinder', 'Binary\ntree',
				  'Binary tree\n(snake)', 'Binary tree\n(spiral)']

		clees = ['fusion', 'randwalk', 'kurskal', 'ticking_f10',
				 'ticking_s10', 'jumper', 'hunter', 'grower_4', 'Eller',
				 'sidewinder', 'bintree', 'bintree_sn', 'bintree_sp']

		mz_sz = '101'
		title = 'Distribution of the number of turns in the optimal path length, '
		title = title+mz_sz+'*'+mz_sz+' maze'
		ylabel = 'Length (number of cell)'

		mi.distribution_caract_1_size(mz_sts, clees, labels, title, mz_sz,
									  ylabel, index=6,
									  savep='nturn_op_dstb_gen.png')

	if 'nturn_opt_di_oshift' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Origin shift\n(fork 1)', 'Origin shift\n(fork 10)',
				  'Origin shift\n(fork 100)', 'Origin shift\n(snake 1)',
				  'Origin shift\n(snake 10)', 'Origin shift\n(snake 100)']

		clees = ['ticking_f1', 'ticking_f10', 'ticking_f100',
				 'ticking_s1', 'ticking_s10', 'ticking_s100']

		mz_sz = '101'
		title = 'Distribution of the number of turns in the optimal path length, '
		title = title+mz_sz+'*'+mz_sz+' maze'
		ylabel = 'Length (number of cell)'

		mi.distribution_caract_1_size(mz_sts, clees, labels, title, mz_sz,
									  ylabel, index=6,
									  savep='nturn_op_dstb_oshift.png')

	if 'nturn_opt_di_grower' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Growing\ntree (1)', 'Growing\ntree (2)', 'Growing\ntree (3)',
				  'Growing\ntree (4)', 'Growing\ntree (5)', 'Growing\ntree (6)',
				  'Growing\ntree (7)', 'Growing\ntree (8)', 'Growing\ntree (9)',
				  'Growing\ntree (10)']

		clees = ['grower_1', 'grower_2', 'grower_3', 'grower_4', 'grower_5',
				 'grower_6', 'grower_7', 'grower_8', 'grower_9', 'grower_10']

		mz_sz = '101'
		title = 'Distribution of the number of turns in the optimal path length, '
		title = title+mz_sz+'*'+mz_sz+' maze'
		ylabel = 'Length (number of cell)'

		mi.distribution_caract_1_size(mz_sts, clees, labels, title, mz_sz,
									  ylabel, index=6,
									  savep='nturn_op_dstb_grower.png')

	if 'nturn_opt_di_bint' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Binary tree', 'Binary tree\n(center)',
				  'Binary tree\n(rand center)', 'Binary tree\n(snake)',
				  'Binary tree\n(spiral)']

		clees = ['bintree', 'bintree_ce', 'bintree_rc', 'bintree_sn',
				 'bintree_sp']

		mz_sz = '101'
		title = 'Distribution of the number of turns in the optimal path length, '
		title = title+mz_sz+'*'+mz_sz+' maze'
		ylabel = 'Length (number of cell)'

		mi.distribution_caract_1_size(mz_sts, clees, labels, title, mz_sz,
									  ylabel, index=6,
									  savep='nturn_op_dstb_bint.png')

	if 'nco_nodes_dstb_gen' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Fusion', 'Random\nwalking', 'Kurskal',
				  'Origin shift\n(fork 10)', 'Origin shift\n(snake 10)',
				  'Jumping\nexplorer', 'Hunt and\nkill', 'Growing\ntree (4)',
				  'Eller', 'Sidewinder', 'Binary\ntree',
				  'Binary tree\n(snake)', 'Binary tree\n(spiral)']

		clees = ['fusion', 'randwalk', 'kurskal', 'ticking_f10',
				 'ticking_s10', 'jumper', 'hunter', 'grower_4', 'Eller',
				 'sidewinder', 'bintree', 'bintree_sn', 'bintree_sp']

		wid = '101'
		idx = np.array([0, 1, 2, 3])
		c_names = ['1 connection', '2 connection',
				   '3 connection', '4 connection']

		title = 'Distribution of the number of connections of the nodes, '
		title = title+wid+'*'+wid+' maze'
		ylabel = 'Proportion'

		mi.distribution_caract_multip(mz_sts, clees, labels, title, wid,
									  ylabel, idx, c_names,
									  savep='connect_numb_distri_gen.png',
									  figsize=(20, 8), ttl_sz=18, label_sz=18,
									  tck_sz=14, ylims=None)

	if 'nco_nodes_dstb_oshift' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Origin shift\n(fork 1)', 'Origin shift\n(fork 10)',
				  'Origin shift\n(fork 100)', 'Origin shift\n(snake 1)',
				  'Origin shift\n(snake 10)', 'Origin shift\n(snake 100)']

		clees = ['ticking_f1', 'ticking_f10', 'ticking_f100',
				 'ticking_s1', 'ticking_s10', 'ticking_s100']

		wid = '101'
		idx = np.array([0, 1, 2, 3])
		c_names = ['1 connection', '2 connection',
				   '3 connection', '4 connection']

		title = 'Distribution of the number of connections of the nodes, '
		title = title+wid+'*'+wid+' maze'
		ylabel = 'Proportion'

		mi.distribution_caract_multip(mz_sts, clees, labels, title, wid,
									  ylabel, idx, c_names,
									  savep='connect_numb_distri_oshift.png',
									  figsize=(20, 8), ttl_sz=18, label_sz=18,
									  tck_sz=14, dstb_w=0.2, ylims=None)

	if 'nco_nodes_dstb_grower' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Growing\ntree (1)', 'Growing\ntree (2)', 'Growing\ntree (3)',
				  'Growing\ntree (4)', 'Growing\ntree (5)', 'Growing\ntree (6)',
				  'Growing\ntree (7)', 'Growing\ntree (8)', 'Growing\ntree (9)',
				  'Growing\ntree (10)']

		clees = ['grower_1', 'grower_2', 'grower_3', 'grower_4', 'grower_5',
				 'grower_6', 'grower_7', 'grower_8', 'grower_9', 'grower_10']

		wid = '101'
		idx = np.array([0, 1, 2, 3])
		c_names = ['1 connection', '2 connection',
				   '3 connection', '4 connection']

		title = 'Distribution of the number of connections of the nodes, '
		title = title+wid+'*'+wid+' maze'
		ylabel = 'Proportion'

		mi.distribution_caract_multip(mz_sts, clees, labels, title, wid,
									  ylabel, idx, c_names,
									  savep='connect_numb_distri_grower.png',
									  figsize=(20, 8), ttl_sz=18, label_sz=18,
									  tck_sz=14, dstb_w=0.35, ylims=None,
									  loc_leg='center right')

	if 'nco_nodes_dstb_bint' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Binary tree', 'Binary tree\n(center)',
				  'Binary tree\n(rand center)', 'Binary tree\n(snake)',
				  'Binary tree\n(spiral)']

		clees = ['bintree', 'bintree_ce', 'bintree_rc', 'bintree_sn',
				 'bintree_sp']

		wid = '101'
		idx = np.array([0, 1, 2, 3])
		c_names = ['1 connection', '2 connection',
				   '3 connection', '4 connection']

		title = 'Distribution of the number of connections of the nodes, '
		title = title+wid+'*'+wid+' maze'
		ylabel = 'Proportion'

		mi.distribution_caract_multip(mz_sts, clees, labels, title, wid,
									  ylabel, idx, c_names,
									  savep='connect_numb_distri_bint.png',
									  figsize=(20, 8), ttl_sz=18, label_sz=18,
									  tck_sz=14, dstb_w=0.3, ylims=None,
									  loc_leg=[0.87, 0.6])

	if 'distrib_orient_gen' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Fusion', 'Random\nwalking', 'Kurskal',
				  'Origin shift\n(fork 10)', 'Origin\nshift\n(snake 10)',
				  'Jumping\nexplorer', 'Hunt\nand kill', 'Growing\ntree (4)',
				  'Eller', 'Sidewinder', 'Binary\ntree',
				  'Binary\ntree\n(snake)', 'Binary\ntree\n(spiral)']

		clees = ['fusion', 'randwalk', 'kurskal', 'ticking_f10',
				 'ticking_s10', 'jumper', 'hunter', 'grower_4', 'Eller',
				 'sidewinder', 'bintree', 'bintree_sn', 'bintree_sp']

		wid = '101'
		idx = np.array([9, 10, 11, 12])
		orient = ['N', 'S', 'E', 'W']
		color = ['steelblue', 'darkorange', 'green', 'red']
		ylabel = 'Node orientation cumulative proportion'
		title = 'Connection orientation distribution (101*101)'

		mi.show_distrib_norm(mz_sts, clees, labels, wid, idx, orient, color,
							 title=title, ylabel=ylabel, text_sz=15,
							 savep='connect_orient_distri_gen.png')

	if 'distrib_orient_grower' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Growing\ntree (1)', 'Growing\ntree (2)',
				  'Growing\ntree (3)', 'Growing\ntree (4)',
				  'Growing\ntree (5)', 'Growing\ntree (6)',
				  'Growing\ntree (7)', 'Growing\ntree (8)',
				  'Growing\ntree (9)', 'Growing\ntree (10)']

		clees = ['grower_1', 'grower_2', 'grower_3', 'grower_4', 'grower_5',
				 'grower_6', 'grower_7', 'grower_8', 'grower_9', 'grower_10']

		wid = '101'
		idx = np.array([9, 10, 11, 12])
		orient = ['N', 'S', 'E', 'W']
		color = ['steelblue', 'darkorange', 'green', 'red']
		ylabel = 'Node orientation cumulative proportion'
		title = 'Connection orientation distribution (101*101)'

		mi.show_distrib_norm(mz_sts, clees, labels, wid, idx, orient, color,
							 title=title, ylabel=ylabel, text_sz=15,
							 savep='connect_orient_distri_grower.png')

	if 'distrib_orient_oshit' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Origin shift\n(fork 1)', 'Origin shift\n(fork 10)',
				  'Origin shift\n(fork 100)', 'Origin shift\n(snake 1)',
				  'Origin shift\n(snake 10)', 'Origin shift\n(snake 100)']

		clees = ['ticking_f1', 'ticking_f10', 'ticking_f100',
				 'ticking_s1', 'ticking_s10', 'ticking_s100']

		wid = '101'
		idx = np.array([9, 10, 11, 12])
		orient = ['N', 'S', 'E', 'W']
		color = ['steelblue', 'darkorange', 'green', 'red']
		ylabel = 'Node orientation cumulative proportion'
		title = 'Connection orientation distribution (101*101)'

		mi.show_distrib_norm(mz_sts, clees, labels, wid, idx, orient, color,
							 title=title, ylabel=ylabel, text_sz=15,
							 figsize=(14, 8), lm_prop=0.06, 
							 savep='connect_orient_distri_oshift.png')

	if 'distrib_orient_bint' in to_plot:
		mz_sts = np.load('../data/maze_caracter_arr.npy',
						 allow_pickle=True)[0]

		labels = ['Binary tree', 'Binary tree\n(center)',
				  'Binary tree\n(rand center)', 'Binary tree\n(snake)',
				  'Binary tree\n(spiral)']

		clees = ['bintree', 'bintree_ce', 'bintree_rc', 'bintree_sn',
				 'bintree_sp']

		wid = '101'
		idx = np.array([9, 10, 11, 12])
		orient = ['N', 'S', 'E', 'W']
		color = ['steelblue', 'darkorange', 'green', 'red']
		ylabel = 'Node orientation cumulative proportion'
		title = 'Connection orientation distribution (101*101)'

		mi.show_distrib_norm(mz_sts, clees, labels, wid, idx, orient, color,
							 title=title, ylabel=ylabel, text_sz=15,
							 figsize=(14, 8), lm_prop=0.07,
							 savep='connect_orient_distri_bint.png')

	if 'len_opti_general' in to_plot:
		m_stats = np.load('../data/stats_maze.npy', allow_pickle=True)[0]
		widths = np.arange(11, 102, 10)
		labels = ['Fusion', 'Random walking', 'Kurskal',
				  'Origin shift (fork 10)', 'Origin shift (snake 10)',
				  'Jumping explorer', 'Hunt and kill', 'Growing tree (4)',
				  'Eller', 'Sidewinder', 'Binary tree',
				  'Binary tree (snake)', 'Binary tree (spiral)']

		clees = ['fusion', 'randwalk', 'kurskal', 'ticking_f10',
				 'ticking_s10', 'jumper', 'hunter', 'grower_4', 'Eller',
				 'sidewinder', 'bintree', 'bintree_sn', 'bintree_sp']

		mi.plot_one_stat(data_stats, axis, widths, keys, labels, title, ylabel,
				  symb='.-', prop_lm=0.02, figsize=(20, 8), ttl_sz=18,
				  label_sz=18, tck_sz=14, leg_sz=14, ylog=False, savep=None)
		mi.plot_one_stat(m_stats, 8, widths, clees, labels,
						 'Median length for maze optimal path',
						 'Length (num cells)',
						 ylog=False, savep='len_evol_med_gen.png')

		mi.plot_one_stat(m_stats, 8, widths, clees, labels,
						 'Median length for maze optimal path',
						 'Length (num cells)',
						 ylog=True, savep='log_len_evol_med_gen.png')

	if 'len_opti_grower' in to_plot:
		m_stats = np.load('../data/stats_maze.npy', allow_pickle=True)[0]
		widths = np.arange(11, 102, 10)
		labels = ['Growing tree (1)', 'Growing tree (2)', 'Growing tree (3)',
				  'Growing tree (4)', 'Growing tree (5)', 'Growing tree (6)',
				  'Growing tree (7)', 'Growing tree (8)', 'Growing tree (9)',
				  'Growing tree (10)']

		clees = ['grower_1', 'grower_2', 'grower_3', 'grower_4', 'grower_5',
				 'grower_6', 'grower_7', 'grower_8', 'grower_9', 'grower_10']

		mi.plot_one_stat(m_stats, 8, widths, clees, labels,
						 'Median length for growing tree maze optimal path',
						 'Length (num cells)',
						 ylog=False, savep='len_evol_med_grower.png')

		mi.plot_one_stat(m_stats, 8, widths, clees, labels,
						 'Median length for growing tree maze optimal path',
						 'Length (num cells)',
						 ylog=True, savep='log_len_evol_med_grower.png')

	if 'len_opti_oshift' in to_plot:
		m_stats = np.load('../data/stats_maze.npy', allow_pickle=True)[0]
		widths = np.arange(11, 102, 10)
		labels = ['Origin shift (fork 1)', 'Origin shift (fork 10)',
				  'Origin shift (fork 100)', 'Origin shift (snake 1)',
				  'Origin shift (snake 10)', 'Origin shift (snake 100)']

		clees = ['ticking_f1', 'ticking_f10', 'ticking_f100', 'ticking_f1',
				 'ticking_f10', 'ticking_f100']

		mi.plot_one_stat(m_stats, 8, widths, clees, labels,
						 'Median length for origin shift maze optimal path',
						 'Length (num cells)',
						 ylog=False, savep='len_evol_med_oshift.png')

		mi.plot_one_stat(m_stats, 8, widths, clees, labels,
						 'Median length for origin shift maze optimal path',
						 'Length (num cells)',
						 ylog=True, savep='log_len_evol_med_oshift.png')

	if 'len_opti_bint' in to_plot:
		m_stats = np.load('../data/stats_maze.npy', allow_pickle=True)[0]
		widths = np.arange(11, 102, 10)
		labels = ['Binary tree', 'Binary tree (center)',
				  'Binary tree (rand cent)', 'Binary tree (snake)',
				  'Binary tree (spiral)']

		clees = ['bintree', 'bintree_ce', 'bintree_rc', 'bintree_sn',
				 'bintree_sp']

		mi.plot_one_stat(m_stats, 8, widths, clees, labels,
						 'Median length for binary tree maze optimal path',
						 'Length (num cells)',
						 ylog=False, savep='len_evol_med_bint.png')

		mi.plot_one_stat(m_stats, 8, widths, clees, labels,
						 'Median length for binary tree maze optimal path',
						 'Length (num cells)',
						 ylog=True, savep='log_len_evol_med_bint.png')
