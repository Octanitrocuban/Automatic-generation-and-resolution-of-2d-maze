# -*- coding: utf-8 -*-
"""
Script to compute anlyze on the generated datasets.
"""

import numpy as np
from tqdm import tqdm
from time import time

import maze_imshow as mi
import maze_solvers as ms
import maze_generators as mg
import maze_statistics as mc
#=============================================================================
to_do = ['']


# If the figures are closed (True) or not(False)
close = True

if 'plot_medians_t_generation' in to_do:
    """
    Show median evolution of maze time generation through width increasing.
    """
    logsc = True
    ylim_b = False
    legtxpad = 0.1
    # For the median speed of maze generation vs their size
    timer = np.load('../data/timers_stats.npy', allow_pickle=True)[0]
    sizes = np.arange(11, 102, 10)
    # General comparison
    keys = ['fusion', 'randwalk', 'kruskal', 'oshift_f10', 'jumper',
            'hunter', 'grower_4', 'Eller', 'sidewinder', 'bintree']

    labels = ['Fusion', 'Random walk', 'Kruskal', 'Origin shift (fork 10k)',
              'Jumping explorer', 'Hunt and kill', 'Growing tree (4)',
              'Eller', 'Sidewinder', 'Binary tree']

    locleg = [0.002, 0.535]
    xylabel = {'xlabel':'Widths of the mazes (number of cell)',
               'ylabel':'Time median (second)', 'fontsize':20}

    save_pn = '../img/log_t_evol_med_gen_all.png'
    mi.plot_median(timer, keys, sizes, labels=labels,
                   locleg=locleg, legtxpad=legtxpad,
                   logscale=logsc, xylabel=xylabel,
                   ylim_b=ylim_b, save_pn=save_pn, close=close)

    # Origin shift comparison
    keys = ['oshift_f1', 'oshift_f10', 'oshift_f100',
            'oshift_s1', 'oshift_s10', 'oshift_s100']

    labels = ['Origin shift (fork 1k)', 'Origin shift (fork 10k)',
              'Origin shift (fork 100k)', 'Origin shift (snake 1k)',
              'Origin shift (snake 10k)', 'Origin shift (snake 100k)']

    locleg = [0.002, 0.70]
    xylabel = {'xlabel':'Widths of the mazes (number of cell)',
               'ylabel':'Time median (second)', 'fontsize':20}

    save_pn = '../img/log_t_evol_med_gen_oshift.png'
    mi.plot_median(timer, keys, sizes, labels=labels,
                   locleg=locleg, legtxpad=legtxpad,
                   logscale=logsc, xylabel=xylabel,
                   ylim_b=ylim_b, save_pn=save_pn, close=close)

    # Growing tree comparison
    keys = ['grower_1', 'grower_2', 'grower_3', 'grower_4', 'grower_5',
            'grower_6', 'grower_7', 'grower_8', 'grower_9', 'grower_10']

    labels = ['Growing tree (1)', 'Growing tree (2)', 'Growing tree (3)',
              'Growing tree (4)', 'Growing tree (5)', 'Growing tree (6)',
              'Growing tree (7)', 'Growing tree (8)', 'Growing tree (9)',
              'Growing tree (10)']

    locleg = [0.002, 0.535]
    xylabel = {'xlabel':'Widths of the mazes (number of cell)',
               'ylabel':'Time median (second)', 'fontsize':20}

    save_pn = '../img/log_t_evol_med_gen_grower.png'
    mi.plot_median(timer, keys, sizes, labels=labels,
                   locleg=locleg, legtxpad=legtxpad,
                   logscale=logsc, xylabel=xylabel,
                   ylim_b=ylim_b, save_pn=save_pn, close=close)

    # Binary tree comparison
    keys = ['bintree', 'bintree_ce', 'bintree_rc', 'bintree_sn',
            'bintree_sp']

    labels = ['Binary tree', 'Binary tree (center)',
              'Binary tree (rand center)', 'Binary tree (snake)',
              'Binary tree (spiral)']

    locleg = [0.002, 0.75]
    xylabel = {'xlabel':'Widths of the mazes (number of cell)',
               'ylabel':'Time median (second)', 'fontsize':20}

    save_pn = '../img/log_t_evol_med_gen_bintree.png'
    mi.plot_median(timer, keys, sizes, labels=labels,
                   locleg=locleg, legtxpad=legtxpad,
                   logscale=logsc, xylabel=xylabel,
                   ylim_b=ylim_b, save_pn=save_pn, close=close)

if 'plot_distrib_t_generation' in to_do:
    """
    Show the distribution of the time for 101 width size maze generation.
    """
    lgsc = False
    ylim_b = True

    xylabel = {'xlabel':'Generating methods',
               'ylabel':'Time (second)', 'fontsize':20}

    # For the distribution of the speed of the maze generation for
    # 101*101 size
    timer = np.load('../data/timer.npy', allow_pickle=True)[0]
    size = '101'
    # General comparison
    keys = ['fusion', 'randwalk', 'kruskal', 'oshift_f10', 'jumper',
            'hunter', 'grower_4', 'Eller', 'sidewinder', 'bintree']

    labels = ['Fusion', 'Random\nwalk', 'Kruskal', 'Origin shift\n(fork 10k)',
              'Jumping\nexplorer', 'Hunt and\nkill', 'Growing\ntree (4)',
              'Eller', 'Sidewinder', 'Binary\ntree']

    save_pn = '../img/distrib_t_gen_all_sz101.png'
    mi.plot_distrib_1sz(timer, keys, size, querry_cc=None, labels=labels,
                        logscale=lgsc, xylabel=xylabel, ylim_b=ylim_b,
                        save_pn=save_pn, close=close)

    # Origin shift comparison
    keys = ['oshift_f1', 'oshift_f10', 'oshift_f100',
            'oshift_s1', 'oshift_s10', 'oshift_s100']

    labels = ['Origin shift\n(fork 1k)', 'Origin shift\n(fork 10k)',
              'Origin shift\n(fork 100k)', 'Origin shift\n(snake 1k)',
              'Origin shift\n(snake 10k)', 'Origin shift\n(snake 100k)']

    save_pn = '../img/distrib_t_gen_oshift_sz101.png'
    mi.plot_distrib_1sz(timer, keys, size, querry_cc=None, labels=labels,
                        logscale=lgsc, xylabel=xylabel, ylim_b=ylim_b,
                        save_pn=save_pn, close=close)

    # Growing tree comparison
    keys = ['grower_1', 'grower_2', 'grower_3', 'grower_4', 'grower_5',
            'grower_6', 'grower_7', 'grower_8', 'grower_9', 'grower_10']

    labels = ['Growing\ntree (1)', 'Growing\ntree (2)', 'Growing\ntree (3)',
              'Growing\ntree (4)', 'Growing\ntree (5)', 'Growing\ntree (6)',
              'Growing\ntree (7)', 'Growing\ntree (8)', 'Growing\ntree (9)',
              'Growing\ntree (10)']

    save_pn = '../img/distrib_t_gen_grower_sz101.png'
    mi.plot_distrib_1sz(timer, keys, size, querry_cc=None, labels=labels,
                        logscale=lgsc, xylabel=xylabel, ylim_b=ylim_b,
                        save_pn=save_pn, close=close)

    # Binary tree comparison
    keys = ['bintree', 'bintree_ce', 'bintree_rc', 'bintree_sn',
            'bintree_sp']

    labels = ['Binary tree', 'Binary tree\n(center)',
              'Binary tree\n(rand center)', 'Binary tree\n(snake)',
              'Binary tree\n(spiral)']

    save_pn = '../img/distrib_t_gen_bintree_sz101.png'
    mi.plot_distrib_1sz(timer, keys, size, querry_cc=None, labels=labels,
                        logscale=lgsc, xylabel=xylabel, ylim_b=ylim_b,
                        save_pn=save_pn, close=close)


# Size of the mazes from witch caracteristics are plot
size = '101'

# Keys used for general mazes carcteristics comparision
keys_gen = ['fusion', 'randwalk', 'kruskal', 'oshift_f10', 'jumper',
            'hunter', 'grower_4', 'Eller', 'sidewinder', 'bintree']

# Models labels used for general mazes carcteristics comparision
labels_gen = ['Fusion', 'Random\nwalk', 'Kruskal', 'Origin shift\n(fork 10k)',
              'Jumping\nexplorer', 'Hunt and\nkill', 'Growing\ntree (4)',
              'Eller', 'Sidewinder', 'Binary\ntree']

# Keys used for origin shift mazes carcteristics comparision
keys_oshift = ['oshift_f1', 'oshift_f10', 'oshift_f100', 'oshift_s1',
               'oshift_s10', 'oshift_s100']

# Models labels used for origin shift mazes carcteristics comparision
labels_oshift = ['Origin shift\n(fork 1k)', 'Origin shift\n(fork 10k)',
                 'Origin shift\n(fork 100k)', 'Origin shift\n(snake 1k)',
                 'Origin shift\n(snake 10k)', 'Origin shift\n(snake 100k)']

# Keys used for growing tree mazes carcteristics comparision
keys_grow = ['grower_1', 'grower_2', 'grower_3', 'grower_4', 'grower_5',
             'grower_6', 'grower_7', 'grower_8', 'grower_9', 'grower_10']

# Models labels used for growing tree mazes carcteristics comparision
labels_grow = ['Growing\ntree (1)', 'Growing\ntree (2)', 'Growing\ntree (3)',
               'Growing\ntree (4)', 'Growing\ntree (5)', 'Growing\ntree (6)',
               'Growing\ntree (7)', 'Growing\ntree (8)', 'Growing\ntree (9)',
               'Growing\ntree (10)']

# Keys used for binary tree mazes carcteristics comparision
keys_bint = ['bintree', 'bintree_ce', 'bintree_rc', 'bintree_sn',
             'bintree_sp']

# Models labels used for binary tree mazes carcteristics comparision
labels_bint = ['Binary tree', 'Binary tree\n(center)',
               'Binary tree\n(rand center)', 'Binary tree\n(snake)',
               'Binary tree\n(spiral)']

# Loading mazes carcteristics dataset
mazes = np.load('../data/maze_caracter_dict.npy', allow_pickle=True)[0]

if 'plot_distrib_opti_path_l' in to_do:
    """
    Show the distribution of the optimal path length for 101 width size maze.
    """
    caracte = 'len_opt_path'
    xylabel = {'xlabel':'Generating methods',
               'ylabel':'Optimal path length\n(number of cell)',
               'fontsize':20}

    # General comparison
    save_pn = '../img/distrib_opti_path_l_all_sz101.png'
    mi.plot_distrib_1sz(mazes, keys_gen, size, querry_cc=caracte,
                        labels=keys_gen, xylabel=xylabel, save_pn=save_pn,
                        close=close)

    # Origin shift comparison
    save_pn = '../img/distrib_opti_path_l_oshift_sz101.png'
    mi.plot_distrib_1sz(mazes, keys_oshift, size, querry_cc=caracte,
                        labels=labels_oshift, xylabel=xylabel,
                        save_pn=save_pn, close=close)

    # Growing tree comparison
    save_pn = '../img/distrib_opti_path_l_grower_sz101.png'
    mi.plot_distrib_1sz(mazes, keys_grow, size, querry_cc=caracte,
                        labels=labels_grow, xylabel=xylabel, save_pn=save_pn,
                        close=close)

    # Binary tree comparison
    save_pn = '../img/distrib_opti_path_l_bintree_sz101.png'
    mi.plot_distrib_1sz(mazes, keys_bint, size, querry_cc=caracte,
                        labels=labels_bint, xylabel=xylabel, save_pn=save_pn,
                        close=close)

if 'plot_distrib_opti_path_nturn' in to_do:
    """
    Show the distribution of the optimal path number of turn for 101 width
    size maze.
    """
    caracte = 'n_turn_op'
    xylabel = {'xlabel':'Generating methods',
            'ylabel':'Number of turn in the optimal\npath (number of cell)',
            'fontsize':20}

    # General comparison
    save_pn = '../img/nturn_op_dstb_gen.png'
    mi.plot_distrib_1sz(mazes, keys_gen, size, querry_cc=caracte,
                        labels=labels_gen, xylabel=xylabel, save_pn=save_pn,
                        close=close)

    # Origin shift comparison
    save_pn = '../img/nturn_op_dstb_oshift.png'
    mi.plot_distrib_1sz(mazes, keys_oshift, size, querry_cc=caracte,
                        labels=labels_oshift, xylabel=xylabel,
                        save_pn=save_pn, close=close)

    # Growing tree comparison
    save_pn = '../img/nturn_op_dstb_grower.png'
    mi.plot_distrib_1sz(mazes, keys_grow, size, querry_cc=caracte,
                        labels=labels_grow, xylabel=xylabel, save_pn=save_pn,
                        close=close)

    # Binary tree comparison
    save_pn = '../img/nturn_op_dstb_bintree.png'
    mi.plot_distrib_1sz(mazes, keys_bint, size, querry_cc=caracte,
                        labels=labels_bint, xylabel=xylabel, save_pn=save_pn,
                        close=close)

if 'plot_distrib_opti_path_occup' in to_do:
    """
    Show the distribution of the maze ground cells occupation proportion by
    the optimal path for 101 width size maze.
    """
    caracte = 'prop_grc_op'
    xylabel = {'xlabel':'Generating methods',
      'ylabel':'Maze ground cells occupation\nproportion by the optimal path',
      'fontsize':20}

    # General comparison
    save_pn = '../img/prop_grc_op_gen.png'
    mi.plot_distrib_1sz(mazes, keys_gen, size, querry_cc=caracte,
                        labels=labels_gen, xylabel=xylabel, save_pn=save_pn,
                        close=close)

    # Origin shift comparison
    save_pn = '../img/prop_grc_op_oshift.png'
    mi.plot_distrib_1sz(mazes, keys_oshift, size, querry_cc=caracte,
                        labels=labels_oshift, xylabel=xylabel,
                        save_pn=save_pn, close=close)

    # Growing tree comparison
    save_pn = '../img/prop_grc_op_grower.png'
    mi.plot_distrib_1sz(mazes, keys_grow, size, querry_cc=caracte,
                        labels=labels_grow, xylabel=xylabel, save_pn=save_pn,
                        close=close)

    # Binary tree comparison
    save_pn = '../img/prop_grc_op_bintree.png'
    mi.plot_distrib_1sz(mazes, keys_bint, size, querry_cc=caracte,
                        labels=labels_bint, xylabel=xylabel, save_pn=save_pn,
                        close=close)

if 'plot_distrib_max_gradient' in to_do:
    """
    Show the distribution of the maximal distance from the end cell for 101
    width size maze.
    """
    caracte = 'max_gradient'
    xylabel = {'xlabel':'Generating methods',
               'ylabel':'Maximum distance from the end\n(number of cells)',
               'fontsize':20}

    # General comparison
    save_pn = '../img/max_grad_gen.png'
    mi.plot_distrib_1sz(mazes, keys_gen, size, querry_cc=caracte,
                        labels=labels_gen, xylabel=xylabel, save_pn=save_pn,
                        close=close)

    # Origin shift comparison
    save_pn = '../img/max_grad_oshift.png'
    mi.plot_distrib_1sz(mazes, keys_oshift, size, querry_cc=caracte,
                        labels=labels_oshift, xylabel=xylabel,
                        save_pn=save_pn, close=close, xlprop=0.05)

    # Growing tree comparison
    save_pn = '../img/max_grad_grower.png'
    mi.plot_distrib_1sz(mazes, keys_grow, size, querry_cc=caracte,
                        labels=labels_grow, xylabel=xylabel, save_pn=save_pn,
                        close=close)

    # Binary tree comparison
    save_pn = '../img/max_grad_bintree.png'
    mi.plot_distrib_1sz(mazes, keys_bint, size, querry_cc=caracte,
                        labels=labels_bint, xylabel=xylabel, save_pn=save_pn,
                        close=close, xlprop=0.05)

if 'plot_distrib_prop_num_con_op' in to_do:
    """
    Show the distribution of the proportion of cells with 2, 3 and 4
    connections in the optimal path for 101 width size maze.
    """
    caracte = 'dsb_opti_p_dir'
    xylabel = {'xlabel':'Generating methods',
               'ylabel':'Proportion of cells with 2, 3 and 4\nconnections in the optimal path',
               'fontsize':20}

    # General comparison
    save_pn = '../img/'+caracte+'_gen.png'
    mi.plot_multidistrib_1sz(mazes, keys_gen, size, querry_cc=caracte,
                             labels=labels_gen, xylabel=xylabel,
                             save_pn=save_pn, close=close)

    # Origin shift comparison
    save_pn = '../img/'+caracte+'_oshift.png'
    mi.plot_multidistrib_1sz(mazes, keys_oshift, size, querry_cc=caracte,
                             labels=labels_oshift, xylabel=xylabel,
                             save_pn=save_pn, close=close, xlprop=0.05)

    # Growing tree comparison
    save_pn = '../img/'+caracte+'_grower.png'
    mi.plot_multidistrib_1sz(mazes, keys_grow, size, querry_cc=caracte,
                             labels=labels_grow, xylabel=xylabel,
                             save_pn=save_pn, close=close)

    # Binary tree comparison
    save_pn = '../img/'+caracte+'_bintree.png'
    mi.plot_multidistrib_1sz(mazes, keys_bint, size, querry_cc=caracte,
                             labels=labels_bint, xylabel=xylabel,
                             save_pn=save_pn, close=close, xlprop=0.05)

if 'plot_distrib_num_co' in to_do:
    """
    Show the distribution of the propotion of cells with 1 to 4 connections
    for 101 width size maze.
    """
    caracte = 'distrib_n_co'
    xylabel = {'xlabel':'Generating methods',
               'ylabel':'Propotion of cells with 1 to\n4 connections',
               'fontsize':20}

    # General comparison
    save_pn = '../img/'+caracte+'_gen.png'
    mi.plot_multidistrib_1sz(mazes, keys_gen, size, querry_cc=caracte,
                             labels=labels_gen, xylabel=xylabel,
                             save_pn=save_pn, close=close)

    # Origin shift comparison
    save_pn = '../img/'+caracte+'_oshift.png'
    mi.plot_multidistrib_1sz(mazes, keys_oshift, size, querry_cc=caracte,
                             labels=labels_oshift, xylabel=xylabel,
                             save_pn=save_pn, close=close, xlprop=0.05)

    # Growing tree comparison
    save_pn = '../img/'+caracte+'_grower.png'
    mi.plot_multidistrib_1sz(mazes, keys_grow, size, querry_cc=caracte,
                             labels=labels_grow, xylabel=xylabel,
                             save_pn=save_pn, close=close)

    # Binary tree comparison
    save_pn = '../img/'+caracte+'_bintree.png'
    mi.plot_multidistrib_1sz(mazes, keys_bint, size, querry_cc=caracte,
                             labels=labels_bint, xylabel=xylabel,
                             save_pn=save_pn, close=close, xlprop=0.05)

if 'plot_distrib_dir_co' in to_do:
    """
    Show the distribution of the proportion of the celles connection
    direction (South ; East ; North ; West) for 101 width size maze.
    """
    caracte = 'distrib_direction'
    xylabel = {'xlabel':'Generating methods',
               'ylabel':'Distribution of cells connection\ndirections',
               'fontsize':20}

    # General comparison
    save_pn = '../img/'+caracte+'_gen.png'
    mi.plot_multidistrib_1sz(mazes, keys_gen, size, querry_cc=caracte,
                             labels=labels_gen, xylabel=xylabel,
                             save_pn=save_pn, close=close)

    # Origin shift comparison
    save_pn = '../img/'+caracte+'_oshift.png'
    mi.plot_multidistrib_1sz(mazes, keys_oshift, size, querry_cc=caracte,
                             labels=labels_oshift, xylabel=xylabel,
                             save_pn=save_pn, close=close, xlprop=0.05)

    # Growing tree comparison
    save_pn = '../img/'+caracte+'_grower.png'
    mi.plot_multidistrib_1sz(mazes, keys_grow, size, querry_cc=caracte,
                             labels=labels_grow, xylabel=xylabel,
                             save_pn=save_pn, close=close)

    # Binary tree comparison
    save_pn = '../img/'+caracte+'_bintree.png'
    mi.plot_multidistrib_1sz(mazes, keys_bint, size, querry_cc=caracte,
                             labels=labels_bint, xylabel=xylabel,
                             save_pn=save_pn, close=close, xlprop=0.05)
