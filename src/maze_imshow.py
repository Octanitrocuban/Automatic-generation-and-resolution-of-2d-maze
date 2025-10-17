# -*- coding: utf-8 -*-
"""
This script contain the functions to show the mazes, their solution path, and
the distance from ground nods to the exit nod.
"""
#import usefull library
import numpy as np
import matplotlib.pyplot as plt
import maze_generators as mg
import maze_solvers as ms
#=============================================================================
def show_maze(maze_map, path=None, gradient=None):
    """
    Function to show the maze once it is build. It is possible to add the
    path from start to exit and the gradient of the distance of the ground
    nods to the exit nod.

    Parameters
    ----------
    maze_map : np.ndarray
        2 dimensions numpy array of the maze. This array will have -1 for
        wall nods, and 0 for ground nods.
    path : np.ndarray, optional
        2 dimensions numpy array of the path that is leading from the start
        to the exit of the maze. The default is None.
    gradient : np.ndarray, optional
        2 dimensions numpy array of the gradient of the maze. This array
        will have -1 for wall nods, and from 1 to N for ground nods. N is
        the maximum number of nods between the a ground nod and the exit
        nod. The default is None.

    Returns
    -------
    maze_map : np.ndarray
        2 dimensions numpy array of the maze. This array will have -1 for
        wall nods, and 0 for ground nods.

    """
    if len(maze_map) <= 29:
        marker = "wo-"

    elif (len(maze_map) > 29) & (len(maze_map) <= 72):
        marker = "k.-"

    elif (len(maze_map) > 72) & (len(maze_map) <= 99):
        marker = "m.-"

    else:
        marker = "r-"

    # experimental equation for the size of the figure
    fig_wh = 10+(len(maze_map)/2)**0.48

    plt.figure(figsize=(fig_wh, fig_wh))
    plt.imshow(maze_map, cmap='binary_r')
    if type(gradient) == np.ndarray:
        plt.imshow(gradient, cmap="jet")

    if type(path) == np.ndarray:
        plt.plot(path[:, 1], path[:, 0], marker)

    plt.axis('off')
    plt.show()
    return maze_map

def full_maze(size, creation, complexit, resolve, plot, timing=True):
    """
    Automatic generation and resolution of mazes with the method in
    MazeGenerators.py and MazeSolvers.py. Show of the maze at different step
    is also possible through the variable plot. It is also possible to know
    the time took by the step throug the variable Timing.

    Parameters
    ----------
    size : int
        Length of the square of the maze. It must be stricly superior at 1.
    creation : str
        From ['fusion', 'randwalk', 'kruskal', 'oshift_f', 'oshift_s',
        'jumper', 'hunter', 'grower', 'Eller', 'sidewinder', 'bintree',
        'bintree_sn', 'bintree_ce', 'bintree_rc', 'bintree_sp']
    complexit : bool
        If True applied complexification function to the maze.
    resolve : list of string
        From ('pre_reduc' or None) and ('right_hand', 'right_hand_single',
        'left_hand', 'left_hand_single', or 'straight').
    plot : string
        from 'all' or 'maze' or 'empty' or None.
    timing : Bool, optional
        Print the time for the execution of the tasks. The default is True.

    Raises
    ------
    TypeError
        size type isn't an int.
    ValueError
        size is inferior at 2.
    TypeError
        creation type isn't a list.
    SyntaxError
        No method for maze generation put in creation list.
    TypeError
        plot type isn't a string.
    TypeError
        resolve type isn't a list.
    TypeError
        timing type isn't a bool.

    Returns
    -------
    plat : 2d numpy array of int
        The map of the maze with -1 for walls and 0 for ground.

    """
    possible_creation = ['randwalk', 'fusion', 'kruskal', 'oshift_f',
                         'oshift_s', 'jumper', 'hunter', 'grower', 'Eller',
                         'sidewinder', 'bintree', 'bintree_sn', 'bintree_ce',
                         'bintree_rc', 'bintree_sp']

    #Checking types&others of parameters
    if type(size) != int:
        raise TypeError("'size' must be list type, found: "+str(type(size)))

    if size < 2:
        raise ValueError("'size' must be superior or equal at 2, found: "+
                         str(size))

    if type(creation) != str:
        raise TypeError("'creation' must be a str, found: "+
                         str(type(creation)))

    if creation not in possible_creation:
        raise SyntaxError("You haven't specify the method of creation"+
                          " of the maze. Choose it in: 'randwalk'/'fusion'")

    if type(complexit) != bool:
        raise SyntaxError("'complexit' must be bool type. Find : "+
                          str(type(complexit)))

    if type(plot) != str:
        raise TypeError("'plot' must be of str, find: "+str(type(plot)))

    if type(resolve) != list:
        raise TypeError("'resolve' must be list of str, find: "+
                        str(type(resolve)))
        
    if type(timing) != bool:
        raise TypeError("'Timing' must be bool, find: "+ str(type(timing)))

    if timing:
        tini = datetime.now()

    if "randwalk" in creation:
        plat = mg.create_maze_base_boolean(size)
        plat = mg.make_maze_exhaustif(plat)

    elif "fusion" in creation:
        plat = mg.create_maze_base(size)
        plat = mg.fusion(plat)

    elif "kruskal" in creation:
        plat = mg.kruskal_maze(size)

    elif 'oshift_f' in creation:
        plat = mg.origin_shift(size)

    elif 'oshift_s' in creation:
        plat = mg.origin_shift(size, 'snake')

    elif 'jumper' in creation:
        plat = mg.create_maze_base_boolean(size)
        plat = mg.jumping_explorer(plat)

    elif 'hunter' in creation:
        plat = mg.create_maze_base_boolean(size)
        plat = mg.hunt_and_kill(plat)

    elif 'grower' in creation:
        plat = mg.create_maze_base_boolean(size)
        plat = mg.growing_tree(plat)

    elif 'Eller' in creation:
        plat = mg.create_maze_base(size)
        plat = mg.Eller(plat)

    elif 'sidewinder' in creation:
        plat = mg.create_maze_base(size)
        plat = mg.sidewinder(plat)

    elif creation == 'bintree':
        plat = mg.create_maze_base_boolean(size)
        plat = mg.binary_tree(plat)

    elif creation == 'bintree_sn':
        plat = mg.create_maze_base_boolean(size)
        plat = mg.binary_tree(plat, 'snake')

    elif creation == 'bintree_ce':
        plat = mg.create_maze_base_boolean(size)
        plat = mg.binary_tree(plat, 'center')

    elif creation == 'bintree_rc':
        plat = mg.create_maze_base_boolean(size)
        plat = mg.binary_tree(plat, 'randcent')

    elif creation == 'bintree_sp':
        plat = mg.create_maze_base_boolean(size)
        plat = mg.binary_tree(plat, 'spiral')

    if complexit:
        plat = mg.make_multipathe(plat)

    if timing:
        tfin = datetime.now()
        print("Time to build the maze =", tfin-tini)

    if (plot == "empty")|(plot == "all"):
        show_maze(plat)

    if "straight" in resolve:
        if timing:
            t0 = datetime.now()

        solution = ms.maze_gradient(plat)
        desc = ms.descente_grad_maze(solution)
        solution = np.array(solution, dtype=float)
        solution[solution == -1] = np.nan
        if timing:
            t1 = datetime.now()
            print("Time to solves the maze =", str(str(t1-t0)))

        if (plot == "maze")|(plot == "all"):
            show_maze(plat, desc, solution)

    elif "pre_reduc" in resolve:
        if timing:
            t0 = datetime.now()

        reduit = ms.maze_reduction(plat)
        solution = ms.maze_gradient(reduit)
        desc = ms.descente_grad_maze(solution)
        solution = np.array(solution, dtype=float)
        solution[solution == -1] = np.nan
        if timing:
            t1 = datetime.now()
            print("Time to solves the maze =", str(str(t1-t0)))

        if (plot == "maze")|(plot == "all"):
            show_maze(plat, desc, solution)

    elif "right_hand" in resolve:
        if timing:
            t0 = datetime.now()

        solution = ms.wall_hand_solve(plat, 'R')
        
        if timing:
            t1 = datetime.now()
            print("Time to solves the maze =", str(str(t1-t0)))

        if (plot == "maze")|(plot == "all"):
            show_maze(plat, solution)

    elif "left_hand" in resolve:
        if timing:
            t0 = datetime.now()

        solution = ms.wall_hand_solve(plat, 'L')
        if timing:
            t1 = datetime.now()
            print("Time to solves the maze =", str(str(t1-t0)))

        if (plot == "maze")|(plot == "all"):
            show_maze(plat, solution)

    elif "right_hand_single" in resolve:
        if timing:
            t0 = datetime.now()

        solution = ms.wall_hand_solve(plat, 'R')
        solution = ms.tri_hand_solve_path(solution)
        if timing:
            t1 = datetime.now()
            print("Time to solves the maze =", str(str(t1-t0)))

        if (plot == "maze")|(plot == "all"):
            show_maze(plat, solution)

    elif "left_hand_single" in resolve:
        if timing:
            t0 = datetime.now()

        solution = ms.wall_hand_solve(plat, 'L')
        solution = ms.tri_hand_solve_path(solution)
        if timing:
            t1 = datetime.now()
            print("Time to solves the maze =", str(str(t1-t0)))

        if (plot == "maze")|(plot == "all"):
            show_maze(plat, solution)

    return plat

def plot_limits(a, p):
    """
    """
    mini = np.min(a)
    maxi = np.max(a)
    extent = (maxi-mini)*p
    limits = (mini-extent, maxi+extent)
    return limits

def plot_median(dict_data, keys, sizes, markers=None, labels=None,
                figsize=(12, 6), dpi=200, locleg=None, legtxpad=0.8,
                logscale=False, grid=True, xylabel=None, ylim_b=True,
                save_pn=None, close=False):
    """
    Ajouter : - title
    """
    if type(markers) == type(None):
        markers = ['.']*len(keys)
    elif len(markers) < len(keys):
        raise ValueError('dimension mismatch ! len(markers) < len(keys) !')

    xlims = plot_limits(sizes, 0.04)

    yarr = []
    plt.figure(figsize=figsize, dpi=dpi)
    plt.grid(grid, which='both', zorder=1)
    if type(labels) == type(None):
        for i, ky in enumerate(keys):
            yarr.append(dict_data[ky][:, 0])
            plt.plot(sizes, dict_data[ky][:, 0], markers[i]+'-')

    else:
        if len(labels) < len(keys):
            raise ValueError('dimension mismatch ! len(labels) < len(keys) !')

        for i, ky in enumerate(keys):
            yarr.append(dict_data[ky][:, 0])
            plt.plot(sizes, dict_data[ky][:, 0], markers[i]+'-',
                     label=labels[i])

        plt.legend(loc=locleg, handletextpad=legtxpad)

    ylims = plot_limits(np.ravel(np.array(yarr)), 0.04)
    if type(xylabel) == dict:
        if 'xlabel' not in list(xylabel.keys()):
            raise ValueError('Missing `xlabel` in xylabel dict !')
        if 'ylabel' not in list(xylabel.keys()):
            raise ValueError('Missing `ylabel` in xylabel dict !')
        if 'fontsize' not in list(xylabel.keys()):
            raise ValueError('Missing `fontsize` in xylabel dict !')

        plt.xlabel(xylabel['xlabel'], fontsize=xylabel['fontsize'])
        plt.ylabel(xylabel['ylabel'], fontsize=xylabel['fontsize'])

    plt.xlim(xlims[0], xlims[1])
    if ylim_b:
        plt.ylim(ylims[0], ylims[1])

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if logscale:
        plt.yscale('log')

    if type(save_pn) == str:
        plt.savefig(save_pn, bbox_inches='tight')

    if close:
        plt.close()

    plt.show()

def plot_distrib_1sz(dict_data, keys, size, querry_cc=None, labels=None,
                     fgsz=(12, 6), dpi=200, logscale=False, grid=True,
                     xylabel=None, ylim_b=True, save_pn=None, close=False,
                     xlprop=0.04):
    """
    Function to .

    Parameters
    ----------
    dict_data : dict
        .
    keys : list
        .
    size : str
        .
    querry_cc : , optional
        . The default value is None.
    labels : list, optional
        . The default value is None.
    fgsz : tuple, optional
        . The default value is (12, 6).
    dpi : int, optional
        . The default value is 200.
    logscale : bool, optional
        . The default value is False.
    grid : bool, optional
        . The default value is True.
    xylabel : , optional
        . The default value is None.
    ylim_b : bool, optional
        . The default value is True.
    save_pn : , optional
        . The default value is None.
    close : bool, optional
        . The default value is False.
    xlprop : float, optional
        . The default value is 0.04.

    Returns
    -------
    None

    """
    if len(labels) < len(keys):
        raise ValueError('dimension mismatch ! len(labels) < len(keys) !')

    if len(labels) > len(keys):
        raise ValueError('dimension mismatch ! len(labels) > len(keys) !')

    yarr = []
    xlims = plot_limits(np.array([0, len(keys)-1]), xlprop)

    plt.figure(figsize=fgsz, dpi=dpi)
    if grid:
        plt.grid(grid, which='both', zorder=1)

    if type(querry_cc) == str:
        for i, ky in enumerate(keys):
            plt.violinplot(dict_data[ky][size][querry_cc], [i],
                           showmedians=True)

            yarr.append([np.min(dict_data[ky][size][querry_cc]),
                         np.max(dict_data[ky][size][querry_cc])])

    else:
        for i, ky in enumerate(keys):
            plt.violinplot(dict_data[ky][size], [i], showmedians=True)
            yarr.append([np.min(dict_data[ky][size]),
                         np.max(dict_data[ky][size])])

    if type(xylabel) == dict:
        if 'xlabel' not in list(xylabel.keys()):
            raise ValueError('Missing `xlabel` in xylabel dict !')
        if 'ylabel' not in list(xylabel.keys()):
            raise ValueError('Missing `ylabel` in xylabel dict !')
        if 'fontsize' not in list(xylabel.keys()):
            raise ValueError('Missing `fontsize` in xylabel dict !')

        plt.xlabel(xylabel['xlabel'], fontsize=xylabel['fontsize'])
        plt.ylabel(xylabel['ylabel'], fontsize=xylabel['fontsize'])

    plt.xticks(np.arange(len(keys)), labels, fontsize=13)
    plt.yticks(fontsize=14)
    plt.xlim(xlims[0], xlims[1])
    if ylim_b:
        ylims = plot_limits(np.ravel(np.array(yarr)), 0.04)
        plt.ylim(ylims[0], ylims[1])

    if logscale:
        plt.yscale('log')

    if type(save_pn) == str:
        plt.savefig(save_pn, bbox_inches='tight')

    if close:
        plt.close()

    plt.show()

def plot_multidistrib_1sz(dict_data, keys, size, querry_cc=None, labels=None,
                          fgsz=(12, 6), dpi=200, logscale=False, grid=True,
                          xylabel=None, ylim_b=True, save_pn=None,
                          close=False, xlprop=0.04):
    """
    Function to .

    Parameters
    ----------
    dict_data : dict
        .
    keys : list
        .
    size : str
        .
    querry_cc : , optional
        . The default value is None.
    labels : list, optional
        . The default value is None.
    fgsz : tuple, optional
        . The default value is (12, 6).
    dpi : int, optional
        . The default value is 200.
    logscale : bool, optional
        . The default value is False.
    grid : bool, optional
        . The default value is True.
    xylabel : , optional
        . The default value is None.
    ylim_b : bool, optional
        . The default value is True.
    save_pn : , optional
        . The default value is None.
    close : bool, optional
        . The default value is False.
    xlprop : float, optional
        . The default value is 0.04.

    Returns
    -------
    None

    """
    if len(labels) < len(keys):
        raise ValueError('dimension mismatch ! len(labels) < len(keys) !')

    if len(labels) > len(keys):
        raise ValueError('dimension mismatch ! len(labels) > len(keys) !')

    # Extract then reshape data to have the correct way of plotting
    if type(querry_cc) == str:
        tbl_0 = dict_data[keys[0]][size][querry_cc]
        table = np.zeros((len(keys), tbl_0.shape[0], tbl_0.shape[1]))
        for i, ky in enumerate(keys):
            table[i] = dict_data[ky][size][querry_cc]

    else:
        tbl_0 = dict_data[keys[0]][size]
        table = np.zeros((len(keys), tbl_0.shape[0], tbl_0.shape[1]))
        for i, ky in enumerate(keys):
            table[i] = dict_data[ky][size]

    xlims = plot_limits(np.array([0, len(keys)-1]), xlprop)

    plt.figure(figsize=fgsz, dpi=dpi)
    if grid:
        plt.grid(grid, which='both', zorder=1)

    kernel = np.arange(len(keys))
    ksubp = np.linspace(-0.33, 0.33, table.shape[2])
    xlims = plot_limits(np.array([ksubp[0], len(keys)-1+ksubp[-1]]), xlprop)
    width = (ksubp[1]-ksubp[0])*0.66
    for i in range(table.shape[2]):
        plt.violinplot(table[:, :, i].T, kernel+ksubp[i], showmedians=True,
                       widths=width)

    if type(xylabel) == dict:
        if 'xlabel' not in list(xylabel.keys()):
            raise ValueError('Missing `xlabel` in xylabel dict !')
        if 'ylabel' not in list(xylabel.keys()):
            raise ValueError('Missing `ylabel` in xylabel dict !')
        if 'fontsize' not in list(xylabel.keys()):
            raise ValueError('Missing `fontsize` in xylabel dict !')

        plt.xlabel(xylabel['xlabel'], fontsize=xylabel['fontsize'])
        plt.ylabel(xylabel['ylabel'], fontsize=xylabel['fontsize'])

    plt.xticks(np.arange(len(keys)), labels, fontsize=13)
    plt.yticks(fontsize=14)
    plt.xlim(xlims[0], xlims[1])
    if ylim_b:
        ylims = plot_limits(np.ravel(table), 0.04)
        plt.ylim(ylims[0], ylims[1])

    if logscale:
        plt.yscale('log')

    if type(save_pn) == str:
        plt.savefig(save_pn, bbox_inches='tight')

    if close:
        plt.close()

    plt.show()
