# -*- coding: utf-8 -*-
"""
This module contain functions to create maze with square array.
Implemented methods:
    - fusion
    - random walk
    - Kruskal
    - Origin shift
    - jumping explorer
    - hunt and kill
    - growing tree
    - Ellers' algorithm
    - binary tree with {none, center, randcent, snake, spiral} initialisation

Additional transformation:
    - multi-pathing

"""
#import usefull library
import numpy as np
from scipy.spatial.distance import cdist
#=============================================================================
def create_maze_base(arrete):
    """
    Generate the basis that will be used by the first method to generate
    automatically a maze with the function fusion.

    Parameters
    ----------
    arrete : int
        Width and height of the maze, it have to be between 3 and +inf. Note
        that if you choose an even number, the output will have for width and
        height the higger odd number to the arrete parameter.

    Returns
    -------
    base : 2d numpy array of int
        The basis used as input by the function fusion. -1 are walls,
        other values are all unic, from one the start node to 
        n = (sup_odd(arrete)-1)**2 /4 +2 the end node. The other are the
        starting ground nodes that will be connected.

    Exemple
    -------
    In [0]: create_maze_base(11)
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 1,  2, -1,  3, -1,  4, -1,  5, -1,  6, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1,  7, -1,  8, -1,  9, -1, 10, -1, 11, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, 12, -1, 13, -1, 14, -1, 15, -1, 16, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, 17, -1, 18, -1, 19, -1, 20, -1, 21, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, 22, -1, 23, -1, 24, -1, 25, -1, 26, 27],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """
    # To adapt if an even value is given
    if arrete%2 == 0:
        base = np.zeros((arrete+1, arrete+1), dtype=int)-1

    else:
        base = np.zeros((arrete, arrete), dtype=int)-1

    base[1::2, 1::2] = 0
    base[1, 0] = 0
    base[-2, -1] = 0
    base[base != -1] = range(1, len(base[base != -1])+1)
    return base

def create_maze_base_boolean(arrete):
    """
    Generate the basis that will be used by the second method to generate
    automatically a maze with the function make_maze_exhaustif.

    Parameters
    ----------
    arrete : int
        Width and height of the maze, it have to be between 3 and +inf. Note
        that if you choose an even number, the output will have for width and
        height the higger odd number to the arrete parameter.

    Returns
    -------
    base : 2d numpy array of int
        The basis used as input by the function make_maze_exhaustif. -1 are
        walls, 0 are the starting ground nodes that will be connected and 1
        are the start and end node.

    Exemple
    -------
    In [0]: create_maze_base_boolean(11)
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0,  1],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """
    if arrete%2 == 0:
        base = np.zeros((arrete+1, arrete+1), dtype=int)-1

    else:
        base = np.zeros((arrete, arrete), dtype=int)-1

    base[1::2, 1::2] = 0
    base[1, 0] = 1
    base[-2, -1] = 1
    return base

def fusion(base):
    """
    Method to generate automatically a maze. It take in input the
    output of the function create_maze_base. It will randomly draw 2 indices
    between 1 and len(base)-2, then if it is a wall it break it by putting a
    0 and the newly 2 conected ground node are set at 2. It continue until
    all the ground, starting and ending nodes are not connected together.

    Parameters
    ----------
    base : 2d numpy array of int
        The maze with -1 for wall and from 1 the starting node to 
        (sup_odd(arrete)-1)**2 /4 +2 the ending node for ground.

    Returns
    -------
    base : 2d numpy array of int
        The maze with -1 for wall and 0 for ground. At this stage, there is
        one possible path to connect starting and ending node without
        re-borrowing the same node several times.

    Exemple
    -------
    In [0]: fusion(create_maze_base(11))
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1,  0, -1,  0, -1, -1, -1],
                      [-1,  0,  0,  0, -1,  0, -1,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0, -1,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1,  0, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1,  0, -1, -1, -1,  0, -1, -1, -1],
                      [-1,  0, -1,  0, -1,  0,  0,  0,  0,  0,  0],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """
    arrete = len(base)
    # Get the positions of the walls which can be broke
    walls = np.zeros((arrete, arrete))
    walls[::2] = 1
    walls = walls+walls.T
    walls = np.argwhere(walls[1:-1, 1:-1] == 1)+1
    order = np.arange(len(walls))
    np.random.shuffle(order)
    walls = walls[order]

    kernel = np.array([[[-1, 0]], [[0, -1]], [[0, 1]], [[1, 0]]])
    dalles = walls+kernel

    labyrinthe = np.copy(base)
    labyrinthe[1, 0] = 0
    labyrinthe[1, 1] = 0
    labyrinthe[-2, -1] = 0
    for i in range(len(walls)):
        dalle = dalles[:, i, :]
        dalle = dalle[labyrinthe[dalle[:, 0], dalle[:, 1]] != -1]
        val_1 = labyrinthe[dalle[0, 0], dalle[0, 1]]
        val_2 = labyrinthe[dalle[1, 0], dalle[1, 1]]

        # if the two path aren't connecte yet
        if val_1 != val_2:
            # break the wall
            labyrinthe[walls[i, 0], walls[i, 1]] = 0
            minima = np.min([val_1, val_2])
            labyrinthe[labyrinthe == val_1] = minima
            labyrinthe[labyrinthe == val_2] = minima
            # if every ground nodes have been reached
            if np.max(labyrinthe) == 0:
                break

    return labyrinthe

def make_maze_exhaustif(base):
    """
    Method to generate automatically a maze. It take in input the output of
    the function create_maze_base_boolean. It randomly draw a position
    correponding to a ground node and change it's value from 0 to one. Then
    it randomly choose a unvisited other ground node in its four neighbours.
    It break the wall wall with a 1 and set the new gound node position to 1.
    It continue while all of his neighbours are visited. Then if all ground,
    starting and ending nodes are connected it stop, else it take the exact
    path it retraces his steps until he finds a possible passage, and rebreak
    the wall.

    Parameters
    ----------
    base : 2d numpy array of int
        The maze with -1 for wall, 0 for ground and 1 for starting and
        ending.

    Returns
    -------
    recadre : 2d numpy array of int
        The maze with -1 for wall and 0 for ground. At this stage, there is
        one possible path to connect starting and ending node without
        re-borrowing the same node several times.
        
    Exemple
    -------
    In [0]: make_maze_exhaustif(create_maze_base_boolean(11))
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 0,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1,  0, -1, -1, -1,  0, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0, -1,  0, -1,  0, -1],
                      [-1, -1, -1,  0, -1,  0, -1,  0, -1, -1, -1],
                      [-1,  0, -1,  0,  0,  0, -1,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0, -1,  0,  0,  0, -1,  0,  0,  0, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0, -1,  0,  0],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """
    # filled with -1
    recadre = np.full((base.shape[0]+4, base.shape[1]+4), -1)

    # filled with -1, 0 and 1
    recadre[2:-2, 2:-2] = base
    arrete4 = len(recadre)
    if arrete4%2 == 0:
        nx = range(3, arrete4-2, 2)
    else :
        nx = range(3, arrete4-3, 2)

    kernel = np.array([[[-2, 0]], [[0, -2]], [[0, 2]], [[2, 0]]], dtype=int)
    kern_cr = np.array([[[-2, 0], [-1, 0]], [[0, -2], [0, -1]],
                        [[ 0, 2], [ 0, 1]], [[2,  0], [1,  0]]], dtype=int)

    open_list = np.zeros((len(nx)**2, 2), dtype=int)

    # random starting point of the walk
    open_list[0] = np.random.choice(nx, 2)
    adven = 0

    # explored node is turn to 1
    recadre[open_list[0, 0], open_list[0, 1]] = 1
    while 0 in recadre:
        # get the neigbor of the current node
        cross = open_list[adven]+kernel[:, 0]
        m = recadre[cross[:, 0], cross[:, 1]]
        possible = np.where(m == 0)[0]

        # if at least one of them aren't connected yet
        if len(possible) > 0:
            # if there is only one possible choice
            if len(possible) == 1:
                chx = possible[0]

            # if there are more than one
            elif len(possible) > 1:
                chx = np.random.choice(possible)

            # to break the right wall in the right direction
            dwarf = open_list[adven]+kern_cr[chx]

            # breaking the wall and connecting the tile
            recadre[dwarf[:, 0], dwarf[:, 1]] = 1

            # updating position
            adven += 1
            open_list[adven] = cross[chx]

        # if no walls to breake => bo back on your feet until you found
        # breakable wall
        else :
            neig = kernel+open_list[:adven]
            mask = np.sum(recadre[neig[:, :, 0], neig[:, :, 1]] == 0,
                          axis=0) > 0

            adven = np.where(mask)[0][-1]

    recadre[recadre == 1] = 0
    recadre = recadre[2:-2, 2:-2]
    return recadre

def kruskal(points):
    """
    Function to compute Kruskal's algorithm.

    Parameters
    ----------
    node_p : numpy.ndarray
        Position of the nodes. It will be used to compute the connection's
        weight trhough euclidian distance.

    Returns
    -------
    tree : numpy.ndarray
        List of the nodes interconnections. The structure is as follow:
        [self indices nodes from nodes_p, ...list of node connected...].

    Example
    -------
    In [0]: p_xy = np.meshgrid(range(5), range(5))
    In [1]: p_xy = np.array([np.ravel(p_xy[1]), np.ravel(p_xy[0])]).T
    In [2]: p_xy_r = p_xy + np.random.uniform(-0.1, 0.1, (len(p_xy), 2))
    In [3]: kruskal(p_xy_r)
    Out[1]: array([[13, 14], [ 6,  7], [16, 21], [10, 15], [21, 22], [ 2,  3],
                   [19, 24], [ 3,  8], [ 3,  4], [ 7, 12], [10, 11], [12, 17],
                   [ 1,  2], [16, 17], [18, 19], [18, 23], [ 5, 10], [13, 18],
                   [ 1,  6], [ 4,  9], [ 8, 13], [15, 16], [ 0,  1], [20, 21]
                   ])

    """
    # calculates the distance matrix
    m_dist = cdist(points, points, metric='euclidean').T
    length = len(points)
    # from the creation of the points, there will be a limited number of
    # iteration, making a part of this huge vector useless.
    # Length of this vector is:
    #     length**4/2 - length**2/2
    # So I compute a treshold based on observation and theoritical knowledge
    # from how points was build to have a bound of n*log(n)
    # The use of the bound to cut useless distance comparison had a great
    # impact on improving performances.
    # This improvement is only guarenteed to work for maze purpose
    iter_n = int(round(length*np.log(length)))

    # vector 1d to track connections in the tree and avoid loop formation
    state = np.arange(length)

    # list of array
    tree = np.zeros((iter_n, 2), dtype=int)
    # upper half diag mask
    mask = (state-state[:, np.newaxis]) > 0

    # lists of index matrices
    indices = np.meshgrid(state, state)

    # Indices sorted in ascending order by distance values
    rank = np.argsort(m_dist[mask])[:iter_n]

    # We flatten the 2d matrix by keeping less than half of the distance
    # matrix not to re-evaluate relationships between pairs of points.
    # Sorting indices and distance values
    p_j = indices[0][mask][rank]
    p_i = indices[1][mask][rank]
    for i in range(len(p_i)):
        # To have no recontection with loops in the tree
        if state[p_i[i]] != state[p_j[i]]:
            tree[i] = p_i[i], p_j[i]

            # Update of the 'state' vector
            minima = np.min([state[p_i[i]], state[p_j[i]]])
            if state[p_i[i]] > state[p_j[i]]:
                state[state == state[p_i[i]]] = minima
            else:
                state[state == state[p_j[i]]] = minima

            # early stoping to avoid useless loop
            if np.max(state) == 0:
                break

    tree = tree[:i+1]
    tree = tree[np.any(tree != 0, axis=1)]
    return tree

def kruskal_maze(n_node):
    """
    Function to create a maze through the computation of a minimum spanning
    tree with Kruskal's algorithm. The result is similar to that obtained
    with fusion.

    Parameters
    ----------
    node_p : numpy.ndarray
        Position of the nodes. It will be used to compute the connection's
        weight trhough euclidian distance.

    Returns
    -------
    carte : numpy.ndarray
        The maze with -1 for wall and 0 for ground. At this stage, there is
        one possible path to connect starting and ending node without
        re-borrowing the same node several times.

    Example
    -------
    In [0]: kruskal_maze(11)
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 0,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1,  0, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0, -1,  0, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0, -1, -1, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0,  0,  0, -1],
                      [-1, -1, -1,  0, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1],
                      [-1,  0, -1, -1, -1,  0, -1,  0, -1,  0, -1],
                      [-1,  0, -1,  0,  0,  0, -1,  0,  0,  0,  0],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """
    if n_node%2 == 0:
        x_nodes = (n_node)//2
    else:
        x_nodes = (n_node-1)//2

    p_xy = np.meshgrid(range(x_nodes), range(x_nodes))
    p_xy = np.array([np.ravel(p_xy[1]), np.ravel(p_xy[0])]).T

    # Put random weigth on the connections and compute MST
    p_xy_r = p_xy + np.random.uniform(-0.1, 0.1, (len(p_xy), 2))
    tree = kruskal(p_xy_r)

    # From the tree to the map => adapt to new version !
    carte = np.zeros((2*x_nodes+1, 2*x_nodes+1), dtype=int)-1
    index = (p_xy*2+1)
    carte[index[:, 0], index[:, 1]] = 0
    mid = (index[tree[:, 0]]+index[tree[:, 1]])//2
    carte[mid[:, 0], mid[:, 1]] = 0

    carte[1, 0] = 0
    carte[-2, -1] = 0
    return carte

def fork_init(size):
    """
    Fork like structure initialization for the origin shift maze.

    Parameters
    ----------
    size : int
        Number of dots.

    Returns
    -------
    directions : numpy.ndarray
        Directions of the nodes connections.

    Example
    -------
    In [0]: fork_init(11)
    Out[0]: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 1], dtype=int8)

    """
    directions = np.zeros((size, size), dtype='int8')
    directions[:, -1] = 1
    directions = np.ravel(directions)
    return directions

def snake_init(size):
    """
    Snake like structure initialization for the origin shift maze.

    Parameters
    ----------
    size : int
        Number of dots.

    Returns
    -------
    directions : numpy.ndarray
        Directions of the nodes connections.

    Example
    -------
    In [0]: snake_init(11)
    Out[0]: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                      2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2,
                      2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 1], dtype=int8)

    """
    directions = np.zeros((size, size), dtype='int8')
    directions[::2, -1] = 1
    directions[1::2, 0] = 1
    directions[1::2, 1:] = 2
    if (size%2) == 0:
        directions[-1, :] = 0

    directions = np.ravel(directions)
    return directions

def origin_shift(arrete, initialization='fork', run_times=1):
    """
    Function to create a maze following the descripted algorithm in the
    youtube video of CaptainLuma in 'New Maze Generating Algorithm
    (Origin Shift)': https://www.youtube.com/watch?v=zbXKcDVV4G0

    Parameters
    ----------
    arrete : int
        Width and height of the maze, it have to be between 3 and +inf. Note
        that if you choose an even number, the output will have for width and
        height the higger odd number to the arrete parameter.
    initialization : str, optional
        Type of initialization. The default is 'fork'.
    run_times : int, optional
        Number of iteration of the algorithm (i.e. number of random step
        applied to shuffle the connections. The default is None.

    Returns
    -------
    carte : numpy.ndarray
        The maze with -1 for wall and 0 for ground. At this stage, there is
        one possible path to connect starting and ending node without
        re-borrowing the same node several times.

    Example
    -------
    In [0]: origin_shift(11)
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 0,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1,  0, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0, -1,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0, -1,  0,  0,  0, -1,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1,  0, -1,  0, -1, -1, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1,  0, -1,  0, -1, -1, -1],
                      [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """

    if arrete%2 == 0:
        size = arrete//2
    else:
        size = (arrete-1)//2

    # Initialization
    max_w_h = 2*size-1
    dots = np.meshgrid(range(0, size*2, 2), range(0, size*2, 2))
    dots = np.array([np.ravel(dots[0]), np.ravel(dots[1])]).T

    # Possible directions
    reffer = np.array([[2, 0], [0, 2], [-2, 0], [0, -2]])
    kernel = np.array([[[1, 0]], [[0, 1]], [[-1, 0]], [[0, -1]]])

    # what is the direction for each dots
    if initialization == 'fork':
        directions = fork_init(size)
    elif initialization == 'snake':
        directions = snake_init(size)

    # to remove the unused connection
    masque = np.ones(len(directions), dtype=bool)
    current = len(directions)-1

    # creating the tree of all possible connections of the dots
    ker = np.arange(4)
    co_dots = np.copy(dots)//2+1+kernel
    map_conec = np.zeros((size+2, size+2), dtype=int)-1
    map_conec[1:-1, 1:-1] = np.arange(size**2).reshape((size, size))
    map_conec = map_conec.T
    tree_arr = map_conec[co_dots[:, :, 0], co_dots[:, :, 1]].T
    tree = []
    dir_tree = []
    for i in range(len(dots)):
        tree.append(tree_arr[i][tree_arr[i] != -1])
        poss = dots[i]+reffer
        dir_tree.append(ker[np.all(poss >= 0, axis=1)&np.all(poss <= max_w_h, axis=1)])

    # number of iteration if not set
    if type(run_times) == type(None):
        run_times = int(size**3)

    for i in range(run_times):
        # nodes connected to the current one
        poss = tree[current]

        # random next connection
        next_i = np.random.randint(0, len(poss))

        # direction to take to create this connection
        directions[current] = dir_tree[current][next_i]

        # next position id
        next_p = poss[next_i]

        # update current node id
        current = next_p

    masque[current] = False
    dots += 1

    # create the maze map
    intersect = dots+reffer[directions]//2
    intersect = intersect[masque]
    carte = np.zeros((size*2+1, size*2+1), dtype=int)-1
    carte[1, 0] = 0
    carte[-2, -1] = 0
    carte[dots[:, 0], dots[:, 1]] = 0
    carte[intersect[:, 0], intersect[:, 1]] = 0
    return carte

def jumping_explorer(base):
    """
    Function to generate automatically a maze. It take in input the output of
    the function create_maze_base_boolean. It randomly draw a position
    correponding to a ground node and change it's value from 0 to one. Then
    it randomly choose a unvisited other ground node in its four neighbours.
    It break the wall wall with a 1 and set the new gound node position to 1.
    It continue while all of his neighbours are visited. Then if all ground,
    starting and ending nodes are connected it stop, else it take a random
    unreached cell and will run random walk until it found the main path or
    is stuck.

    Parameters
    ----------
    base : 2d numpy array of int
        The maze with -1 for wall, 0 for ground and 1 for starting and
        ending.

    Returns
    -------
    recadre : 2d numpy array of int
        The maze with -1 for wall and 0 for ground. At this stage, there is
        one possible path to connect starting and ending node without
        re-borrowing the same node several times.

    Exemple
    -------
    In [0]: jumping_explorer(create_maze_base_boolean(11))
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0, -1,  0,  0,  0, -1,  0,  0,  0, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0, -1, -1, -1],
                      [-1,  0,  0,  0, -1,  0, -1,  0, -1,  0, -1],
                      [-1, -1, -1, -1, -1, -1, -1,  0, -1,  0, -1],
                      [-1,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1,  0, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """
    recadre = np.full((base.shape[0]+4, base.shape[1]+4), -1)
    recadre[2:-2, 2:-2] = base
    arrete4 = len(recadre)
    if arrete4%2 == 0:
        nx = range(3, arrete4-2, 2)
    else :
        nx = range(3, arrete4-3, 2)

    kernel = np.array([[[-2, 0]], [[0, -2]], [[0, 2]], [[2, 0]]], dtype=int)
    kern_cr = np.array([[[-2, 0], [-1, 0]], [[0, -2], [0, -1]],
                        [[ 0, 2], [ 0, 1]], [[2,  0], [1,  0]]], dtype=int)

    cells_p = np.meshgrid(nx, nx)
    cells_p = np.array([np.ravel(cells_p[0]), np.ravel(cells_p[1])]).T
    adven = 1
    position = np.random.choice(nx, 2)
    cells_p = cells_p[(cells_p[:, 0] != position[0])|(
                       cells_p[:, 1] != position[1])]

    recadre[position[0], position[1]] = 1
    while 0 in recadre:
        # get the neigbor of the current node
        cross = position+kernel[:, 0]
        m = recadre[cross[:, 0], cross[:, 1]]
        if adven == 1:
            possible = np.where((m == 0))[0]
        else:
            possible = np.where((m > -1)&(m < adven))[0]

        # if at least one of them aren't connected yet
        if len(possible) > 0:
            # if there is only one possible choice
            if len(possible) == 1:
                chx = possible[0]

            # if there are more than one
            elif len(possible) > 1:
                chx = np.random.choice(possible)

            # to break the right wall in the right direction
            dwarf = position+kern_cr[chx]

            # breaking the wall and connecting the tile
            recadre[dwarf[:, 0], dwarf[:, 1]] = adven
            if m[chx] != 0:
                # if we connect to a path
                recadre[recadre == adven] = m[chx]
                if len(cells_p) > 0:
                    # if there are still unexplored cells
                    position = cells_p[np.random.randint(len(cells_p))]
                    cells_p = cells_p[(cells_p[:, 0] != position[0])|(
                                       cells_p[:, 1] != position[1])]

                    adven += 1
                    recadre[position[0], position[1]] = adven

                else:
                    break

            else:
                # updating position
                position = cross[chx]
                cells_p = cells_p[(cells_p[:, 0] != position[0])|(
                                   cells_p[:, 1] != position[1])]

        elif len(cells_p) > 0:
            # if there are still unexplored cells
            position = cells_p[np.random.randint(len(cells_p))]
            cells_p = cells_p[(cells_p[:, 0] != position[0])|(
                               cells_p[:, 1] != position[1])]

            adven += 1
            recadre[position[0], position[1]] = adven

        else:
            break

    # start and stop get the same values as their connected path
    recadre[ 3,  2] = recadre[ 3,  3]
    recadre[-4, -3] = recadre[-4, -4]

    while len(recadre[recadre > 1]) > 0:
        # while are the sub-path are not connected:
        # unconnected path values
        unic = np.unique(recadre[recadre > 1])
        for i in range(len(unic)):
            loc = np.argwhere(recadre == unic[i])
            cross = loc + kernel
            m = recadre[cross[:, :, 0], cross[:, :, 1]]
            mini = np.min(m[(m > 0)&(m != unic[i])])
            connect = np.argwhere(m == mini)
            if len(connect) == 1:
                chx = 0
            elif len(connect) > 1:
                chx = np.random.randint(len(connect))
            else:
                pass

            # "drilling" the connection between two sub-path
            dwarf = loc[connect[chx, 1]]+kern_cr[connect[chx, 0]]
            recadre[dwarf[:, 0], dwarf[:, 1]] = mini
            recadre[recadre == unic[i]] = mini

    recadre[recadre > 0] = 0
    recadre = recadre[2:-2, 2:-2]
    return recadre

def hunt_and_kill(base):
    """
    Function to generate a maze with hunt and kill algorithm.

    Parameters
    ----------
    base : 2d numpy array of int
        The maze with -1 for wall, 0 for ground and 1 for starting and
        ending.

    Returns
    -------
    recadre : 2d numpy array of int
        The maze with -1 for wall and 0 for ground. At this stage, there is
        one possible path to connect starting and ending node without
        re-borrowing the same node several times.


    Exemple
    -------
    In [0]: hunt_and_kill(create_maze_base_boolean(11))
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0, -1,  0, -1,  0,  0,  0, -1,  0, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0, -1,  0, -1,  0, -1],
                      [-1, -1, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0, -1,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1,  0, -1, -1, -1,  0, -1],
                      [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """
    recadre = np.full((base.shape[0]+4, base.shape[1]+4), -1)
    recadre[2:-2, 2:-2] = base
    arrete4 = len(recadre)
    if arrete4%2 == 0:
        nx = range(3, arrete4-2, 2)
    else :
        nx = range(3, arrete4-3, 2)

    kernel = np.array([[[-2, 0]], [[0, -2]], [[0, 2]], [[2, 0]]], dtype=int)
    kern_cr = np.array([[[-2, 0], [-1, 0]], [[0, -2], [0, -1]],
                        [[ 0, 2], [ 0, 1]], [[2,  0], [1,  0]]], dtype=int)

    cells_p = np.meshgrid(nx, nx)
    cells_p = np.array([np.ravel(cells_p[0]), np.ravel(cells_p[1])]).T
    position = np.random.choice(nx, 2)
    recadre[position[0], position[1]] = 1
    while 0 in recadre:
        # get the neigbor of the current node
        cross = position+kernel[:, 0]
        m = recadre[cross[:, 0], cross[:, 1]]
        possible = np.where((m == 0))[0]

        # if at least one of them aren't connected yet
        if len(possible) > 0:
            # if there is only one possible choice
            if len(possible) == 1:
                chx = possible[0]

            # if there are more than one
            elif len(possible) > 1:
                chx = np.random.choice(possible)

            # to break the right wall in the right direction
            dwarf = position+kern_cr[chx]

            # breaking the wall and connecting the tile
            recadre[dwarf[:, 0], dwarf[:, 1]] = 1

            # update the position
            position = position + kernel[chx, 0]

        else:

            target = cells_p[recadre[cells_p[:, 0], cells_p[:, 1]] == 1]
            neig = target+kernel
            m0_neig_v = np.sum(recadre[neig[:, :, 0], neig[:, :, 1]] == 0,
                               axis=0) > 0

            target = target[m0_neig_v]
            target = target[target[:, 0] == np.max(target[:, 0])]
            if len(target) > 1:
                target = target[target[:, 1] == np.max(target[:, 1])]

            position = target[0]

    recadre[recadre > 0] = 0
    recadre = recadre[2:-2, 2:-2]
    return recadre

def growing_tree(base, iterations=4):
    """
    Function to generate a maze with growing treee algorithm.

    Parameters
    ----------
    base : 2d numpy array of int
        The maze with -1 for wall, 0 for ground and 1 for starting and
        ending.
    iterations : int, optional
        Number of step done during the random walk. The default is 4.

    Returns
    -------
    recadre : 2d numpy array of int
        The maze with -1 for wall and 0 for ground. At this stage, there is
        one possible path to connect starting and ending node without
        re-borrowing the same node several times.

    Exemple
    -------
    In [0]: growing_tree(create_maze_base_boolean(11))
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 0,  0,  0,  0,  0,  0, -1,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1,  0, -1,  0, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0, -1,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1],
                      [-1,  0, -1,  0, -1, -1, -1,  0, -1, -1, -1],
                      [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """
    recadre = np.full((base.shape[0]+4, base.shape[1]+4), -1)
    recadre[2:-2, 2:-2] = base
    arrete4 = len(recadre)
    if arrete4%2 == 0:
        nx = range(3, arrete4-2, 2)
    else :
        nx = range(3, arrete4-3, 2)

    kernel = np.array([[[-2, 0]], [[0, -2]], [[0, 2]], [[2, 0]]], dtype=int)
    kern_cr = np.array([[[-2, 0], [-1, 0]], [[0, -2], [0, -1]],
                        [[ 0, 2], [ 0, 1]], [[2,  0], [1,  0]]], dtype=int)

    open_list = []
    position = np.random.choice(nx, 2)
    recadre[position[0], position[1]] = 1
    open_list.append(position)
    for i in range(iterations):
        # get the neigbor of the current node
        cross = position+kernel[:, 0]
        m = recadre[cross[:, 0], cross[:, 1]]
        possible = np.where((m == 0))[0]

        # if at least one of them aren't connected yet
        if len(possible) > 0:
            # if there is only one possible choice
            if len(possible) == 1:
                chx = possible[0]

            # if there are more than one
            elif len(possible) > 1:
                chx = np.random.choice(possible)

            # to break the right wall in the right direction
            dwarf = position+kern_cr[chx]

            # breaking the wall and connecting the tile
            recadre[dwarf[:, 0], dwarf[:, 1]] = 1

            # update the position
            position = position + kernel[chx, 0]
            open_list.append(position)
        else:
            break

    while len(open_list) > 0:
        idx = np.random.randint(len(open_list))
        position = open_list[idx]
        neig = position+kernel[:, 0]
        neig_v = recadre[neig[:, 0], neig[:, 1]]
        if 0 not in neig_v:
            open_list.pop(idx)
        else:
            for i in range(iterations):
                # get the neigbor of the current node
                cross = position+kernel[:, 0]
                m = recadre[cross[:, 0], cross[:, 1]]
                possible = np.where((m == 0))[0]

                # if at least one of them aren't connected yet
                if len(possible) > 0:
                    # if there is only one possible choice
                    if len(possible) == 1:
                        chx = possible[0]
                        
                    # if there are more than one
                    elif len(possible) > 1:
                        chx = np.random.choice(possible)

                    # to break the right wall in the right direction
                    dwarf = position+kern_cr[chx]

                    # breaking the wall and connecting the tile
                    recadre[dwarf[:, 0], dwarf[:, 1]] = 1

                    # update the position
                    position = position + kernel[chx, 0]
                    open_list.append(position)
                else:
                    break

        if 0 not in recadre:
            break

    recadre[recadre > 0] = 0
    recadre = recadre[2:-2, 2:-2]
    return recadre

def Eller(base):
    """
    Function to generate a maze with Ellers' algorithm.

    Parameters
    ----------
    base : 2d numpy array of int
        The maze with -1 for wall and from 1 the starting node to 
        (sup_odd(arrete)-1)**2 /4 +2 the ending node for ground.

    Returns
    -------
    base : 2d numpy array of int
        The maze with -1 for wall and 0 for ground. At this stage, there is
        one possible path to connect starting and ending node without
        re-borrowing the same node several times.

    Exemple
    -------
    In [0]: Eller(create_maze_base(11))
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1,  0, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0, -1,  0, -1],
                      [-1, -1, -1, -1, -1, -1, -1,  0, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0, -1,  0, -1],
                      [-1,  0, -1,  0, -1,  0, -1, -1, -1,  0, -1],
                      [-1,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0, -1, -1, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0,  0,  0,  0],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """
    size = base.shape[0]
    base[-2, -1] = -1
    base[ 1,  0] = -1
    nx = np.arange(1, size, 2)
    for r in range(size-2, 0, -2):
        for c in range(2, size-1, 2):
            if base[r, c-1] != base[r, c+1]:
                p_dwarf = np.random.rand()
                if (p_dwarf > 0.5)|(r == 1):
                    base[r, c] = base[r, c-1]
                    base[base == base[r, c+1]] = base[r, c-1]

        if r > 1:
            unic = np.unique(base[r])[1:]
            for i in range(len(unic)):
                splittable = np.argwhere(base[r] == unic[i])[:, 0]
                splittable = splittable[splittable%2 == 1]
                n_split = np.random.randint(1, len(splittable)+1)
                idx = np.random.choice(splittable, n_split)
                base[r-1, idx] = base[r, idx]
                base[r-2, idx] = base[r, idx]

    base[base > 0] = 0
    base[-2, -1] = 0
    base[ 1,  0] = 0
    return base

def sidewinder(base):
    """
    Function generate a maze with side winder algorithm.

    Parameters
    ----------
    base : 2d numpy array of int
        The maze with -1 for wall and from 1 the starting node to 
        (sup_odd(arrete)-1)**2 /4 +2 the ending node for ground.

    Returns
    -------
    base : 2d numpy array of int
        The maze with -1 for wall and 0 for ground. At this stage, there is
        one possible path to connect starting and ending node without
        re-borrowing the same node several times.

    Exemple
    -------
    In [0]: sidewinder(create_maze_base(11))
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 0,  0, -1,  0, -1,  0, -1,  0,  0,  0, -1],
                      [-1,  0, -1,  0, -1,  0, -1, -1, -1,  0, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                      [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                      [-1,  0, -1,  0, -1,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1,  0, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0, -1,  0,  0,  0, -1,  0, -1,  0, -1],
                      [-1,  0, -1, -1, -1,  0, -1,  0, -1,  0, -1],
                      [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """
    size = base.shape[0]
    base[-2, 1:] = base[-2, 1]
    base[1, 0] = -1
    for r in range(size-4, 0, -2):
        for c in range(2, size-1, 2):
            p_dwarf = np.random.rand()
            if p_dwarf > 0.5:
                base[r, c] = base[r, c-1]
                base[r, c+1] = base[r, c-1]

        unic = np.unique(base[r])[1:]
        for i in range(len(unic)):
            splittable = np.argwhere(base[r] == unic[i])[:, 0]
            splittable = splittable[splittable%2 == 1]
            idx = np.random.choice(splittable)
            base[r+1, idx] = base[r, idx]

    base[base > 0] = 0
    base[-2, -1] = 0
    base[ 1,  0] = 0
    return base

def spiral(size):
    """
    Version encore plus optimisée utilisant des calculs purement vectorisés.
    
    Args:
        n (int): Taille de la matrice (n×n)
    
    Returns:
        numpy.ndarray: Matrice n×n

    """
    n = int(size/2)
    if n <= 0:
        return np.array([])

    yi, xj = np.mgrid[:n, :n]
    i, j = np.ogrid[:n, :n]

    layer = np.minimum(np.minimum(i, j), np.minimum(i[::-1], j[:, ::-1]))
    layer_size = n - 2 * layer
    layer_start = 4 * layer * (n - layer)

    mask_t = (i == layer) & (j >= layer)
    mask_r = (j == n - 1 - layer) & (i > layer)
    mask_b = (i == n - 1 - layer) & (j < n - 1 - layer)
    mask_l = (j == layer) & (i < n - 1 - layer) & (i > layer)

    pos_in_layer = np.zeros((n, n), dtype=int)
    pos_in_layer[mask_t] = xj[mask_t] - layer[mask_t]
    pos_in_layer[mask_r] = (layer_size[mask_r]-1)+(yi[mask_r]-layer[mask_r])
    pos_in_layer[mask_b] = 2*(layer_size[mask_b]-1)+(layer_size[mask_b]-
                                               1-(xj[mask_b]-layer[mask_b]))

    pos_in_layer[mask_l] = 3*(layer_size[mask_l]-1)+(layer_size[mask_l]-
                                               1-(yi[mask_l]-layer[mask_l]))

    result = pos_in_layer+layer_start+1

    return np.abs(result - n**2)

def binary_tree(base, constraint='none'):
    """
    Function generate a maze with binary tree algorithm. The function can use
    various unitialisation. It was largely inspired by DqwertyC video: "A new
    maze algorithm optimized for Redstone"
    (https://www.youtube.com/watch?v=o7OhjEqCvSo)

    Parameters
    ----------
    base : 2d numpy array of int
        The maze with -1 for wall, 0 for ground and 1 for starting and
        ending.
    constraint : str, optional
        If (and which) there is a constraint on the generation.

    Returns
    -------
    base : 2d numpy array of int
        The maze with -1 for wall and 0 for ground. At this stage, there is
        one possible path to connect starting and ending node without
        re-borrowing the same node several times.

    Exemple
    -------
    In [0]: binary_tree(create_maze_base(11), 'none')
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 0,  0, -1,  0, -1,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1,  0, -1,  0, -1, -1, -1, -1, -1],
                      [-1,  0, -1,  0,  0,  0, -1,  0,  0,  0, -1],
                      [-1,  0, -1,  0, -1, -1, -1,  0, -1, -1, -1],
                      [-1,  0, -1,  0,  0,  0, -1,  0,  0,  0, -1],
                      [-1,  0, -1,  0, -1, -1, -1,  0, -1, -1, -1],
                      [-1,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """
    size = base.shape[0]
    base[1, 0] = 0
    base[-2, -1] = 0
    if constraint == 'none':
        base[-2, 1:-1] = 0
        base[1:-1, 1] = 0
        kern_cr = np.array([[1, 0], [0, -1]], dtype=int)
        opening = np.random.randint(0, 2, (size//2-1)**2)
        grid = np.meshgrid(range(3, size-1, 2), range(1, size-3, 2))
        grid = np.array([np.ravel(grid[1]), np.ravel(grid[0])]).T
        dwarf = grid + kern_cr[opening]
        base[dwarf[:, 0], dwarf[:, 1]] = 0

    else:
        kern_cr = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]], dtype=int)
        kernel = np.array([[[-2, 0]], [[0, -2]], [[0, 2]], [[2, 0]]],
                          dtype=int)

        recadre = np.zeros((size+2, size+2))+size**2
        recadre[1:-1, 1:-1] = base
        grid = np.meshgrid(range(2, size, 2), range(2, size, 2))
        cells_p = np.array([np.ravel(grid[1]), np.ravel(grid[0])]).T

        if constraint == 'snake':
            hsz = size//2
            cells_v = np.zeros((hsz, hsz))
            cells_v[::2] = np.arange(hsz)
            cells_v[1::2] = np.arange(hsz)[::-1]
            cells_v = np.ravel(cells_v + np.arange(hsz)[:, np.newaxis]*hsz)

        elif constraint == 'center':
            if size%4 == 3:
                cells_v = np.ravel(np.abs(grid[0]-(size//2+1))+
                                   np.abs(grid[1]-(size//2+1)))

            elif size%4 == 1:
                cells_v = np.ravel(np.abs(grid[0]-(size//2))+
                                   np.abs(grid[1]-(size//2)))

        elif constraint == 'randcent':
            center = cells_p[np.random.randint(0, len(cells_p))]
            cells_v = np.ravel(np.abs(grid[0]-center[0])+
                               np.abs(grid[1]-center[1]))

        elif constraint == 'spiral':
            cells_v = np.ravel(spiral(size))

        recadre[cells_p[:, 0], cells_p[:, 1]] = cells_v
        neig = cells_p+kernel
        neig_v = recadre[neig[:, :, 0], neig[:, :, 1]]
        poss_co = neig_v < cells_v
        lim_prob = np.zeros((5, len(cells_v)))
        lim_prob[1:] = np.cumsum(poss_co/np.sum(poss_co, axis=0), axis=0)
        lim_prob[np.isnan(lim_prob)] = 0
        rand = np.random.rand(len(cells_v))
        idx_m = np.argwhere((rand < lim_prob[1:])&(rand > lim_prob[:-1]))
        dwarf = cells_p[idx_m[:, 1]] + kern_cr[idx_m[:, 0]]
        recadre[dwarf[:, 0], dwarf[:, 1]] = 0

        base = recadre[1:-1, 1:-1]

    base[base > 0] = 0
    return base

def make_multipathe(base):
    """
    This function will transform the maze in order that their will multiple
    paths from start to end. To achieve this goal, it randomly break some
    walls that are separating two ground nodes.

    Parameters
    ----------
    base : 2d numpy array of int
        The maze with -1 for wall and 0 for ground.

    Returns
    -------
    base : 2d numpy array of int
        The maze with -1 for wall and 0 for ground.
        
    Exemple
    -------
    In [0]: fusion(create_maze_base(11))
    Out[0]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                      [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1,  0, -1,  0, -1, -1, -1],
                      [-1,  0,  0,  0, -1,  0, -1,  0,  0,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1, -1, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0, -1,  0, -1],
                      [-1,  0, -1, -1, -1, -1, -1,  0, -1,  0, -1],
                      [-1,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
                      [-1,  0, -1,  0, -1, -1, -1,  0, -1, -1, -1],
                      [-1,  0, -1,  0, -1,  0,  0,  0,  0,  0,  0],
                      [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
    
    [In 1]: make_multipathe(_)
    [Out 1]: np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                       [-1,  0, -1,  0, -1,  0, -1,  0, -1, -1, -1],
                       [-1,  0,  0,  0,  0,  0, -1,  0,  0,  0, -1],
                       [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
                       [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
                       [-1,  0, -1, -1, -1, -1, -1,  0, -1,  0, -1],
                       [-1,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
                       [-1,  0, -1,  0, -1, -1, -1,  0, -1, -1, -1],
                       [-1,  0, -1,  0, -1,  0,  0,  0,  0,  0,  0],
                       [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

    """
    arrete = len(base)
    for i in range(arrete):
        x = np.random.randint(1, arrete-2)
        if x%2 == 0:
            ch_y = np.arange(1, arrete, 2)
        else :
            ch_y = np.arange(2, arrete-1, 2)

        y = np.random.choice(ch_y)
        base[x, y] = 0

    return base
