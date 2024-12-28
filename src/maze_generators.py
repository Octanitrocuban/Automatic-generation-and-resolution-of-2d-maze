# -*- coding: utf-8 -*-
"""
This module contain functions to create maze with square array.
Implemented methods:
	- fusion
	- random walk
	- kurskal
	- ticking (Origin shift)
	- jumping explorer

"""
#import usefull library
import numpy as np
from scipy.spatial.distance import cdist
#=============================================================================
def create_maze_base(arrete):
	"""
	Generate the basis that will be used by the first method to generate
	automatically a maze with the function maze_formation.

	Parameters
	----------
	arrete : int
		Width and height of the maze, it have to be between 3 and +inf. Note
		that if you choose an even number, the output will have for width and
		height the higger odd number to the arrete parameter.

	Returns
	-------
	base : 2d numpy array of int
		The basis used as input by the function maze_formation. -1 are walls,
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

def maze_formation(base):
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
	In [0]: maze_formation(create_maze_base(11))
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

	# walls which aren't break yet
	unbreaked = np.ones(len(walls), dtype=bool)
	labyrinthe = np.copy(base)
	labyrinthe[1, 0] = 0
	labyrinthe[1, 1] = 0
	labyrinthe[-2, -1] = 0
	kernel = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])
	for i in range(len(walls)):
		# draw the which could have one of its wall broke down
		rand = np.random.randint(len(walls))

		# set the breakability to False
		unbreaked[rand] = False
		choice = walls[rand]
		dalles = choice+kernel
		dalles = dalles[labyrinthe[dalles[:, 0], dalles[:, 1]] != -1]
		val_1 = labyrinthe[dalles[0, 0], dalles[0, 1]]
		val_2 = labyrinthe[dalles[1, 0], dalles[1, 1]]

		# if the two path aren't connecte yet
		if val_1 != val_2:
			# break the wall
			labyrinthe[choice[0], choice[1]] = 0
			minima = np.min([val_1, val_2])
			labyrinthe[labyrinthe == val_1] = minima
			labyrinthe[labyrinthe == val_2] = minima
			# if every ground nodes have been reached
			if np.max(labyrinthe) == 0:
				break

		# remove breaked wall
		walls = walls[unbreaked]
		unbreaked = unbreaked[unbreaked]

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
			mask = np.sum(recadre[neig[:, :, 0], neig[:, :, 1]] == 0, axis=0) > 0
			adven = np.where(mask)[0][-1]

	recadre[recadre == 1] = 0
	recadre = recadre[2:-2, 2:-2]
	return recadre

def kurskal(points):
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
	In [0]: dots = np.random.uniform(-3, 10, (11, 2))
	In [1]: Kruskal_algorithm(dots)
	Out[1]: array([array([0, 3, 7, 6]), array([1, 3]), array([ 2,  8, 10]),
				   array([3, 0, 1]), array([ 4,  6,  9, 10]), array([5, 7]),
				   array([6, 4, 0]), array([7, 5, 0]), array([8, 2]),
				   array([9, 4]), array([10,  2,  4])], dtype=object)

	"""
	# calculates the distance matrix
	m_dist = cdist(points, points, metric='euclidean').T
	length = len(points)

	# list of array
	tree = list(np.arange(length)[:, np.newaxis])
	mask = (np.arange(length)-np.arange(length)[:, np.newaxis]) > 0

	# lists of index matrices
	indices = list(np.meshgrid(range(length), range(length)))

	# vector 1d to track connections in the tree and avoid loop formation
	state = np.arange(length)

	# We flatten the 2d matrix by keeping less than half of the distance
	# matrix not to re-evaluate relationships between pairs of points.
	sort_d = m_dist[mask]

	# The same is done for index matrices
	p_j = indices[0][mask]
	p_i = indices[1][mask]

	# Indices sorted in ascending order by distance values
	rank = np.argsort(sort_d)

	# Sorting indices and distance values
	p_i = p_i[rank]
	p_j = p_j[rank]
	sort_d = sort_d[rank]
	for i in range(len(sort_d)):
		# To have no recontection with loops in the tree
		if state[p_i[i]] != state[p_j[i]]:
			tree[p_i[i]] = np.append(tree[p_i[i]], p_j[i])
			tree[p_j[i]] = np.append(tree[p_j[i]], p_i[i])

			# Update of the 'state' vector
			minima = np.min([state[p_i[i]], state[p_j[i]]])
			state[state == state[p_i[i]]] = minima
			state[state == state[p_j[i]]] = minima

			# early stoping to avoid useless loop
			if len(state[state != minima]) == 0:
				break

	tree = np.array(tree, dtype=object)
	return tree

def kurskal_maze(n_node):
	"""
	Function to create a maze through the computation of a minimum spanning
	tree with Kruskal's algorithm. The result is similar to that obtained
	with maze_formation.

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
	In [0]: kurskal_maze(11)
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

	# Put random weigth on the connections
	p_xy_r = p_xy + np.random.uniform(-0.1, 0.1, (len(p_xy), 2))
	tree = kurskal(p_xy_r)

	# From the tree to the map
	carte = np.zeros((2*x_nodes+1, 2*x_nodes+1), dtype=int)-1
	index = (p_xy*2+1)
	carte[index[:, 0], index[:, 1]] = 0
	for i in range(len(tree)):
		mid = (index[tree[i][0]]+index[tree[i][1:]])//2
		carte[mid[:, 0], mid[:, 1]] = 0

	carte[1, 0] = 0
	carte[-2, -1] = 0
	return carte

def fork_init(size):
	"""
	Fork like structure initialization for the ticking maze.

	Parameters
	----------
	size : int
		DESCRIPTION.

	Returns
	-------
	directions : numpy.ndarray
		Directions of the nodes connections.

	Example
	-------
	In [0]: fork_init(11)
	Out[0]: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
				   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
				   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
				   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
				   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
				   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int8)

	"""
	directions = np.zeros((size, size), dtype='int8')
	directions[:, -1] = 1
	directions = np.ravel(directions)
	return directions

def ticking_maze(arrete, run_times=None):
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
	In [0]: ticking_maze(11)
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
	dots = np.meshgrid(range(0, size*2, 2), range(0, size*2, 2))
	dots = np.array([np.ravel(dots[0]), np.ravel(dots[1])]).T

	# Possible directions
	reffer = np.array([[2, 0], [0, 2], [-2, 0], [0, -2]])

	# what is the direction for each dots
	directions = fork_init(size)

	# to remove the unused connection
	masque = np.ones(len(directions), dtype=bool)
	current = len(directions)-1

	# creating the tree of all possible connections of the dots
	all_connect = cdist(dots, dots)
	mask_tree = (all_connect > 0)&(all_connect < 2.1)
	tree = []
	dir_tree = []
	for i in range(mask_tree.shape[0]):
		tree.append(np.argwhere(mask_tree[i])[:, 0])
		direc = dots[mask_tree[i]] - dots[i]
		direc = np.sum(direc[:, np.newaxis] == reffer, axis=2) == 2
		dir_tree.append(np.argwhere(direc)[:, 1])

	# number of iteration
	if type(run_times) == type(None):
		run_times = int(size**2)*10

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
	Method to generate automatically a maze. It take in input the output of
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

def make_maze_complex(base):
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
	In [0]: maze_formation(create_maze_base(11))
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
	
	[In 1]: make_maze_complex(_)
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
