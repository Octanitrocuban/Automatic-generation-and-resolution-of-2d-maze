# -*- coding: utf-8 -*-
"""
This module contain functions to create maze with square array.
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
	[In]: create_maze_base(8)
	[Out]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
				  [ 1,  2, -1,  3, -1,  4, -1,  5, -1],
				  [-1, -1, -1, -1, -1, -1, -1, -1, -1],
				  [-1,  6, -1,  7, -1,  8, -1,  9, -1],
				  [-1, -1, -1, -1, -1, -1, -1, -1, -1],
				  [-1, 10, -1, 11, -1, 12, -1, 13, -1],
				  [-1, -1, -1, -1, -1, -1, -1, -1, -1],
				  [-1, 14, -1, 15, -1, 16, -1, 17, 18],
				  [-1, -1, -1, -1, -1, -1, -1, -1, -1]])

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
	[In]: create_maze_base_boolean(8)
	[Out]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
			      [ 1,  0, -1,  0, -1,  0, -1,  0, -1],
				  [-1, -1, -1, -1, -1, -1, -1, -1, -1],
				  [-1,  0, -1,  0, -1,  0, -1,  0, -1],
				  [-1, -1, -1, -1, -1, -1, -1, -1, -1],
				  [-1,  0, -1,  0, -1,  0, -1,  0, -1],
				  [-1, -1, -1, -1, -1, -1, -1, -1, -1],
				  [-1,  0, -1,  0, -1,  0, -1,  0,  1],
				  [-1, -1, -1, -1, -1, -1, -1, -1, -1]])

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
	First method to generate automatically a maze. It take in input the
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
	[In]: maze_formation(create_maze_base(8))
	[Out]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
				  [ 0,  0, -1,  0, -1,  0, -1,  0, -1],
				  [-1,  0, -1,  0, -1,  0, -1,  0, -1],
				  [-1,  0, -1,  0, -1,  0,  0,  0, -1],
				  [-1,  0, -1,  0, -1, -1, -1,  0, -1],
				  [-1,  0, -1,  0, -1,  0,  0,  0, -1],
				  [-1,  0, -1,  0, -1,  0, -1, -1, -1],
				  [-1,  0,  0,  0,  0,  0,  0,  0,  0],
				  [-1, -1, -1, -1, -1, -1, -1, -1, -1]])

	"""
	arrete = len(base)
	walls = np.zeros((arrete, arrete))
	walls[::2] = 1
	walls = walls+walls.T
	walls = np.argwhere(walls[1:-1, 1:-1] == 1)+1
	unbreaked = np.ones(len(walls), dtype=bool)
	labyrinthe = np.copy(base)
	labyrinthe[1, 0] = 0
	labyrinthe[1, 1] = 0
	labyrinthe[-2, -1] = 0
	kernel = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])
	for i in range(len(walls)):
		rand = np.random.randint(len(walls))
		unbreaked[rand] = False
		choice = walls[rand]
		dalles = choice+kernel
		dalles = dalles[labyrinthe[dalles[:, 0], dalles[:, 1]] != -1]
		val_1 = labyrinthe[dalles[0, 0], dalles[0, 1]]
		val_2 = labyrinthe[dalles[1, 0], dalles[1, 1]]
		if val_1 != val_2:
			labyrinthe[choice[0], choice[1]] = 0
			minima = np.min([val_1, val_2])
			labyrinthe[labyrinthe == val_1] = minima
			labyrinthe[labyrinthe == val_2] = minima
			if np.max(labyrinthe) == 0:
				break

		walls = walls[unbreaked]
		unbreaked = unbreaked[unbreaked]

	return labyrinthe

def make_maze_exhaustif(base):
	"""
	Second method to generate automatically a maze. It take in input the
	output of the function create_maze_base_bool. It randomly draw a position
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
	[In]: make_maze_exhaustif(create_maze_base_boolean(8))
	[Out]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
				  [ 0,  0, -1,  0,  0,  0,  0,  0, -1],
				  [-1,  0, -1,  0, -1, -1, -1,  0, -1],
				  [-1,  0, -1,  0,  0,  0, -1,  0, -1],
				  [-1,  0, -1, -1, -1, -1, -1,  0, -1],
				  [-1,  0,  0,  0,  0,  0,  0,  0, -1],
				  [-1, -1, -1, -1, -1, -1, -1,  0, -1],
				  [-1,  0,  0,  0,  0,  0,  0,  0,  0],
				  [-1, -1, -1, -1, -1, -1, -1, -1, -1]])

	"""
	recadre = np.full((base.shape[0]+4, base.shape[1]+4), -1)
	recadre[2:-2, 2:-2] = base
	arrete4 = len(recadre)
	if arrete4%2 == 0:
		nx = range(3, arrete4-2, 2)
	else :
		nx = range(3, arrete4-3, 2)

	x0, y0 = np.random.choice(nx), np.random.choice(nx)
	open_list = []
	open_list.append([x0, y0])
	recadre[x0, y0] = 1
	while len(np.unique(recadre)) != 2:
		m = np.array([recadre[x0-2, y0], recadre[x0, y0-2],
					  recadre[x0, y0+2], recadre[x0+2, y0]])
		chx = None
		if len(np.where(m == 0)[0]) > 0:
			if len(np.where(m == 0)[0])== 1:
				chx = np.where(m == 0)[0]
			elif len(np.where(m == 0)[0]) > 1:
				chx = np.random.choice(np.where(m == 0)[0])

			if chx == 0:
				recadre[x0-2:x0, y0] = 1
				x0, y0 = x0-2, y0
				open_list.append([x0, y0])
			elif chx == 1:
				recadre[x0, y0-2:y0] = 1
				x0, y0 = x0, y0-2
				open_list.append([x0, y0])
			elif chx == 2:
				recadre[x0, y0:y0+3] = 1
				x0, y0 = x0, y0+2
				open_list.append([x0, y0])
			elif chx == 3:
				recadre[x0:x0+3, y0] = 1
				x0, y0 = x0+2, y0
				open_list.append([x0, y0])

		else :
			back = 1
			while 0 not in m :
				x0, y0 = open_list[-back]
				m = np.array([recadre[x0-2, y0], recadre[x0, y0-2],
							  recadre[x0, y0+2], recadre[x0+2, y0]])
				back += 1
				chx = "No choice"

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
	Out[0]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
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

	p_xy = np.meshgrid(range(x_node), range(x_node))
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
	[In 0]: _ = maze_formation(create_maze_base(8))
	[Out 0]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
					[ 0,  0, -1,  0, -1,  0, -1,  0, -1],
					[-1,  0, -1,  0, -1,  0, -1,  0, -1],
					[-1,  0, -1,  0, -1,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1,  0, -1, -1, -1],
					[-1,  0,  0,  0, -1,  0,  0,  0,  0],
					[-1, -1, -1, -1, -1, -1, -1, -1, -1]])
	
	[In 1]: make_maze_complex(_)
	[Out 1]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
					[ 0,  0,  0,  0, -1,  0, -1,  0, -1],
					[-1,  0, -1,  0, -1,  0, -1,  0, -1],
					[-1,  0, -1,  0, -1,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1,  0, -1, -1, -1],
					[-1,  0,  0,  0,  0,  0,  0,  0,  0],
					[-1, -1, -1, -1, -1, -1, -1, -1, -1]])

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
