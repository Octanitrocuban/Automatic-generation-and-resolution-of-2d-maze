# -*- coding: utf-8 -*-
"""
@author: Matthieu Nougaret
		 PhD student, Institut de Physique du Globe de Paris
		 Volcanic Systems

This module contain functions to create maze with square array
"""
#import usefull library
import numpy as np
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
	base[1, 0] = 0
	base[-2, -1] = 0
	return base

def maze_formation(base):
	"""
	First method to generate automatically a maze. It take in input the output
	of the function create_maze_base. It will randomly draw 2 indices between 1
	and len(base)-2, then if it is a wall it break it by putting a 0 and the
	newly 2 conected ground node are set at 2. It continue until all the
	ground, starting and ending nodes are not connected together.

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
		The maze with -1 for wall, 0 for ground and 1 for starting and ending.

	Returns
	-------
	Ret : 2d numpy array of int
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
	Ret = np.full((base.shape[0]+4, base.shape[1]+4), -1)
	Ret[2:-2, 2:-2] = base
	arrete4 = len(Ret)
	if arrete4%2 == 0:
		Nx = range(3, arrete4-2, 2)
	else :
		Nx = range(3, arrete4-3, 2)

	x0, y0 = np.random.choice(Nx), np.random.choice(Nx)
	open_list = []
	open_list.append([x0, y0])
	Ret[x0, y0] = 1
	while len(np.unique(Ret)) != 2:
		m = np.array([Ret[x0-2, y0], Ret[x0, y0-2],
					  Ret[x0, y0+2], Ret[x0+2, y0]])
		chx = None
		if len(np.where(m == 0)[0]) > 0:
			if len(np.where(m == 0)[0])== 1:
				chx = np.where(m == 0)[0]
			elif len(np.where(m == 0)[0]) > 1:
				chx = np.random.choice(np.where(m == 0)[0])

			if chx == 0:
				Ret[x0-2:x0, y0] = 1
				x0, y0 = x0-2, y0
				open_list.append([x0, y0])
			elif chx == 1:
				Ret[x0, y0-2:y0] = 1
				x0, y0 = x0, y0-2
				open_list.append([x0, y0])
			elif chx == 2:
				Ret[x0, y0:y0+3] = 1
				x0, y0 = x0, y0+2
				open_list.append([x0, y0])
			elif chx == 3:
				Ret[x0:x0+3, y0] = 1
				x0, y0 = x0+2, y0
				open_list.append([x0, y0])

		else :
			back = 1
			while 0 not in m :
				x0, y0 = open_list[-back]
				m = np.array([Ret[x0-2, y0], Ret[x0, y0-2],
							  Ret[x0, y0+2], Ret[x0+2, y0]])
				back += 1
				chx = "No choice"

	Ret[Ret == 1] = 0
	Ret = Ret[2:-2, 2:-2]
	return Ret

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
			CH_y = np.arange(1, arrete, 2)
		else :
			CH_y = np.arange(2, arrete-1, 2)

		y = np.random.choice(CH_y)
		base[x, y] = 0

	return base
