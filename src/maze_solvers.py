# -*- coding: utf-8 -*-
"""
@author: Matthieu Nougaret
		 PhD student, Institut de Physique du Globe de Paris
		 Volcanic Systems

This module contain functions to solve maze created with the module
maze_generator.
"""
#import usefull library
import numpy as np
#=============================================================================
def maze_reduction(base):
	"""
	Look at the ground nodes, and if they are dead end, it turn them in wall.
	Note that if you did not use the function make_maze_complex, there will be
	only remain the unique straight path.

	Parameters
	----------
	base : numpy.ndarray
		Map of the maze, -1 are walls and 0 are ground.

	Returns
	-------
	core : numpy.ndarray
		Partial copy of the map of the maze, with -1 are walls and 0 are
		ground, but the dead end were replace by walls.
		
	Exemple
	-------
	[In 0]: _ = make_maze_complex(maze_formation(create_maze_base(11)))
	[Out 0]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
					[ 0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1, -1, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1],
					[-1,  0, -1, -1, -1, -1, -1,  0, -1,  0, -1],
					[-1,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1,  0, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1],
					[-1, -1, -1,  0, -1,  0, -1, -1, -1, -1, -1],
					[-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
					[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
	
	[In 1]: reduce = maze_reduction(_)
	[Out 1]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
					[ 0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1],
					[-1,  0, -1,  0, -1, -1, -1, -1, -1, -1, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0, -1, -1, -1],
					[-1,  0, -1, -1, -1, -1, -1,  0, -1, -1, -1],
					[-1,  0,  0,  0, -1, -1, -1,  0,  0,  0, -1],
					[-1, -1, -1,  0, -1, -1, -1,  0, -1,  0, -1],
					[-1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1],
					[-1, -1, -1,  0, -1,  0, -1, -1, -1, -1, -1],
					[-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0],
					[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

	"""
	core = np.copy(base)
	arrete = len(core)
	kernel = np.array([[[-1, 0]], [[0, -1]], [[0, 1]], [[1, 0]]])
	stop = False
	while stop != True:
		places = np.argwhere(core[1:-1, 1:-1] == 0)+1
		autour = places+kernel
		values = core[autour[:, :, 0], autour[:, :, 1]]
		mask = np.sum(values == -1, axis=0)
		places = places[mask == 3]
		if len(places) > 0:
			core[places[:, 0], places[:, 1]] = -1
		else:
			break

	return core

def maze_gradient(array):
	"""
	Generate 2d numpy array of int with -1 for walls, and the other values are
	the number of nodes to the ending node which is at 0. It continue until
	all the ground nodes have not been explored. It is a self implementaion of
	Dijkstra's algorithm.

	Parameters
	----------
	array : numpy.ndarray
		A 2 dimension array witch is the map on witch we want to compute the
		distance of cells from the [1, 0] position. Note that here this
		algorithm will see all cells that doesn't have the same value as the
		one at FromPos position as wall.

	Returns
	-------
	gradient : numpy.ndarray
		The map of the distance to the ending node with the number of ground
		nodes for metric. -1 values are the walls.

	Exemple
	-------
	[In 0]: _ = make_maze_complex(maze_formation(create_maze_base(11)))
	[Out 0]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
					[ 0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1, -1, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1],
					[-1,  0, -1, -1, -1, -1, -1,  0, -1,  0, -1],
					[-1,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1,  0, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1],
					[-1, -1, -1,  0, -1,  0, -1, -1, -1, -1, -1],
					[-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
					[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

	[In 1]: gradient = maze_gradient(_, np.array([9, 10]))
	[Out 1]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
					[22, 21, 20, 19, 20, 21, 22, 23, 24, 25, -1],
					[-1, 20, -1, 18, -1, -1, -1, -1, -1, 26, -1],
					[-1, 19, 18, 17, 18, 19, 20, 21, -1, 27, -1],
					[-1, -1, -1, 16, -1, -1, -1, -1, -1, 28, -1],
					[-1, 13, 14, 15, 16, 17, 18, 19, -1, 29, -1],
					[-1, 12, -1, 14, -1, 16, -1, -1, -1, -1, -1],
					[-1, 11, 12, 13, 14, 15, 16, 17, -1,  3, -1],
					[-1, 10, -1, -1, -1, -1, -1, -1, -1,  2, -1],
					[-1,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0],
					[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

	"""
	shp = array.shape
	# calculate the position at the end of the maze
	positions = np.copy([[shp[0]-2, shp[1]-1]])
	gradient = np.copy(array)
	stop = False
	dist = 1
	while stop != True:
		gradient[positions[:, 0], positions[:, 1]] = dist
		positions = np.array([[positions[:, 0]-1, positions[:, 1]],
							  [positions[:, 0]+1, positions[:, 1]],
							  [positions[:, 0], positions[:, 1]-1],
							  [positions[:, 0], positions[:, 1]+1]])

		positions = np.concatenate(positions, axis=1).T
		positions = np.unique(positions, axis=0)
		positions = positions[(positions[:, 0] >= 0)&(
							   positions[:, 1] >= 0)&(
							   positions[:, 0] < shp[0])&(
							   positions[:, 1] < shp[1])]

		positions = positions[gradient[positions[:, 0],
									   positions[:, 1]] == 0]
		dist += 1
		if len(positions) == 0:
			stop = True

	gradient = gradient-1
	gradient[gradient == -2] = -1
	return gradient

def descente_grad_maze(gradient_map):
	"""
	Using the distance map to the ending node with the number of ground nodes
	for metric, generated by the function maze_gradient, it begin at the
	starting node and choose the lowest value superior to -1 until it reached
	the ending node.

	Parameters
	----------
	gradient_map : numpy.ndarray
		Map of the distance to the ending node with the number of ground nodes
		for metric. Generated by the function maze_gradient. -1 are walls.

	Returns
	-------
	trajet : numpy.ndarray
		All the ground nodes position that constitute the shortest path from
		the starting to the ending node sorted down from the number of ground
		nodes at the ending node.

	Exemple
	-------
	[In 0]: gradient
	[Out 0]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
					[19, 18, 19, 20, -1, 18, 17, 16, 17, 18, -1],
					[-1, 17, -1, 19, -1, -1, -1, 15, -1, -1, -1],
					[-1, 16, -1, 18, 17, 16, 15, 14, -1, 16, -1],
					[-1, 15, -1, -1, -1, -1, -1, 13, -1, 15, -1],
					[-1, 14, 13, 12, -1, 14, 13, 12, 13, 14, -1],
					[-1, 15, -1, 11, -1, -1, -1, 11, -1, 13, -1],
					[-1, 16, -1, 10,  9,  8,  9, 10, 11, 12, -1],
					[-1, -1, -1,  9, -1,  7, -1, -1, -1, -1, -1],
					[-1, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1],
					[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

	[In 1]: path_way = descente_grad_maze(gradient)
	[Out 1]: array([[ 1,  0], [ 1,  1], [ 2,  1], [ 3,  1], [ 4,  1],
					[ 5,  1], [ 5,  2], [ 5,  3], [ 6,  3], [ 7,  3],
					[ 8,  3], [ 9,  3], [ 9,  4], [ 9,  5], [ 9,  6],
					[ 9,  7], [ 9,  8], [ 9,  9], [ 9, 10]])

	"""
	width = len(gradient_map)
	recadre = np.full((width+2, width+2), -1)
	recadre[1:-1, 1:-1] = gradient_map
	recadre[recadre == -1] = 10e6
	x = 2
	y = 1
	trajet = [[x, y]]
	while [width-1, width] not in trajet:
		up = recadre[x-1, y]
		down = recadre[x+1, y]
		right = recadre[x, y+1]
		left = recadre[x, y-1]
		if up <= down and up <= left and up <= right:
			x -= 1
		elif down <= up and down <= left and down <= right:
			x += 1
		elif left <= down and left <= up and left <= right:
			y -= 1
		elif right <= down and right <= left and right <= up:
			y += 1

		trajet.append([x, y])

	trajet = np.array(trajet)-1
	return trajet

def step_right_hand(maze_map, x, y, regard):
	"""
	Updating the position and the direction of the direction of the explorator
	in function of its position and watch direction, with condition that he
	have to always keep his 'right hand' glued to a wall.

	Parameters
	----------
	maze_map : numpy.ndarray
		Map of the maze, with -1 for walls and 0 for ground.
	x : int
		Row indice wich is the row where the explorator is.
	y : int
		Column indice wich is the column where the explorator is.
	regard : string
		Direction in witch the 'explorator goes.

	Returns
	-------
	x : int
		Row indice wich is the row where the explorator is after update.
	y : int
		Column indice wich is the column where the explorator is after update.
	regard : string
		Direction in witch the 'explorator goes after update.

	Exemple
	-------
	[In 0]: Maze
	[Out 0]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
					[ 0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1, -1, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1],
					[-1,  0, -1, -1, -1, -1, -1,  0, -1,  0, -1],
					[-1,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1,  0, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1],
					[-1, -1, -1,  0, -1,  0, -1, -1, -1, -1, -1],
					[-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
					[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

	[In 1]: step_right_hand(Maze, 1, 1, 'E')
	[Out 1]: (2, 1, 'S')
	

	"""
	if regard == "E":
		if maze_map[x+1, y] == 0:
			regard = "S"
			x += 1
		elif maze_map[x, y+1] == 0:
			regard = "E"
			y += 1
		elif maze_map[x-1, y] == 0:
			regard = "N"
			x -= 1
		elif maze_map[x, y-1] == 0:
			regard = "O"
			y -= 1

	elif regard == "O":
		if maze_map[x-1, y] == 0:
			regard = "N"
			x -= 1
		elif maze_map[x, y-1] == 0:
			regard = "O"
			y -= 1
		elif maze_map[x+1, y] == 0:
			regard = "S"
			x += 1
		elif maze_map[x, y+1] == 0:
			regard = "E"
			y += 1

	elif regard == "N":
		if maze_map[x, y+1] == 0:
			regard = "E"
			y += 1
		elif maze_map[x-1, y] == 0:
			regard = "N"
			x -= 1
		elif maze_map[x, y-1] == 0:
			regard = "O"
			y -= 1
		elif maze_map[x+1, y] == 0:
			regard = "S"
			x += 1

	elif regard == "S":
		if maze_map[x, y-1] == 0:
			regard = "O"
			y -= 1
		elif maze_map[x+1, y] == 0:
			regard = "S"
			x += 1
		elif maze_map[x, y+1] == 0:
			regard = "E"
			y += 1
		elif maze_map[x-1, y] == 0:
			regard = "N"
			x -= 1

	return x, y, regard

def step_left_hand(maze_map, x, y, regard):
	"""
	Updating the position and the direction of the direction of the explorator
	in function of its position and watch direction, with condition that he
	have to always keep his 'left hand' glued to a wall.

	Parameters
	----------
	maze_map : numpy.ndarray
		Map of the maze, with -1 for walls and 0 for ground.
	x : int
		Row indice wich is the row where the explorator is.
	y : int
		Column indice wich is the column where the explorator is.
	regard : string
		Direction in witch the 'explorator goes.

	Returns
	-------
	x : int
		Row indice wich is the row where the explorator is after update.
	y : int
		Column indice wich is the column where the explorator is after update.
	regard : string
		Direction in witch the 'explorator goes after update.

	 Exemple
	--------
	[In 0]: Maze
	[Out 0]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
					[ 0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1, -1, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1],
					[-1,  0, -1, -1, -1, -1, -1,  0, -1,  0, -1],
					[-1,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1,  0, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1],
					[-1, -1, -1,  0, -1,  0, -1, -1, -1, -1, -1],
					[-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
					[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

	[In 1]: step_left_hand(Maze, 1, 1, 'E')
	[Out 1]: (1, 2, 'E')

	"""
	if regard == "E":
		if maze_map[x-1, y] == 0:
			regard = "N"
			x -= 1
		elif maze_map[x, y+1] == 0:
			regard = "E"
			y += 1
		elif maze_map[x+1, y] == 0:
			regard = "S"
			x += 1
		elif maze_map[x, y-1] == 0:
			regard = "O"
			y -= 1

	elif regard == "O":
		if maze_map[x+1, y] == 0:
			regard = "S"
			x += 1
		elif maze_map[x, y-1] == 0:
			regard = "O"
			y -= 1
		elif maze_map[x-1, y] == 0:
			regard = "N"
			x -= 1
		elif maze_map[x, y+1] == 0:
			regard = "E"
			y += 1

	elif regard == "N":
		if maze_map[x, y-1] == 0:
			regard = "O"
			y -= 1
		elif maze_map[x-1, y] == 0:
			regard = "N"
			x -= 1
		elif maze_map[x, y+1] == 0:
			regard = "E"
			y += 1
		elif maze_map[x+1, y] == 0:
			regard = "S"
			x += 1

	elif regard == "S":
		if maze_map[x, y+1] == 0:
			regard = "E"
			y += 1
		elif maze_map[x+1, y] == 0:
			regard = "S"
			x += 1
		elif maze_map[x, y-1] == 0:
			regard = "O"
			y -= 1
		elif maze_map[x-1, y] == 0:
			regard = "N"
			x -= 1

	return x, y, regard

def wall_hand_solve(base, hand):
	"""
	Solve the maze with the mehod of following continuously the same wall
	while we are out from the starting to the ending node. Note that there
	will probably be multiple 'backward on his steps'.

	Parameters
	----------
	base : numpy.ndarray
		Map of the maze, with -1 for walls and 0 for ground.
	hand : string
		What 'hand' will be used by the 'explorator'. In other words what wall
		will he follow. It can be: ['r', 'R', 'right', 'Right', 'RIGHT', 'l',
		'L', 'left', 'Left', 'LEFT']

	Raise
	-----
	if hand not in ['r', 'R', 'right', 'Right', 'RIGHT', 'l', 'L', 'left',
	'Left', 'LEFT'] raise ValueError('hand is not method')

	Returns
	-------
	trajet : numpy.ndarray
		All the ground nodes position that constitute the path from the
		starting to the ending node in the same order as their was explored.

	Exemple
	-------
	[In 0]: Maze
	[Out 0]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
					[ 0,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1, -1, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0, -1,  0, -1],
					[-1,  0, -1, -1, -1, -1, -1,  0, -1,  0, -1],
					[-1,  0,  0,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1,  0, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0,  0,  0, -1],
					[-1, -1, -1,  0, -1,  0, -1, -1, -1, -1, -1],
					[-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
					[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

	[In 1]: wall_hand_solve(Maze, 'l')
	[Out 1]: array([[1, 0], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 4],
					[3, 5], [3, 6], [3, 7], [2, 7], [1, 7], [1, 6], [1, 5],
					[1, 6], [1, 7], [1, 8], [1, 9], [1, 8], [1, 7], [2, 7],
					[3, 7], [4, 7], [5, 7], [5, 8], [5, 9], [4, 9], [3, 9],
					[4, 9], [5, 9], [6, 9], [7, 9], [7, 8], [7, 7], [7, 6],
					[7, 5], [8, 5], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
					[9,10]])

	"""
	solve_method =  ['r','R','right','Right','RIGHT','l','L','left','Left','LEFT']
	if hand not in solve_method:
		raise ValueError(hand+' is not a possible method: '+str(solve_method))

	width = len(base)
	recadre = np.full((width+2, width+2), -1)
	recadre[1:-1, 1:-1] = base
	position = [2, 1]
	direction = "E" # Hand on the south wall
	end = [np.shape(recadre)[0]-3, np.shape(recadre)[1]-2]
	c = 0
	trajet = []
	trajet.append([position[0], position[1]])
	if hand in ['r', 'R', 'right', 'Right', 'RIGHT']:
		while position != end:
			position[0], position[1], direction = step_right_hand(recadre,
																  position[0],
																  position[1],
																  direction)
			trajet.append([position[0], position[1]])
			c += 1
			# security stop
			if c > np.size(recadre)*2:
				print("security break")
				break

	elif hand in ['l', 'L', 'left', 'Left', 'LEFT']:
		while position != end:
			position[0], position[1], direction = step_left_hand(recadre,
																 position[0],
																 position[1],
																 direction)
			trajet.append([position[0], position[1]])
			c += 1
			# security stop
			if c > np.size(recadre)*2:
				print("security break")
				break

	trajet = np.array(trajet)-1
	return trajet

def tri_hand_solve_path(path):
	"""
	This function will browse the list of positions in the array 'path' and
	remove all the 'backward' due to dead-end or loop (roundabout).

	Parameters
	----------
	path : numpy.ndarray
		Path from the starting to the ending node returned by wall_hand_solve,
		and that probably be multiple 'backward on his steps'.

	Returns
	-------
	trajet : numpy.ndarray
		Path from the starting to the ending node without passing more than
		one time per node.

	Exemple
	-------
	[In 0]: path
	[Out 0]: array([[1, 0], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 4],
					[3, 5], [3, 6], [3, 7], [2, 7], [1, 7], [1, 6], [1, 5],
					[1, 6], [1, 7], [1, 8], [1, 9], [1, 8], [1, 7], [2, 7],
					[3, 7], [4, 7], [5, 7], [5, 8], [5, 9], [4, 9], [3, 9],
					[4, 9], [5, 9], [6, 9], [7, 9], [7, 8], [7, 7], [7, 6],
					[7, 5], [8, 5], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
					[9,10]])

	[In 1]: tri_hand_solve_path(path)
	[Out 1]: array([[1, 0], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 4],
					[3, 5], [3, 6], [3, 7], [4, 7], [5, 7], [5, 8], [5, 9],
					[6, 9], [7, 9], [7, 8], [7, 7], [7, 6], [7, 5], [8, 5],
					[9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9,10]])

	"""
	trajet = np.copy(path)
	stop = False
	while stop != True:
		visited = []
		doub = False
		for i in range(len(trajet)):
			cell = list(trajet[i])
			if cell not in visited:
				visited.append(cell)
			else:
				visited.append(cell)
				doub = True
				break

		if doub == True:
			vs_ar = np.array(visited)
			vs_ar = (vs_ar[:, 0] == cell[0])&(vs_ar[:, 1] == cell[1])
			idx = np.arange(len(vs_ar))[vs_ar]
			trajet = np.concatenate((trajet[:np.min(idx)],
									 trajet[np.max(idx):]))

		if doub == False:
			stop = True

	return trajet
