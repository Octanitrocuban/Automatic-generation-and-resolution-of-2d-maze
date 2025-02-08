import numpy as np
import maze_solvers as ms
import matplotlib.pyplot as plt


def caracterisation_nco(maze_map, ground_nodes):
	"""
	Function to compute the number of ground nodes with 1, 2, 3 and 4
	connections with other ground nodes.

	Parameters
	----------
	maze_map : numpy.ndarray
		2 dimensions numpy array of the maze. This array will have -1 for
		wall nods, and 0 for ground nodes.
	ground_nodes : numpy.ndarray
		Positions of the tested nodes.

	Returns
	-------
	distrib : numpy.ndarray
		1d array (vectore) storing the number of ground nodes with 1, 2, 3
		and 4 connections with other ground nodes at index 0, 1, 2, and 3
		respectively.

	"""
	kernel = np.array([[[1, 0]], [[0, 1]], [[-1, 0]], [[0, -1]]])
	neighbors = ground_nodes+kernel
	neighbors = np.sum(maze_map[neighbors[:, :, 0],
							    neighbors[:, :, 1]] == 0, axis=0)

	distribution = np.zeros(4)
	values, counts = np.unique(neighbors, return_counts=True)
	distribution[values-1] = counts/np.sum(counts)
	return distribution

def path_opti_stats(maze_map, prob_map):
	"""
	Function to compute caracteristics of the optimal path of a given maze.

	Parameters
	----------
	maze_map : numpy.ndarray
		2d array maze map with 0 for ground and -1 for wall.
	prob_map : numpy.ndarray
		2d array of floats to compile the path of the mazes.

	Returns
	-------
	max_grad : int
		Number of cells between the maze end and it farest cell.
	len_p : int
		Length of the optimal path in number of cells.
	num_turn : int
		Number of turn in the optimal path.
	max_dx : int
		Longest horizontal way in the optimal path.
	max_dy : int
		Longest vertical way in the optimal path.
	prob_map : numpy.ndarray
		2d array of floats to compile the path of the mazes.

	"""
	gradient = ms.maze_gradient(maze_map)
	path = ms.descente_grad_maze(gradient)
	max_grad = np.max(gradient)

	# length path
	len_p = len(path)

	# number of turns
	diff_x = np.diff(np.diff(path[:, 0]))
	num_turn = len(diff_x[diff_x != 0])

	# max horiz len
	dif_x = np.argwhere(np.diff(path[:, 0]) == 0)[:, 0]
	if len(dif_x) <= 1:
		max_dx = 0
	else:
		dif_x = np.argwhere(np.diff(dif_x) != 1)[:, 0]
		if len(dif_x) <= 1:
			max_dx = 0
		else:
			max_dx = np.max(np.diff(dif_x))+1

	# max vert len
	dif_y = np.argwhere(np.diff(path[:, 1]) == 0)[:, 0]
	if len(dif_y) <= 1:
		max_dy = 0
	else:
		dif_y = np.argwhere(np.diff(dif_y) != 1)[:, 0]
		if len(dif_y) <= 1:
			max_dy = 0
		else:
			max_dy = np.max(np.diff(dif_y))+1

	# probability map
	prob_map[path[:, 0], path[:, 1]] = prob_map[path[:, 0], path[:, 1]] + 1

	return max_grad, len_p, num_turn, max_dx, max_dy, prob_map

def caracterisation_orient(maze_map, ground_nodes):
	"""
	Function to compute the distribution of the orientation of the connection
	of the cells.

	Parameters
	----------
	maze_map : numpy.ndarray
		2d array maze map with 0 for ground and -1 for wall.
	ground_nodes : numpy.ndarray
		Positions of the ground nodes (nodes which are allways ground).

	Returns
	-------
	distribution : numpy.ndarray
		Distribution of the orientation of the nodes.

	"""
	# orientations: South, East, North West
	kernel = np.array([[[1, 0]], [[0, 1]], [[-1, 0]], [[0, -1]]])
	neighbors = ground_nodes+kernel
	neighbors = maze_map[neighbors[:, :, 0], neighbors[:, :, 1]] == 0
	distribution = np.zeros(4)
	values, counts = np.unique(np.argwhere(neighbors)[:, 0],
							   return_counts=True)

	distribution[values-1] = counts/np.sum(counts)
	return distribution
