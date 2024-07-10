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
		From ['randwalk', 'fusion', 'kurskal' or 'ticking']
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
	possible_creation = ['randwalk', 'fusion', 'kurskal', 'ticking']
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
		plat = mg.maze_formation(plat)

	elif "kurskal" in creation:
		plat = mg.kurskal_maze(size)

	elif 'ticking' in creation:
		plat = mg.ticking_maze(size)

	if complexit:
		plat = mg.make_maze_complex(plat)

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

def caracterisation(maze_map):
	"""
	Function to compute the number of ground nodes with 1, 2, 3 and 4
	connections with other ground nodes.

	Parameters
	----------
	maze_map : numpy.ndarray
		2 dimensions numpy array of the maze. This array will have -1 for
		wall nods, and 0 for ground nods.

	Returns
	-------
	distrib : numpy.ndarray
		1d array (vectore) storing the number of ground nodes with 1, 2, 3
		and 4 connections with other ground nodes at index 0, 1, 2, and 3
		respectively.

	"""
	kernel = np.array([[[1, 0]], [[0, 1]], [[-1, 0]], [[0, -1]]])
	
	ground_nodes = np.argwhere(maze_map[1:-1, 1:-1] == 0)
	neighbors = ground_nodes+1+kernel
	neighbors = np.sum(maze_map[neighbors[:, :, 0],
							    neighbors[:, :, 1]] == 0, axis=0)

	distribution = np.zeros(4)
	values, counts = np.unique(neighbors, return_counts=True)
	distribution[values-1] = counts/len(ground_nodes)
	return distribution
