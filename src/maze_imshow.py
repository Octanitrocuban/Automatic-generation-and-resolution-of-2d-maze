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
def delta_lims(a, p):
	"""
	Function to compute limits of an array given a width proportion for x and
	y limits in figure.

	Parameters
	----------
	a : numpy.ndarray
		Array from which we want to compute the limits.
	p : float
		Proportion of the limits. If is 0.1, it will return the minimum value
		of the array minus 1% of the extent of the array values and the
		maximum value of the array plus 1% of the extent of the array values.

	Returns
	-------
	float
		Minimum value.
	float
		Maximum value.

	"""
	mini = np.min(a)
	maxi = np.max(a)
	delta = (maxi-mini)*p
	return (mini-delta, maxi+delta)

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
		From ['fusion', 'randwalk', 'kurskal', 'ticking_f', 'ticking_s',
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
	possible_creation = ['randwalk', 'fusion', 'kurskal', 'ticking_f',
						 'ticking_s', 'jumper', 'hunter', 'grower', 'Eller',
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

	elif "kurskal" in creation:
		plat = mg.kurskal_maze(size)

	elif 'ticking_f' in creation:
		plat = mg.ticking_maze(size)

	elif 'ticking_s' in creation:
		plat = mg.ticking_maze(size, 'snake')

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

def distribution_caract_1_size(data, keys, labels, title, width, ylabel,
							   index=None, savep=None, figsize=(20, 8),
							   ttl_sz=18, label_sz=18, tck_sz=14, ylims=None):
	"""
	Function to plot the distribution of a caracteristic.

	Parameters
	----------
	data : dict
		Dictionary of dictionary of numpy.ndarray. Structure is:
			{method-i: {width-j: np.array(data-k)}}
	keys : list
		List of the keys to use to access the wanted data.
	labels : list
		List of the methods.
	title : str
		Title of the plot.
	width : str
		Key to access to the 'width-j' maze.
	ylabel : str
		Y label string.
	index : int, optional
		If not None, the index of the array to plot through the 2'nd axis.
	savep : str
		Path and name to save the figure.
	figsize : tuple, optional
		Size of the figure. The default is (20, 8).
	ttl_sz : float, optional
		Size of the title. The default is 18.
	label_sz : float, optional
		Size of the labels. The default is 18.
	tck_sz : float, optional
		Size of the ticks. The default is 14.
	ylims : list, optional
		Iterable object for the size limits on the y-axis of the plot. The
		default is None.

	Returns
	-------
	None.

	"""
	plt.figure(figsize=figsize)
	plt.title(title, fontsize=ttl_sz)
	for i in range(len(keys)):
		if type(index) == type(None):
			plt.violinplot(data[keys[i]][width], positions=[i],
						   showmedians=True)

		elif type(index) == int:
			plt.violinplot(data[keys[i]][width][:, index], positions=[i],
						   showmedians=True)

	plt.xlabel('Methods', fontsize=label_sz)
	plt.ylabel(ylabel, fontsize=label_sz)
	plt.xticks(np.arange(len(labels)), labels, fontsize=tck_sz)
	plt.yticks(fontsize=tck_sz)
	plt.xlim(-0.5, i+0.5)
	if (type(ylims) == list)|(type(ylims) == np.ndarray)|(type(ylims) == tuple):
		plt.ylim(ylims[0], ylims[1])

	if type(savep) == str:
		plt.savefig(savep, bbox_inches='tight')

	plt.show()

def distribution_caract_multip(data, keys, labels, title, width, ylabel,
							   index, caract_name, savep=None,
							   figsize=(20, 8), ttl_sz=18, label_sz=18,
							   tck_sz=14, dstb_w=0.4, ylims=None,
							   loc_leg='best'):
	"""
	Function to plot the distribution of multiple caracteristics.

	Parameters
	----------
	data : dict
		Dictionary of dictionary of numpy.ndarray. Structure is:
			{method-i: {width-j: np.array(data-k)}}
	keys : list
		List of the keys to use to access the wanted data.
	labels : list
		List of the methods.
	title : str
		Title of the plot.
	width : str
		Key to access to the 'width-j' maze.
	ylabel : str
		Y label string.
	index : numpy.ndarray
		Indexes of the caracteristics to plot.
	caract_name : list
		List of the characteritics name.
	savep : str, optional
		Path and name to save the figure.
	figsize : tuple, optional
		Size of the figure. The default is (20, 8).
	ttl_sz : float, optional
		Size of the title. The default is 18.
	label_sz : float, optional
		Size of the labels. The default is 18.
	tck_sz : float, optional
		Size of the ticks. The default is 14.
	dstb_w : float
		Width of the distribution markers.
	ylims : list, optional
		Iterable object for the size limits on the y-axis of the plot. The
		default is None.
	loc_leg : str, optional
		Postion of the legend. The default is 'best'.

	Returns
	-------
	None.

	"""
	cyclc_color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
				   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

	n_caracte = len(index)
	n_keys = len(keys)
	positions = np.arange(n_keys)
	delta = np.linspace(-0.40, 0.40, n_caracte)
	num_repet = data[keys[0]][width].shape[0]
	caracter = np.zeros((n_caracte, n_keys, num_repet))
	for i in range(n_keys):
		caracter[:, i] = data[keys[i]][width][:, index].T

	plt.figure(figsize=figsize)
	plt.title(title, fontsize=ttl_sz)
	for i in range(n_caracte):
		plt.plot(-1, 0.1, label=caract_name[i],
				 color=cyclc_color[i%n_caracte])

		violinpart = plt.violinplot(caracter[i].T,
									positions=positions+delta[i],
									widths=dstb_w, showmedians=True)

	plt.legend(loc=loc_leg, fontsize=14)
	plt.xlabel('Methods', fontsize=label_sz)
	plt.ylabel(ylabel, fontsize=label_sz)
	plt.xticks(np.arange(len(labels)), labels, fontsize=tck_sz)
	plt.yticks(fontsize=tck_sz)

	plt.xlim(-.6, positions[-1]+.6)
	if (type(ylims) == list)|(type(ylims) == np.ndarray)|(type(ylims) == tuple):
		plt.ylim(ylims[0], ylims[1])

	if type(savep) == str:
		plt.savefig(savep, bbox_inches='tight')

	plt.show()


def plot_one_stat(data_stats, axis, widths, keys, labels, title, ylabel,
				  symb='.-', prop_lm=0.02, figsize=(20, 8), ttl_sz=18,
				  label_sz=18, tck_sz=14, leg_sz=14, ylog=False, savep=None):
	"""
	Function to plot the evolution of a caracteritic.

	Parameters
	----------
	data_stats : dict
		Dictionary of numpy.ndarray.
	axis : int
		Axis to plot from the numpy.ndarray.
	widths : numpy.ndarray
		1d array of the widths of the maze to which the statistics comes from.
	keys : list
		List of the keys to use to access the wanted data.
	labels : list
		List of the methods.
	title : str
		Title of the plot.
	ylabel : str
		Y label string.
	symb : str, optional
		Marker and line symbol. The default is '.-'.
	prop_lm : float, optional
		Proportion to use when computing figure x and y limits. The default is
		0.02.
	figsize : tuple, optional
		Figure size in inches. The default is (20, 8).
	ttl_sz : float, optional
		Title size. The default is 18.
	label_sz : float, optional
		X and y label size. The default is 18.
	tck_sz : float, optional
		X and y ticks size. The default is 14.
	leg_sz : float, optional
		legend size. The default is 14.
	ylog : bool, optional
		If the y axis is in log scale (True) or not (False). The default is
		False.
	savep : str, optional
		Path and name to save the figure. The default is None.

	Returns
	-------
	None.

	"""
	xlms = delta_lims(widths, 0.02)
	ylms = [np.inf, -np.inf]

	plt.figure(figsize=figsize)
	plt.title(title, fontsize=ttl_sz)
	for i in range(len(keys)):
		curve = data_stats[keys[i]][:, axis]
		plt.plot(widths, curve, symb, label=labels[i])
		if np.min(curve) < ylms[0]:
			ylms[0] = np.min(curve) 
		if np.max(curve) > ylms[1]:
			ylms[1] = np.max(curve)

	plt.xlabel('Widths', fontsize=label_sz)
	plt.ylabel(ylabel, fontsize=label_sz)

	plt.xticks(widths, widths.astype(str), fontsize=tck_sz)
	plt.yticks(fontsize=tck_sz)

	plt.legend(fontsize=leg_sz)

	plt.xlim(xlms[0], xlms[1])
	if ylog:
		plt.yscale('log')
	else:
		ylms = delta_lims(ylms, prop_lm)
		plt.ylim(ylms[0], ylms[1])

	if type(savep) == str:
		plt.savefig(savep, bbox_inches='tight')

	plt.show()

def show_distrib_norm(data, clees, labels, width, indexes, features_names,
					  colors, title, n_sample=1000,
					  figsize=(20, 8), ylabel=None, xlims=None, lm_prop=0.04,
					  text_sz=10, savep=None):
	"""
	Function to plot normalized distribution of caracteristics.

	Parameters
	----------
	data : Dictionary of dictionary of numpy.ndarray. Structure is:
			{method-i: {width-j: np.array(data-k)}}
	clees : list
		List of the keys to use to access the wanted data.
	labels : list
		List of the methods name.
	width : str
		Key to access to the 'width-j' maze.
	indexes : numpy.ndarray
		Indexes of the caracteristics to plot.
	features_names : list
		List of the characteritics name.
	colors : list
		List of the color (one per characteristics).
	title : str
		Title of the plot.
	n_sample : int, optional
		Number of created maze. The default is 1000.
	figsize : tuple, optional
		Size of the figure. The default is (20, 8).
	ylabel : str, optional
		Y label string. The default is None.
	xlims : tuple, optional
		Figure limits values on the x-axis. The default is None.
	lm_prop : float, optional
		Proportion to use when computing figure x and y limits. The default is
		0.04.
	text_sz : float, optional
		Size of the text in the builted legend. The default is 10.
	savep : str, optional
		Path and name to save the figure. The default is None.

	Returns
	-------
	None.

	"""
	n_features = len(features_names)
	distribs = np.zeros((n_sample, len(clees), n_features))
	medians = np.zeros((len(labels), n_features))
	bounds = np.zeros((len(labels), 2, n_features))
	for i in range(len(clees)):
		distribs[:, i] = data[clees[i]][width][:, indexes]
		medians[i] = np.median(distribs[:, i], axis=0)
		bounds[i, 0] = np.min(distribs[:, i], axis=0)
		bounds[i, 1] = np.max(distribs[:, i], axis=0)

	limits = np.zeros((n_features+1, len(clees)))
	limits[1:] = np.cumsum(bounds[:, 1], axis=1).T
	limits = limits/limits[-1]

	medians_prop = ((medians-bounds[:, 0])/(bounds[:, 1]-bounds[:, 0])).T
	kernel = np.arange(len(clees))

	if type(xlims) == type(None):
		delta = (len(clees) + 1) * lm_prop
		xlims = (-1-delta, len(clees)-1+delta)

	if type(labels) == list:
		xticks = ['Legend']+labels
	elif type(labels) == np.array:
		xticks = np.append('Legend', labels)

	plt.figure(figsize=figsize)
	plt.title(title, fontsize=text_sz)
	for i in range(n_features):
		heights_lms = limits[i+1]-limits[i]
		plt.bar(kernel, heights_lms, width=0.6, bottom=limits[i], alpha=0.5,
				color=colors[i], edgecolor=colors[i], facecolor=colors[i],
				linewidth=3)

		posi_medians = medians_prop[i]*heights_lms+limits[i]
		plt.plot(kernel, posi_medians, 'P', ms=10, color=colors[i])
		plt.plot(kernel, posi_medians, 'k+', ms=25)

	# Adding a legend
	for i in range(n_features):
		plt.fill_between([-1.3, -0.7], i/(n_features+1),
						 (i+1)/(n_features+1), alpha=0.5,
						 color=colors[i], edgecolor=colors[i],
						 facecolor=colors[i], linewidth=3)

		plt.text(-1, (i+0.5)/(n_features+1), features_names[i],
				 ha='center', va='center', fontsize=text_sz)

	plt.plot(-1, 0.9, 'P', ms=10, color=colors[0])
	plt.plot(-1, 0.9, 'k+', ms=25)
	plt.text(-1, 0.95, 'Median', ha='center', va='center', fontsize=text_sz)

	plt.xlabel('Methods', fontsize=18)
	if type(ylabel) == str:
		plt.ylabel(ylabel, fontsize=18)

	plt.xticks(np.arange(-1, len(clees)), xticks, fontsize=16)
	plt.yticks(fontsize=16)
	plt.xlim(xlims[0], xlims[1])
	if type(savep) == str:
		plt.savefig(savep, bbox_inches='tight')

	plt.show()
