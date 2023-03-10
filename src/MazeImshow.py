# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:40:12 2022

@author: Matthieu Nougaret

This script contain the functions to show the mazes, their solution path, and
the distance from ground nods to the exit nod.
"""
#import usefull library
import numpy as np
import matplotlib.pyplot as plt
from MazeGenerators import *
from MazeSolvers import *
#=============================================================================
def ShowMaze(MazeMap, Path=None, gradient=None):
	"""
	Function to show the maze once it is build. It is possible to add the path
	from start to exit and the gradient of the distance of the ground nods to
	the exit nod.

	Parameters
	----------
	MazeMap : np.ndarray
		2 dimensions numpy array of the maze. This array will have -1 for wall
		nods, and 0 for ground nods.
	Path : np.ndarray, optional
		2 dimensions numpy array of the path that is leading from the start to
		the exit of the maze. The default is None.
	gradient : np.ndarray, optional
		2 dimensions numpy array of the gradient of the maze. This array will
		have -1 for wall nods, and from 1 to N for ground nods. N is the
		maximum number of nods between the a ground nod and the exit nod. The
		default is None.

	Returns
	-------
	MazeMap : np.ndarray
		2 dimensions numpy array of the maze. This array will have -1 for wall
		nods, and 0 for ground nods.

	"""
	if len(MazeMap) <= 29:
		Marker = "wo-"

	elif (len(MazeMap) > 29) & (len(MazeMap) <= 72):
		Marker = "k.-"

	elif (len(MazeMap) > 72) & (len(MazeMap) <= 99):
		Marker = "m.-"

	else:
		Marker = "r-"
		
	C = 10+(len(MazeMap)/2)**0.48

	plt.figure(figsize=(C, C))
	plt.imshow(MazeMap, cmap='binary_r')
	if type(gradient) == np.ndarray:
		plt.imshow(gradient, cmap="jet")
	if type(Path) == np.ndarray:
		plt.plot(Path[:, 1], Path[:, 0], Marker)
	plt.axis('off')
	plt.show()
	return MazeMap


def FullLabyrinthe(Size, creation, resolve, Plot, Timing=True):
	"""
	Automatic generation and resolution of mazes with the method in
	MazeGenerators.py and MazeSolvers.py. Show of the maze at different step
	is also possible through the variable Plot. It is also possible to know
	the time took by the step throug the variable Timing.

	Parameters
	----------
	Size : int
		Length of the square of the maze. It must be stricly superior at 1.
	creation : list of string
		from ['Exhaustif' or 'Fusion' & None or 'Complexe']
	resolve : list of string
		from ('PreReduc' or None) and ('RH' or 'RHSingle' or 'Straight').
	Plot : string
		from 'all' or 'Labyrinthe' or 'PlotVide' or 'None'.
	Timing : Bool, optional
		Print the time for the execution of the tasks. The default is True.

	Raises
	------
	TypeError
		Size type isn't an int.
	ValueError
		Size is inferior at 2.
	TypeError
		creation type isn't a list.
	SyntaxError
		No method for maze generation put in creation list.
	TypeError
		Plot type isn't a string.
	TypeError
		resolve type isn't a list.
	TypeError
		Timing type isn't a bool.

	Returns
	-------
	Plat : 2d numpy array of int
		The map of the maze with -1 for walls and 0 for ground.

	"""
	#Checking types&others of parameters
	if type(Size) != int:
		raise TypeError("'Size' must be list type, found: "+str(type(Size)))

	if Size < 2:
		raise ValueError("'Size' must be superior or equal at 2, found: "+
						 str(Size))

	if type(creation) != list:
		raise TypeError("'creation' must be list of str, found: "+
						 str(type(creation)))

	if ("Exhaustif" not in creation)&("Fusion" not in creation):
		raise SyntaxError("You haven't specify the method of creation of "+
						  "the maze. Choose it in: 'Exhaustif'/'Fusion'")

	if type(Plot) != str:
		raise TypeError("'Plot' must be of str, find: "+str(type(Plot)))

	if type(resolve) != list:
		raise TypeError("'resolve' must be list of str, find: "+
						str(type(resolve)))
		
	if type(Timing) != bool:
		raise TypeError("'Timing' must be bool, find: "+ str(type(Timing)))

	tini = datetime.now()
	if "Exhaustif" in creation:
		Plat = CreateMazeBaseBoolean(Size)
		Plat = MakeMazeExhaustif(Plat)

	elif "Fusion" in creation:
		Plat = CreateMazeBase(Size)
		Plat = MazeFormation(Plat)

	if "Complexe" in creation:
		Plat = MakeMazeComplex(Plat)

	tfin = datetime.now()
	if Timing == True:
		print("Temps pour construire le labyrinthe =", tfin-tini)

	if (Plot == "PlotVide")|(Plot == "all"):
		ShowMaze(Plat)

	if "Straight" in resolve:
		t0 = datetime.now()
		S = MazeGradient(Plat)
		Desc = DescenteGradMaze(S)
		S = np.array(S, dtype=float)
		S[S == -1] = np.nan
		t1 = datetime.now()
		if Timing == True:
			print("Time to solves the maze =", str(str(t1-t0)))
		if (Plot == "Labyrinthe")|(Plot == "all"):
			ShowMaze(Plat, Desc, S)

	elif "PreReduc" in resolve:
		t0 = datetime.now()
		V = MazeReductionDE(Plat)
		SS = MazeGradient(V)
		DescDesc = DescenteGradMaze(SS)
		SS = np.array(SS, dtype=float)
		SS[SS == -1] = np.nan
		t1 = datetime.now()
		if Timing == True:
			print("Time to solves the maze =", str(str(t1-t0)))
		if (Plot == "Labyrinthe")|(Plot == "all"):
			ShowMaze(Plat, DescDesc, SS)

	elif "RH" in resolve:
		t0 = datetime.now()
		Chemin = WallHandSolve(Plat, 'R')
		t1 = datetime.now()
		if Timing == True:
			print("Time to solves the maze =", str(str(t1-t0)))
		if (Plot == "Labyrinthe")|(Plot == "all"):
			ShowMaze(Plat, Chemin)

	elif "LH" in resolve:
		t0 = datetime.now()
		Chemin = WallHandSolve(Plat, 'L')
		t1 = datetime.now()
		if Timing == True:
			print("Time to solves the maze =", str(str(t1-t0)))
		if (Plot == "Labyrinthe")|(Plot == "all"):
			ShowMaze(Plat, Chemin)

	elif "RHSingle" in resolve:
		t0 = datetime.now()
		Chemin = WallHandSolve(Plat, 'R')
		Chemin = TriHandSolvePath(Chemin)
		t1 = datetime.now()
		if Timing == True:
			print("Time to solves the maze =", str(str(t1-t0)))
		if (Plot == "Labyrinthe")|(Plot == "all"):
			ShowMaze(Plat, Chemin)

	elif "LHSingle" in resolve:
		t0 = datetime.now()
		Chemin = WallHandSolve(Plat, 'L')
		Chemin = TriHandSolvePath(Chemin)
		t1 = datetime.now()
		if Timing == True:
			print("Time to solves the maze =", str(str(t1-t0)))
		if (Plot == "Labyrinthe")|(Plot == "all"):
			ShowMaze(Plat, Chemin)

	return Plat
