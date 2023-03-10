# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:40:12 2022

@author: Matthieu Nougaret
"""
#import usefull library
import numpy as np
#=============================================================================
def MazeReductionDE(Base):
	"""
	Look at the ground nodes, and if they are dead end, it turn them in wall.
	Note that if you did not use the function MakeMazeComplex, there will be
	only remain the unic straight path.

	Parameters
	----------
	Base : numpy.ndarray
		Map of the maze, -1 are walls and 0 are ground.

	Returns
	-------
	Core : numpy.ndarray
		Partial copy of the map of the maze, with -1 are walls and 0 are
		ground, but the dead end were replace by walls.
		
	Exemple
	-------
	[In 0]: _ = MakeMazeComplex(MazeFormation(CreateMazeBase(11)))
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
	
	[In 1]: reduce = MazeReductionDE(_)
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
	Core = np.copy(Base)
	arrete = len(Core)
	Stop = False
	while Stop != True:
		instep = 0
		for i in range(1, arrete-1):
			for j in range(1, arrete-1):
				if Core[i, j] == 0:
					Take = np.array([Core[i-1, j], Core[i+1, j],
									 Core[i, j-1], Core[i, j+1]])
					if Take.sum() == -3 :
						Core[i, j] = -1
						instep += 1
		if instep == 0:
			break
	return Core

def MazeGradient(Base):
	"""
	Generate 2d numpy array of int with -1 for walls, and the other values are
	the number of nodes to the ending node which is at 0. It continue until
	all the ground nodes have not been explored.

	Parameters
	----------
	Base : numpy.ndarray
		The map of the maze, a 2d numpy array with -1 for walls and 0 for
		grounds.

	Returns
	-------
	Recadr : numpy.ndarray
		The map of the distance to the ending node with the number of ground
		nodes for metric. -1 values are the walls.

	Exemple
	-------
	[In 0]: _ = MakeMazeComplex(MazeFormation(CreateMazeBase(11)))
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

	[In 1]: gradient = MazeGradient(_)
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
	Arrete = len(Base)
	Recadr = np.full((Arrete+2, Arrete+2), -1)
	Recadr[1:-1, 1:-1] = Base
	distance = 1
	open_liste = [[Arrete-1, Arrete]]
	close_liste = []
	temp_open_liste = []
	Recadr[open_liste[0][0], open_liste[0][1]] = distance
	while len(open_liste) > 0:
		try :
			distance += 1
			for n in range(len(open_liste)):
				Mm = np.array([Recadr[open_liste[n][0]-1, open_liste[n][1]],
							   Recadr[open_liste[n][0], open_liste[n][1]-1],
							   Recadr[open_liste[n][0], open_liste[n][1]+1],
							   Recadr[open_liste[n][0]+1, open_liste[n][1]]])
				for i in range(4):
					if Mm[i] == 0:
						if i == 0:
							xy = [open_liste[n][0]-1, open_liste[n][1]]
						elif i == 1:
							xy = [open_liste[n][0], open_liste[n][1]-1]
						elif i == 2:
							xy = [open_liste[n][0], open_liste[n][1]+1]
						elif i == 3:
							xy = [open_liste[n][0]+1, open_liste[n][1]]

						if xy not in close_liste :
							temp_open_liste.append([xy[0], xy[1]])
							Recadr[xy[0], xy[1]] = distance
			close_liste.append(open_liste[0])#close_liste =
			open_liste.remove(open_liste[0])
			open_liste = temp_open_liste.copy()
			temp_open_liste.remove(temp_open_liste[0])
		except :
			break
	Recadr = Recadr[1:-1, 1:-1]-1
	return Recadr

def FastMazeGradient(Array):
	"""
	A fasted method to generate 2d numpy array of int with -1 for walls, and
	the other values are the number of nodes to the ending node which is at 0.
	It continue until all the ground nodes have not been explored.

	Parameters
	----------
	Array : numpy.ndarray
		A 2 dimension array witch is the map on witch we want to compute the
		distance of cells from the FromPos position. Note that here this
		algorithm will see all cells that doesn't have the same value as the
		one at FromPos position as wall.

	Returns
	-------
	gradient : numpy.ndarray
		The map of the distance to the ending node with the number of ground
		nodes for metric. -1 values are the walls.

	Exemple
	-------
	[In 0]: _ = MakeMazeComplex(MazeFormation(CreateMazeBase(11)))
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

	[In 1]: gradient = FastMazeGradient(_, np.array([9, 10]))
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
	shp = Array.shape
	e = np.copy([[shp[0]-2, shp[1]-1]])
	v = Array[e[:, 0], e[:, 1]]
	gradient = (np.copy(Array) == v).astype(int)-1
	Stop = False ; c = 1
	while Stop != True:
		gradient[e[:, 0], e[:, 1]] = c
		e = np.array([[e[:, 0]-1, e[:, 1]], [e[:, 0]+1, e[:, 1]],
					  [e[:, 0], e[:, 1]-1], [e[:, 0], e[:, 1]+1]])
		e = np.concatenate(e, axis=1).T
		e = np.unique(e, axis=0)
		e = e[e[:, 0] >= 0]
		e = e[e[:, 1] >= 0]
		e = e[e[:, 0] < shp[0]]
		e = e[e[:, 1] < shp[1]]
		e = e[gradient[e[:, 0], e[:, 1]] == 0]
		c += 1
		if len(e) == 0:
			Stop = True
	gradient = gradient-1
	gradient[gradient == -2] = -1
	return gradient

def DescenteGradMaze(GradientMap):
	"""
	Using the distance map to the ending node with the number of ground nodes
	for metric, generated by the function MazeGradient, it begin at the
	starting node and choose the lowest value superior to -1 until it reached
	the ending node.

	Parameters
	----------
	GradientMap : numpy.ndarray
		Map of the distance to the ending node with the number of ground nodes
		for metric. Generated by the function MazeGradient. -1 are walls.

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

	[In 1]: PathWay = DescenteGradMaze(gradient)
	[Out 1]: array([[ 1,  0], [ 1,  1], [ 2,  1], [ 3,  1], [ 4,  1],
					[ 5,  1], [ 5,  2], [ 5,  3], [ 6,  3], [ 7,  3],
					[ 8,  3], [ 9,  3], [ 9,  4], [ 9,  5], [ 9,  6],
					[ 9,  7], [ 9,  8], [ 9,  9], [ 9, 10]])

	"""
	Arrete = len(GradientMap)
	Recadr = np.full((Arrete+2, Arrete+2), -1)
	Recadr[1:-1, 1:-1] = GradientMap
	Recadr[Recadr == -1] = 10e6
	x = 2
	y = 1
	trajet = [[x, y]]
	while [Arrete-1, Arrete] not in trajet:
		up = Recadr[x-1, y]
		down = Recadr[x+1, y]
		right = Recadr[x, y+1]
		left = Recadr[x, y-1]
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

def StepRightHandSolve(MazeMap, X, Y, Regard):
	"""
	Updating the position and the direction of the direction of the explorator
	in function of its position and watch direction, with condition that he
	have to always keep his 'right hand' glued to a wall.

	Parameters
	----------
	MazeMap : numpy.ndarray
		Map of the maze, with -1 for walls and 0 for ground.
	X : int
		Row indice wich is the row where the explorator is.
	Y : int
		Column indice wich is the column where the explorator is.
	Regard : string
		Direction in witch the 'explorator goes.

	Returns
	-------
	X : int
		Row indice wich is the row where the explorator is after update.
	Y : int
		Column indice wich is the column where the explorator is after update.
	Regard : string
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

	[In 1]: StepRightHandSolve(Maze, 1, 1, 'E')
	[Out 1]: (2, 1, 'S')
	

	"""
	if Regard == "E":
		if MazeMap[X+1, Y] == 0:
			Regard = "S"
			X += 1
		elif MazeMap[X, Y+1] == 0:
			Regard = "E"
			Y += 1
		elif MazeMap[X-1, Y] == 0:
			Regard = "N"
			X -= 1
		elif MazeMap[X, Y-1] == 0:
			Regard = "O"
			Y -= 1
	elif Regard == "O":
		if MazeMap[X-1, Y] == 0:
			Regard = "N"
			X -= 1
		elif MazeMap[X, Y-1] == 0:
			Regard = "O"
			Y -= 1
		elif MazeMap[X+1, Y] == 0:
			Regard = "S"
			X += 1
		elif MazeMap[X, Y+1] == 0:
			Regard = "E"
			Y += 1
	elif Regard == "N":
		if MazeMap[X, Y+1] == 0:
			Regard = "E"
			Y += 1
		elif MazeMap[X-1, Y] == 0:
			Regard = "N"
			X -= 1
		elif MazeMap[X, Y-1] == 0:
			Regard = "O"
			Y -= 1
		elif MazeMap[X+1, Y] == 0:
			Regard = "S"
			X += 1
	elif Regard == "S":
		if MazeMap[X, Y-1] == 0:
			Regard = "O"
			Y -= 1
		elif MazeMap[X+1, Y] == 0:
			Regard = "S"
			X += 1
		elif MazeMap[X, Y+1] == 0:
			Regard = "E"
			Y += 1
		elif MazeMap[X-1, Y] == 0:
			Regard = "N"
			X -= 1
	return X, Y, Regard

def StepLeftHandSolve(MazeMap, X, Y, Regard):
	"""
	Updating the position and the direction of the direction of the explorator
	in function of its position and watch direction, with condition that he
	have to always keep his 'left hand' glued to a wall.

	Parameters
	----------
	MazeMap : numpy.ndarray
		Map of the maze, with -1 for walls and 0 for ground.
	X : int
		Row indice wich is the row where the explorator is.
	Y : int
		Column indice wich is the column where the explorator is.
	Regard : string
		Direction in witch the 'explorator goes.

	Returns
	-------
	X : int
		Row indice wich is the row where the explorator is after update.
	Y : int
		Column indice wich is the column where the explorator is after update.
	Regard : string
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

	[In 1]: StepLeftHandSolve(Maze, 1, 1, 'E')
	[Out 1]: (1, 2, 'E')

	"""
	if Regard == "E":
		if MazeMap[X-1, Y] == 0:
			Regard = "N"
			X -= 1
		elif MazeMap[X, Y+1] == 0:
			Regard = "E"
			Y += 1
		elif MazeMap[X+1, Y] == 0:
			Regard = "S"
			X += 1
		elif MazeMap[X, Y-1] == 0:
			Regard = "O"
			Y -= 1
	elif Regard == "O":
		if MazeMap[X+1, Y] == 0:
			Regard = "S"
			X += 1
		elif MazeMap[X, Y-1] == 0:
			Regard = "O"
			Y -= 1
		elif MazeMap[X-1, Y] == 0:
			Regard = "N"
			X -= 1
		elif MazeMap[X, Y+1] == 0:
			Regard = "E"
			Y += 1
	elif Regard == "N":
		if MazeMap[X, Y-1] == 0:
			Regard = "O"
			Y -= 1
		elif MazeMap[X-1, Y] == 0:
			Regard = "N"
			X -= 1
		elif MazeMap[X, Y+1] == 0:
			Regard = "E"
			Y += 1
		elif MazeMap[X+1, Y] == 0:
			Regard = "S"
			X += 1
	elif Regard == "S":
		if MazeMap[X, Y+1] == 0:
			Regard = "E"
			Y += 1
		elif MazeMap[X+1, Y] == 0:
			Regard = "S"
			X += 1
		elif MazeMap[X, Y-1] == 0:
			Regard = "O"
			Y -= 1
		elif MazeMap[X-1, Y] == 0:
			Regard = "N"
			X -= 1
	return X, Y, Regard

def WallHandSolve(Base, hand):
	"""
	Solve the maze with the mehod of following continuously the same wall
	while we are out from the starting to the ending node. Note that there
	will probably be multiple 'backward on his steps'.

	Parameters
	----------
	Base : numpy.ndarray
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

	[In 1]: WallHandSolve(Maze, 'l')
	[Out 1]: array([[1, 0], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 4],
					[3, 5], [3, 6], [3, 7], [2, 7], [1, 7], [1, 6], [1, 5],
					[1, 6], [1, 7], [1, 8], [1, 9], [1, 8], [1, 7], [2, 7],
					[3, 7], [4, 7], [5, 7], [5, 8], [5, 9], [4, 9], [3, 9],
					[4, 9], [5, 9], [6, 9], [7, 9], [7, 8], [7, 7], [7, 6],
					[7, 5], [8, 5], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
					[9,10]])

	"""
	SlvHMet =  ['r','R','right','Right','RIGHT','l','L','left','Left','LEFT']
	if hand not in SlvHMet:
		raise ValueError(hand+' is not a possible method: '+SlvHMet)
	Arrete = len(Base)
	Recadr = np.full((Arrete+2, Arrete+2), -1)
	Recadr[1:-1, 1:-1] = Base
	Pos = [2, 1]
	Dir = "E"#la main est donc sur le mur Sud
	End = [np.shape(Recadr)[0]-3, np.shape(Recadr)[1]-2]
	c = 0
	trajet = []
	trajet.append([Pos[0], Pos[1]])
	if hand in ['r', 'R', 'right', 'Right', 'RIGHT']:
		while Pos != End:
			Pos[0], Pos[1], Dir = StepRightHandSolve(Recadr, Pos[0],
													 Pos[1], Dir)
			trajet.append([Pos[0], Pos[1]])
			c += 1
			if c > np.size(Recadr)*2:
				print("BREAK")
				break
	elif hand in ['l', 'L', 'left', 'Left', 'LEFT']:
		while Pos != End:
			Pos[0], Pos[1], Dir = StepLeftHandSolve(Recadr, Pos[0],
													 Pos[1], Dir)
			trajet.append([Pos[0], Pos[1]])
			c += 1
			if c > np.size(Recadr)*2:
				print("BREAK")
				break
	trajet = np.array(trajet)-1
	return trajet

def TriHandSolvePath(Chemin):
	"""
	This function will browse the list of positions in the array Chemin and
	remove all the 'backward' due to dead-end or loop (roundabout).

	Parameters
	----------
	Chemin : numpy.ndarray
		Path from the starting to the ending node returned by WallHandSolve,
		and that probably be multiple 'backward on his steps'.

	Returns
	-------
	trajet : numpy.ndarray
		Path from the starting to the ending node without passing more than
		one time per node.

	Exemple
	-------
	[In 0]: Chemin
	[Out 0]: array([[1, 0], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 4],
					[3, 5], [3, 6], [3, 7], [2, 7], [1, 7], [1, 6], [1, 5],
					[1, 6], [1, 7], [1, 8], [1, 9], [1, 8], [1, 7], [2, 7],
					[3, 7], [4, 7], [5, 7], [5, 8], [5, 9], [4, 9], [3, 9],
					[4, 9], [5, 9], [6, 9], [7, 9], [7, 8], [7, 7], [7, 6],
					[7, 5], [8, 5], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
					[9,10]])

	[In 1]: TriHandSolvePath(Chemin)
	[Out 1]: array([[1, 0], [1, 1], [1, 2], [1, 3], [2, 3], [3, 3], [3, 4],
					[3, 5], [3, 6], [3, 7], [4, 7], [5, 7], [5, 8], [5, 9],
					[6, 9], [7, 9], [7, 8], [7, 7], [7, 6], [7, 5], [8, 5],
					[9, 5], [9, 6], [9, 7], [9, 8], [9, 9], [9,10]])

	"""
	trajet = np.copy(Chemin)
	Stop = False
	while Stop != True:
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
			Stop = True
	return trajet
