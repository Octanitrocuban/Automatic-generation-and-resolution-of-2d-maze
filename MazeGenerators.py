# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 16:40:12 2022

@author: Matthieu Nougaret
"""
#import usefull library
import numpy as np

#=============================================================================

def CreateMazeBase(Arrete):
	"""
	Generate the basis that will be used by the first method to generate
	automatically a maze with the function MazeFormation.

	Parameters
	----------
	Arrete : int
		Width and height of the maze, it have to be between 3 and +inf. Note
		that if you choose an even number, the output will have for width and
		height the higger odd number to the Arrete parameter.

	Returns
	-------
	Base : 2d numpy array of int
		The basis used as input by the function MazeFormation. -1 are walls,
		other values are all unic, from one the start node to 
		n = (sup_odd(Arrete)-1)**2 /4 +2 the end node. The other are the
		starting ground nodes that will be connected.

	Exemple
	-------
	[In]: CreateMazeBase(8)
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
	if Arrete%2 == 0:
		Bande1 = np.full(Arrete+1, -1)
		Bande2 = np.full(Arrete, -1)
		Bande2[::2] = 0
		Bande2 = np.append(-1, Bande2)
		Base = []
		for i in range(Arrete):
			if i%2 == 0:
				Base.append(Bande1)
			else :
				Base.append(Bande2)
		Base.append(Bande1)
		Base = np.array(Base)
	else :
		Bande1 = np.full(Arrete+1, -1)
		Bande2 = np.full(Arrete, -1)
		Bande2[::2] = 0
		Bande2 = np.append(-1, Bande2)
		Base = []
		for i in range(Arrete):
			if i%2 == 0:
				Base.append(Bande1)
			else :
				Base.append(Bande2)
		Base = np.array(Base)[:, :-1]
	Base[1, 0] = 0
	Base[-2, -1] = 0
	Base[Base != -1] = range(1, len(Base[Base != -1])+1)
	return Base

def CreateMazeBaseBoolean(Arrete):
	"""
	Generate the basis that will be used by the second method to generate
	automatically a maze with the function MakeMazeExhaustif.

	Parameters
	----------
	Arrete : int
		Width and height of the maze, it have to be between 3 and +inf. Note
		that if you choose an even number, the output will have for width and
		height the higger odd number to the Arrete parameter.

	Returns
	-------
	Base : 2d numpy array of int
		The basis used as input by the function MakeMazeExhaustif. -1 are
		walls, 0 are the starting ground nodes that will be connected and 1
		are the start and end node.

	Exemple
	-------
	[In]: CreateMazeBaseBoolean(8)
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
	if Arrete%2 == 0:
		Bande1 = np.full(Arrete+1, -1)
		Bande2 = np.full(Arrete, -1)
		Bande2[::2] = 0
		Bande2 = np.append(-1, Bande2)
		Base = []
		for i in range(Arrete):
			if i%2 == 0:
				Base.append(Bande1)
			else :
				Base.append(Bande2)
		Base.append(Bande1)
		Base = np.array(Base)
	else :
		Bande1 = np.full(Arrete+1, -1)
		Bande2 = np.full(Arrete, -1)
		Bande2[::2] = 0
		Bande2 = np.append(-1, Bande2)
		Base = []
		for i in range(Arrete):
			if i%2 == 0:
				Base.append(Bande1)
			else :
				Base.append(Bande2)
		Base = np.array(Base)[:, :-1]
	Base[1, 0] = 1
	Base[-2, -1] = 1
	return Base

def MazeFormation(Base):
	"""
	First method to generate automatically a maze. It take in input the output
	of the function CreateMazeBase. It will randomly draw 2 indices between 1
	and len(Base)-2, then if it is a wall it break it by putting a 0 and the
	newly 2 conected ground node are set at 2. It continue until all the
	ground, starting and ending nodes are not connected together.

	Parameters
	----------
	Base : 2d numpy array of int
		The maze with -1 for wall and from 1 the starting node to 
		(sup_odd(Arrete)-1)**2 /4 +2 the ending node for ground.

	Returns
	-------
	Base : 2d numpy array of int
		The maze with -1 for wall and 0 for ground. At this stage, there is
		one possible path to connect starting and ending node without
		re-borrowing the same node several times.

	Exemple
	-------
	[In]: MazeFormation(CreateMazeBase(8))
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
	Arrete = len(Base)
	while len(np.unique(Base)) != 5:
		Xrand = np.random.randint(1, Arrete-1)
		if Xrand%2 == 0 :
			CH_y = np.arange(1, Arrete, 2)
		else :
			CH_y = np.arange(2, Arrete-1, 2 )
		Yrand = np.random.choice(CH_y)
		Choice = [Xrand, Yrand]
		if Base[Choice[0]+1, Choice[1]] != -1 :
			cellule1 = Base[Choice[0]-1, Choice[1]]
			cellule2 = Base[Choice[0]+1, Choice[1]]
		else :
			cellule1 = Base[Choice[0], Choice[1]-1]
			cellule2 = Base[Choice[0], Choice[1]+1]

		if cellule1 != cellule2:
			Base[Choice[0], Choice[1]] = 0
			for i in range(1, Arrete-1, 2):
				for j in range(1, Arrete-1, 2):
					if Base[i, j] == cellule2:
						Base[i, j] = cellule1
	Base[Base != -1] = 0
	return Base

def MakeMazeExhaustif(Base):
	"""
	Second method to generate automatically a maze. It take in input the
	output of the function CreateMazeBase. It randomly draw a position
	correponding to a ground node and change it's value from 0 to one. Then
	it randomly choose a unvisited other ground node in its four neighbours.
	It break the wall wall with a 1 and set the new gound node position to 1.
	It continue while all of his neighbours are visited. Then if all ground,
	starting and ending nodes are connected it stop, else it take the exact
	path it retraces his steps until he finds a possible passage, and rebreak
	the wall.

	Parameters
	----------
	Base : 2d numpy array of int
		The maze with -1 for wall, 0 for ground and 1 for starting and ending.

	Returns
	-------
	Ret : 2d numpy array of int
		The maze with -1 for wall and 0 for ground. At this stage, there is
		one possible path to connect starting and ending node without
		re-borrowing the same node several times.
		
	Exemple
	-------
	[In]: MakeMazeExhaustif(CreateMazeBaseBoolean(8))
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
	Ret = np.full((Base.shape[0]+4, Base.shape[1]+4), -1)
	Ret[2:-2, 2:-2] = Base
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

def MakeMazeComplex(Base):
	"""
	This function will transform the maze in order that their will multiple
	paths from start to end. To achieve this goal, it randomly break some
	walls that are separating two ground nodes.

	Parameters
	----------
	Base : 2d numpy array of int
		The maze with -1 for wall and 0 for ground.

	Returns
	-------
	Base : 2d numpy array of int
		The maze with -1 for wall and 0 for ground.
		
	Exemple
	-------
	[In 0]: _ = MazeFormation(CreateMazeBase(8))
	[Out 0]: array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
					[ 0,  0, -1,  0, -1,  0, -1,  0, -1],
					[-1,  0, -1,  0, -1,  0, -1,  0, -1],
					[-1,  0, -1,  0, -1,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1, -1, -1,  0, -1],
					[-1,  0, -1,  0,  0,  0,  0,  0, -1],
					[-1,  0, -1,  0, -1,  0, -1, -1, -1],
					[-1,  0,  0,  0, -1,  0,  0,  0,  0],
					[-1, -1, -1, -1, -1, -1, -1, -1, -1]])
	
	[In 1]: MakeMazeComplex(_)
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
	Arrete = len(Base)
	for i in range(Arrete):
		x = np.random.randint(1, Arrete-2)
		if x%2 == 0:
			CH_y = np.arange(1, Arrete, 2)
		else :
			CH_y = np.arange(2, Arrete-1, 2)
		y = np.random.choice(CH_y)
		Base[x, y] = 0
	return Base
