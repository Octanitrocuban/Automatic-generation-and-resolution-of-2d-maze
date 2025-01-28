Here is an analysis of the generated mazes for the various algorithms.

## Time consumtion
To analyse the time consumtion to generate the maze was done by creating 1 000 random mazes for each method and eache maze size.

Time took to create maze depends of their size and of the algorithm as we can se on the figures below where we plot the natural logarithm of the median of the time to construct the labyrinth versus their size.

This figure contain one example of every algorithm for at leat one of their initialization and/or value of parameters. All algorithm shows an increasing trends with their size. At the exception of the binary tree, all method shows a logarithm trend pattern in this x-linear y-log space. The binary tree method seems to follow a linear trend and is in order of magnitude faster than all of the other methods. This comes from the indepent definition of the connections between the nods.

![Time took](../img/log_t_evol_med_gener.png)

This figure shows the two possible initialisation of the origin shift algorithm for three number of iteration. The indication 1, 10 and 100 is the factor by wich the mazes is shuffled. The number of iteration is the factor times the the square of the width of the maze. The initialization does not seems to have any inmpact on the computational time as the curves are overlaping. Thanks to it simplicity within the shuffiling iterations, the increased computational time are mostly linear by the multiplicativ factor.

![Time took](../img/log_t_evol_med_oshift.png)

This figure shows ten possible values for the growing tree algorithm. All of the curve shows logarithm trend pattern, and are -at one execption- thightly packed together. The exception is the 1 growing step value, which is more time expensive than the other. We can also see that the curves show a subtle but systematic decrease in time consumption when increasing the number of growing step. This indicate that the part of growing branch is faster than the sampling part by a small margin.

![Time took](../img/log_t_evol_med_grower.png)

This figure show the five possible initialisation for the binary tree algorithm. All curves -at the exception of the method without initialisation- shows clear logarithm trend pattern and are nerly ovelapping. This behavior difference is coming from the creation of the and not from their initialization. Indeed, I was not able yet to make the nodes connection indepandant as in the method without initialization. The logarithm trend pattern comes from the square augmentation of the number of nodes to visit.

![Time took](../img/log_t_evol_med_bint.png)


## Maze structure analysis
Distribution of nodes with one (dead end), two, three and four connections. The test was done by creating 1 000 random mazes for each method and eache maze size.
![Analysis](../img/distribution_of_connections_1.png)

![Analysis](../img/distribution_of_connections_2.png)

![Analysis](../img/distribution_of_connections_3.png)

![Analysis](../img/distribution_of_connections_4.png)
