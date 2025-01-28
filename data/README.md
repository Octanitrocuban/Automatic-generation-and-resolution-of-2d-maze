Here is an analysis of the generated mazes for the various algorithms.

## Time consumtion
To analyse the time consumtion to generate the maze was done by creating 1 000 random mazes for each method and eache maze size.

Time took to create maze depends of their size and of the algorithm as we can se on the figures below where we plot the natural logarithm of the median of the time to construct the labyrinth versus their size.

This figure contain one example of every algorithm for at leat one of their initialization and/or value of parameters.
![Time took](../img/log_t_evol_med_gener.png)

![Time took](../img/log_t_evol_med_oshift.png)

![Time took](../img/log_t_evol_med_grower.png)

![Time took](../img/log_t_evol_med_bint.png)


## Maze structure analysis
Distribution of nodes with one (dead end), two, three and four connections. The test was done by creating 1 000 random mazes for each method and eache maze size.
![Analysis](../img/distribution_of_connections_1.png)

![Analysis](../img/distribution_of_connections_2.png)

![Analysis](../img/distribution_of_connections_3.png)

![Analysis](../img/distribution_of_connections_4.png)
