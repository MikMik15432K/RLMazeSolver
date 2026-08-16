# Reinforcement Learning 2D Maze Solver

A Reinforcement Learning project utilizing a Q-Table to solve mazes produced using Prim's maze generation algorithm.


## Core Components

* **Procedural Maze Generation** Prim's algorithm initialized at the grid center.
* **Goal Placement:** Breadth-First Search (BFS) places the target at the leaf node furthest from the start position $(0, 0)$.
* **Q-Learning** Trains Q-Table using e-greedy exploration and decaying learning rates.
* **Path Evaluation** Tests the trained agent without random exploration to verify the optimal path.
* **Visualization** Uses Matplotlib to animate and display the agent navigating through the maze.
