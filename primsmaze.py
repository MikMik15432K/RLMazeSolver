import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation


class MazeEnv:
    def __init__(self, width=20, height=20):
        #odd dimensions for primms maze generation
        
        if width % 2 != 0:
            self.width = width 
        else:  
            self.width = width + 1 
        if height % 2 != 0:
            self.height = height 
        else:  
            self.height = height + 1 
        
        self.start_pos = (0, 0)
        self.current_pos = self.start_pos
        
        #initial maze generation
        self.maze = self._generate_maze()
        self.goal_pos = self._find_furthest_goal()

    #position to state id 
    def pos_to_state(self, row, col):
        return row * self.width + col
    #state id to position
    def state_to_pos(self, state_id):
        return state_id // self.width, state_id % self.width

    #maze generation with primms algorithm
    def _generate_maze(self):
        maze = [[1 for _ in range(self.width)] for _ in range(self.height)]
        
        center_r = (self.height // 2) - ((self.height // 2) % 2)
        center_c = (self.width // 2) - ((self.width // 2) % 2)
        maze[center_r][center_c] = 0
        
        frontier = []
        def add_frontier(r, c):
            directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if maze[nr][nc] == 1:
                        frontier.append((nr, nc, r, c))

        add_frontier(center_r, center_c)

        while frontier:
            rand_idx = random.randint(0, len(frontier) - 1)
            tr, tc, pr, pc = frontier.pop(rand_idx)
            if maze[tr][tc] == 1:
                maze[tr][tc] = 0
                maze[(pr + tr) // 2][(pc + tc) // 2] = 0
                add_frontier(tr, tc)

        #making sure goal is accessible
        maze[self.start_pos[0]][self.start_pos[1]] = 0
        if self.start_pos[0] + 1 < self.height and maze[self.start_pos[0] + 1][self.start_pos[1]] == 1:
            if self.start_pos[1] + 1 < self.width and maze[self.start_pos[0]][self.start_pos[1] + 1] == 1:
                maze[self.start_pos[0] + 1][self.start_pos[1]] = 0

        return np.array(maze)

    def _find_furthest_goal(self):
        queue = deque([(self.start_pos[0], self.start_pos[1], 0)])
        visited = {self.start_pos}
        furthest_node = self.start_pos
        max_dist = 0

        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while queue:
            r, c, dist = queue.popleft()
            if dist > max_dist:
                max_dist = dist
                furthest_node = (r, c)

            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if self.maze[nr][nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc, dist + 1))

        return furthest_node

    #interface for rl 
    def reset(self, regenerate_maze=False):
        """Resets agent position to start. Optionally generates a new maze."""
        if regenerate_maze:
            self.maze = self._generate_maze()
            self.goal_pos = self._find_furthest_goal()
        self.current_pos = self.start_pos
        return self.pos_to_state(*self.start_pos)

    def step(self, action):
        """Actions: 0=Up, 1=Down, 2=Left, 3=Right"""
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dr, dc = moves[action]
        r, c = self.current_pos
        nr, nc = r + dr, c + dc

        #colission with wall or out of bounds
        if 0 <= nr < self.height and 0 <= nc < self.width and self.maze[nr][nc] == 0:
            self.current_pos = (nr, nc)

        #goal check
        done = (self.current_pos == self.goal_pos)
        reward = 100.0 if done else -1.0  # Step penalty encourages shortest path

        return self.pos_to_state(*self.current_pos), reward, done

    #visuals
    def plot_grid(self, ax=None, show_agent=False):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))

        cmap = colors.ListedColormap(['white', 'black'])
        bounds = [0, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        ax.imshow(self.maze, cmap=cmap, norm=norm)

        height, width = self.maze.shape
        ax.grid(which='major', axis='both', linestyle='-', color='gray', linewidth=1)
        ax.set_xticks(np.arange(-0.5, width, 1))
        ax.set_yticks(np.arange(-0.5, height, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.add_patch(plt.Rectangle((self.start_pos[1] - 0.5, self.start_pos[0] - 0.5), 1, 1, color='green', alpha=0.5))
        ax.add_patch(plt.Rectangle((self.goal_pos[1] - 0.5, self.goal_pos[0] - 0.5), 1, 1, color='red', alpha=0.5))
        
        if show_agent:
            ax.add_patch(plt.Circle((self.current_pos[1], self.current_pos[0]), 0.3, color='blue'))

        return ax

    def animate_agent_on_grid(self, path_states):
        fig, ax = plt.subplots(figsize=(6, 6))
        self.plot_grid(ax)

        trail_line, = ax.plot([], [], color='royalblue', linewidth=3, zorder=2)
        agent_dot = plt.Circle((self.start_pos[1], self.start_pos[0]), 0.3, color='blue', zorder=3)
        ax.add_patch(agent_dot)

        trace_x = []
        trace_y = []

        def update(frame_state_id):
            # Fixed method call: state_to_pos instead of _get_pos_from_id
            row, col = self.state_to_pos(frame_state_id)
            agent_dot.center = (col, row)
            
            trace_x.append(col)
            trace_y.append(row)
            trail_line.set_data(trace_x, trace_y)

            return trail_line, agent_dot

        ani = animation.FuncAnimation(
            fig, update, frames=path_states, interval=100, repeat=False, blit=False
        )
        
        plt.title("Agent Solution Path")
        plt.show()
        return ani