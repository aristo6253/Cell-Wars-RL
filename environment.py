import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation

class CellWars(gym.Env):
    def __init__(self, grid_size=20, num_players=2, max_turns=1, cells_per_player=10, max_steps=100):
        super(CellWars, self).__init__()
        self.grid_size = grid_size
        self.num_players = num_players
        self.max_turns = max_turns
        self.cells_per_player = cells_per_player
        self.max_steps = max_steps

        # The action space is a tuple (x, y) representing a position on the grid
        self.action_space = spaces.MultiDiscrete([grid_size, grid_size])
        # The observation space is the grid itself, with a separate channel for each player
        self.observation_space = spaces.Box(low=0, high=num_players, shape=(grid_size, grid_size), dtype=np.int32)

        self.reset()


    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.ownership = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.turn = 0
        self.current_player = 0
        self.cells_placed = np.zeros(self.num_players, dtype=np.int32)
        self.simulation_steps = 0
        self.history = [self.get_observation().copy()]
        self.player_stats = {player: {'current_cells': 0, 'survival_time': 0} for player in range(1, self.num_players + 1)}
        return self.get_observation()
    
    def step(self, action):
        x, y = action
        # print(f"Player {self.current_player+1} placed his {self.cells_placed[self.current_player]+1} cell at ({x}, {y})")

        if self.is_valid_action(x, y):
            # print("Valid action")
            self.grid[x, y] = 1
            self.ownership[x, y] = self.current_player + 1
            self.cells_placed[self.current_player] += 1

            # Switch to the next player
            self.current_player = (self.current_player + 1) % self.num_players
        # else:
        #     print("Invalid action")

        # This should be outside of the step function
        placement_done = np.all(self.cells_placed == self.cells_per_player)
        if placement_done:
            print(f"Simulation step {self.simulation_steps + 1} / {self.max_steps}")
            self.play_game_of_life()
            self.simulation_steps += 1
            self.turn += 1

        # Update player statistics
        for player in range(1, self.num_players + 1):
            current_cells = np.sum(self.ownership == player)
            if current_cells > 0:
                self.player_stats[player]['current_cells'] = current_cells
                self.player_stats[player]['survival_time'] = self.simulation_steps
            else:
                self.player_stats[player]['current_cells'] = 0

        reward = self.calculate_reward(placement_done) if placement_done else 0
        done = self.simulation_steps >= self.max_steps

        return self.grid, reward, done, {'placement': placement_done}
    
    def is_valid_action(self, x, y):
        if self.grid[x, y] == 0 and self.cells_placed[self.current_player] < self.cells_per_player:
            return True
        return False
    
    def play_game_of_life(self,):
        new_grid = np.zeros_like(self.grid)
        new_ownership = np.zeros_like(self.ownership)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                cell_data = self.get_cell_data(x, y)
                num_neighbors = sum(cell_data['neighbors'].values()) - cell_data['state']

                if cell_data['state'] == 1 and (num_neighbors == 2 or num_neighbors == 3):
                    new_grid[x, y] = 1
                    new_ownership[x, y] = cell_data['ownership']
                elif cell_data['state'] == 0 and num_neighbors == 3:
                    new_grid[x, y] = 1
                    majority_owner = max(cell_data['neighbors'], key=cell_data['neighbors'].get)
                    new_ownership[x, y] = majority_owner

        self.grid = new_grid
        self.ownership = new_ownership

        self.history.append(self.get_observation().copy())

    def get_cell_data(self, x, y):
        data = {
            'state': self.grid[x, y],
            'ownership': self.ownership[x, y],
            'neighbors': {player: 0 for player in range(1, self.num_players + 1)}
        }

        for player in range(1, self.num_players + 1):
            data['neighbors'][player] = np.sum(self.ownership[max(0, x-1):min(self.grid_size, x+2), max(0, y-1):min(self.grid_size, y+2)] == player)

        return data
        

    def calculate_reward(self, placement_done):
        # Reward function: base reward for remaining cells and extra rewards based on ranking
        rewards = np.zeros(self.num_players)
        
        if placement_done:
            for player in range(1, self.num_players + 1):
                current_cells = self.player_stats[player]['current_cells']
                survival_time = self.player_stats[player]['survival_time']
                extinction_penalty = -10 if current_cells == 0 else 0
                
                # Base reward: number of remaining cells
                base_reward = current_cells
                
                # Survival time bonus
                survival_bonus = survival_time
                
                # Total reward
                rewards[player - 1] = base_reward + survival_bonus + extinction_penalty
        
        return rewards[self.current_player]

    def get_observation(self):
        # Combine grid and ownership for observation
        return self.ownership

    def render_text(self):
        """
        Render the grid and ownership information as text.

        This function creates a copy of the grid and iterates over each cell. If the ownership of a cell is not zero,
        it updates the corresponding cell in the combined grid with the ownership value. Then, it prints each row of
        the combined grid, replacing zero values with a dot. Finally, it prints a newline character.

        This function does not take any parameters and does not return any values.
        """
        # Simple print-based rendering
        combined_grid = self.grid.copy()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.ownership[i, j] != 0:
                    combined_grid[i, j] = self.ownership[i, j]
        for row in combined_grid:
            print(' '.join(['.' if cell == 0 else str(cell) for cell in row]))
        print()

    def render(self, mode='human'):
        # Rendering using matplotlib
        cmap = ListedColormap(['white'] + [plt.cm.tab10(i) for i in range(self.num_players)])
        fig, ax = plt.subplots()
        cax = ax.matshow(self.ownership, cmap=cmap)
        
        # Create a colorbar with player numbers
        cbar = fig.colorbar(cax, ticks=range(self.num_players + 1))
        cbar.ax.set_yticklabels(['None'] + [f'Player {i+1}' for i in range(self.num_players)])
        
        plt.show()