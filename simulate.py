import gym
from environment import CellWars

CASE_0 = [[3,3], [4,4], [2,5], [3,5], [4,5]]
CASE_1 = [[10,10], [11,11], [11,10], [10,11], [19,19]]
if __name__ == '__main__':
    env = CellWars(grid_size=20, num_players=2, max_turns=1, cells_per_player=5, max_steps=20)
    env.reset()
    env.render()
    done = False
    player = 0
    i = 0
    j = 0
    while not done:
        print(f"{player = }\n{i = }\n{j = }\n")
        if i < len(CASE_0) and player == 0:
            print(f"*Here {player = }")
            action = CASE_0[i]
            i += 1
            player = 1
        elif j < len(CASE_1) and player == 1:
            print(f"-Here {player = }")
            action = CASE_1[j]
            j += 1
            player = 0 
        else:
            action = env.action_space.sample()
        # action = env.action_space.sample()
        print(f"Action: {action}")
        observation, reward, done, info = env.step(action)
        if info['placement']:
            env.render()
            # env.render_text()
        print(f"Reward: {reward}")
        print(f"Player Stats: {env.player_stats}")
        print()