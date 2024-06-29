import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from environment import CellWars

if __name__ == "__main__":
    env = CellWars(grid_size=10, num_players=2, max_turns=1, cells_per_player=4, max_steps=10)
    
    # Check the environment
    check_env(env)

    # Vectorize the environment
    vec_env = DummyVecEnv([lambda: env])

    # Create the PPO model
    model = PPO('MlpPolicy', env, verbose=1)

    # Train the model
    model.learn(total_timesteps=10_000)

    # Save the model
    model.save("ppo_cell_wars")

    # To visualize the trained model
    obs = env.reset()
    for i in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        env.render()