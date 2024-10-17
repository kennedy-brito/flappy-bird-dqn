import torch
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN


device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:
    
  def run(self, is_training = True, render=False):
    # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    env = gymnasium.make("CartPole-v1", render_mode="human")

    num_actions = env.action_space.n

    num_states = env.observation_space.shape[0]

    policy_dqn = DQN(num_states, num_actions).to(device)

    obs, _ = env.reset()

    while True:
      # Next action
      # (feed the observation to your agent here)
      action = env.action_space.sample()

      # processing
      obs, reward, terminated, _, info = env.step(action)

      # checking if the player is still alive
      if terminated:
        break

    env.close()