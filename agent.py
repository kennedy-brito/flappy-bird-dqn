import yaml
import torch
import itertools
import gymnasium
from dqn import DQN
import flappy_bird_gymnasium
from experience_replay import ReplayMemory


device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent:

  def __init__(self, hyperparameter_set) -> None:
    
    with open("hyperparameters.yml", 'r') as file:
      all_hyperparameters_sets = yaml.safe_load(file)
      hyperparameters = all_hyperparameters_sets[hyperparameter_set]

    # Define os atributos com base nos hiperparâmetros
    self.env_id             = hyperparameters.get('env_id')
    self.epsilon_min        = hyperparameters.get('epsilon_min')
    self.epsilon_init       = hyperparameters.get('epsilon_init')
    self.epsilon_decay      = hyperparameters.get('epsilon_decay')
    self.mini_batch_size    = hyperparameters.get('mini_batch_size')
    self.replay_memory_size = hyperparameters.get('replay_memory_size')
    
  def run(self, is_training = True, render=False):
    # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    env = gymnasium.make("CartPole-v1", render_mode="human")

    num_actions = env.action_space.n

    num_states = env.observation_space.shape[0]

    rewards_per_episode = []

    policy_dqn = DQN(num_states, num_actions).to(device)

    # during training we have to know what each action made
    if is_training:
      memory = ReplayMemory(self.replay_memory_size)

    for episode in itertools.count():
      state, _ = env.reset()
      terminated = False
      episode_reward = 0

      while not terminated:
        # Next action
        # (feed the observation to your agent here)
        action = env.action_space.sample()

        # processing
        new_state, reward, terminated, _, info = env.step(action)

        # accumulate reward
        episode_reward += reward

        if is_training:
          memory.append((state, action, new_state, reward, terminated))
        
        # move to new state
        state = new_state

      rewards_per_episode.append(episode_reward)
    