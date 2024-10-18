import yaml
import torch
import random
import itertools
import gymnasium
from dqn import DQN
from torch import nn
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
    self.learning_rate      = hyperparameters.get('learning_rate')
    self.mini_batch_size    = hyperparameters.get('mini_batch_size')
    self.discount_factor_g    = hyperparameters.get('discount_factor')
    self.network_sync_rate  = hyperparameters.get('network_sync_rate')
    self.replay_memory_size = hyperparameters.get('replay_memory_size')


    self.loss_fn = nn.MSELoss() # Mean Squared Error
    self.optimizer = None # initialize later
    
  def run(self, is_training = True, render=False):
    # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    env = gymnasium.make("CartPole-v1", render_mode="human")

    num_actions = env.action_space.n

    num_states = env.observation_space.shape[0]

    rewards_per_episode = []
    epsilon_history = []

    policy_dqn = DQN(num_states, num_actions).to(device)

    # during training we have to know what each action made
    if is_training:
      memory = ReplayMemory(self.replay_memory_size)

      epsilon = self.epsilon_init

      # we create a target network, she is the model who will constantly change
      # because of this she is unstable
      target_dqn = DQN(num_states, num_actions).to(device)
      
      # we sync their policy's
      # this should be redone a number of times, normally we sync after a number of episodes
      target_dqn.load_state_dict(policy_dqn.state_dict())

      # track number of steps taken, used to syncing policy => target network
      step_count = 0

      # policy optimizer. "Adam" can be swapped by something else
      self.optimizer = torch.optin.Adam(
        policy_dqn.parameters(), 
        lr=self.learning_rate)


    for episode in itertools.count():
      state, _ = env.reset()

      # torch operates in tensor, so we have to convert the state
      state = torch.tensor(state, dtype=torch.float, device=device)
      
      terminated = False
      episode_reward = 0

      while not terminated:
        


        # Next action
        # (feed the observation to your agent here)
        # decides based in epsilon-greedy if the next action must be random or not
        if is_training and random.random() < epsilon:
          action = env.action_space.sample()
          action = torch.tensor(action, dtype=torch.int64, device=device)

        else:
          with torch.no_grad():
            # gets the action index who has the maximum reward based in current state
            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
          

        # processing
        new_state, reward, terminated, _, info = env.step(action.item())

        # accumulate reward
        episode_reward += reward

        if is_training:
          memory.append((state, action, new_state, reward, terminated))

          step_count+=1
        
        #converts to tensor
        new_state = torch.tensor(new_state, dtype=torch.float, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)
        
        # move to new state
        state = new_state

      rewards_per_episode.append(episode_reward)

      # we decrease epsilon in the end of each episode

      epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
      epsilon_history.append(epsilon)

      # if we have enough experience has been collected
      if len(memory) > self.mini_batch_size:

        # sample from memory
        mini_batch = memory.sample(self.mini_batch_size)

        self.optimize(mini_batch, policy_dqn, target_dqn)

        # copy policy network to target network after a certain number of steps
        if step_count > self.network_sync_rate:
          target_dqn.load_state_dict(policy_dqn.state_dict())
          step_count = 0

  def optimize(self, mini_batch, policy_dqn, target_dqn):

    for state, action, new_state, reward, terminated in mini_batch:
      
      if terminated:
        target_q = reward
      else:
        with torch.no_grad():
          target_q = reward + self.discount_factor_g * target_q(new_state).max()
      
      current_q = policy_dqn(state)

      # compute loss for the whole minibatch
      loss = self.loss_fn(current_q, target_q)

      # optimize the model
      self.optimizer.zero_grad()  # clear gradients
      loss.backwards()            # compute gradients (backpropagation)
      self.optimizer.step()       # update network parameters i.e. weight and bias

if __name__ == "__main__":
  agent = Agent('cartpole1')
  agent.run(is_training=True, render=True)