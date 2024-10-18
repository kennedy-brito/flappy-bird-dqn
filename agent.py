import torch
from torch import nn

import gymnasium
import flappy_bird_gymnasium

import yaml
import random

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from dqn import DQN
from experience_replay import ReplayMemory

import itertools
from datetime import datetime, timedelta
import argparse
import os

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# "Agg": used to generate plots as image and save them to a file instead of rendering to screen

device = "cuda" if torch.cuda.is_available() else "cpu"

# Deep Q-Learning Agent
class Agent:

  def __init__(self, hyperparameter_set) -> None:
    
    with open("hyperparameters.yml", 'r') as file:
      all_hyperparameters_sets = yaml.safe_load(file)
      hyperparameters = all_hyperparameters_sets[hyperparameter_set]
      # print(hyperparameters)

    self.hyperparameter_set = hyperparameter_set
    # Define os atributos com base nos hiperparâmetros
    self.env_id             = hyperparameters.get('env_id')
    self.fc1_nodes          = hyperparameters.get('fc1_nodes')            # nodes of the hidden layer
    self.epsilon_min        = hyperparameters.get('epsilon_min')
    self.epsilon_init       = hyperparameters.get('epsilon_init')
    self.epsilon_decay      = hyperparameters.get('epsilon_decay')        #epsilon
    self.learning_rate      = hyperparameters.get('learning_rate')        # alpha
    self.stop_on_reward     = hyperparameters.get('stop_on_reward')       # stop training after reaching this number of rewards
    self.mini_batch_size    = hyperparameters.get('mini_batch_size')
    self.discount_factor_g  = hyperparameters.get('discount_factor')      # gamma
    self.enable_double_dqn  = hyperparameters.get('enable_double_dqn')    # enables the use of the double dqn method
    self.network_sync_rate  = hyperparameters.get('network_sync_rate')
    self.replay_memory_size = hyperparameters.get('replay_memory_size')
    self.env_make_params    = hyperparameters.get('env_make_params', {})  # get optional environment-specific parameters, default to empty dict

    # neural network
    self.loss_fn = nn.MSELoss() # Mean Squared Error
    self.optimizer = None # initialize later
    
    # Path to run info
    self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
    self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
    self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

  def run(self, is_training = True, render=False):
    
    if is_training:
      start_time = datetime.now()
      last_graph_update_time = datetime.now()
      
      log_message = f'{start_time.strftime(DATE_FORMAT)}: Training starting...'
      print(log_message)

      with open(self.LOG_FILE, 'w') as file:
        file.write(log_message + '\n')

    # env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
    env = gymnasium.make(self.env_id, render_mode="human", **self.env_make_params)

    num_actions = env.action_space.n

    num_states = env.observation_space.shape[0]

    rewards_per_episode = []

    # create policy and target network, Number of nodes can be adjusted
    policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

    # during training we have to know what each action made
    if is_training:
      memory = ReplayMemory(self.replay_memory_size)

      epsilon = self.epsilon_init

      # we create a target network, she is the model who will constantly change
      # because of this she is unstable
      target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)
      
      # we sync their policy's
      # this should be redone a number of times, normally we sync after a number of episodes
      target_dqn.load_state_dict(policy_dqn.state_dict())


      # policy optimizer. "Adam" can be swapped by something else
      self.optimizer = torch.optim.Adam(
        policy_dqn.parameters(), 
        lr=self.learning_rate)


      # track number of steps taken, used to syncing policy => target network
      step_count = 0
      
      epsilon_history = []
      
      # track best reward
      best_reward = -999999
    else:
      # Load learned policy
      policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

      # switch model to evaluation mode
      policy_dqn.eval()

    # train INDEFINITELY, manually stop when you are satisfied or unsatisfied with the results
    for episode in itertools.count():
      
      state, _ = env.reset()

      # torch operates in tensor, so we have to convert the state
      state = torch.tensor(state, dtype=torch.float, device=device)
      
      terminated = False
      episode_reward = 0

      # Perform actions until episode terminates or reaches max rewards
      # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
      while not terminated and episode_reward < self.stop_on_reward:
        
        # decides based in epsilon-greedy if the next action must be random or not
        if is_training and random.random() < epsilon:
          action = env.action_space.sample()
          action = torch.tensor(action, dtype=torch.int64, device=device)

        else:
          # select best action
          with torch.no_grad():
            # gets the action index who has the maximum reward based in current state
            action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
          

        # Execute action
        new_state, reward, terminated, truncated, info = env.step(action.item())

        # accumulate reward
        episode_reward += reward

        #converts new state and reward to tensors on device
        new_state = torch.tensor(new_state, dtype=torch.float, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)
        
        if is_training:
          memory.append((state, action, new_state, reward, terminated))

          step_count+=1
        
        
        # move to new state
        state = new_state

      # keep track of rewards collected per episode
      rewards_per_episode.append(episode_reward)

      # save model when new best reward is obtained
      if is_training:
        if episode_reward > best_reward:
          log_message = f'{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model...'

          print(log_message)

          with open(self.LOG_FILE, 'a') as file:
            file.write(log_message + '\n')

            torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
            best_reward = episode_reward

        # update graph every x second
        current_time = datetime.now()

        if current_time - last_graph_update_time > timedelta(seconds=10):
          self.save_graph(rewards_per_episode, epsilon_history)
          last_graph_update_time = current_time


        # if we have enough experience has been collected
        if len(memory) > self.mini_batch_size:

          # sample from memory
          mini_batch = memory.sample(self.mini_batch_size)

          self.optimize(mini_batch, policy_dqn, target_dqn)

          # we decrease epsilon
          epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
          epsilon_history.append(epsilon)

          # copy policy network to target network after a certain number of steps
          if step_count > self.network_sync_rate:
            target_dqn.load_state_dict(policy_dqn.state_dict())
            step_count = 0

  def optimize(self, mini_batch, policy_dqn, target_dqn):

    # transpose the list of experience and separate each element
    states, actions, new_states, rewards, terminations = zip(*mini_batch)

    # stack tensors to create batch tensors
    # tensor([ [1, 2, 3] ])
    states        = torch.stack(states)
    actions       = torch.stack(actions)
    rewards       = torch.stack(rewards)
    new_states    = torch.stack(new_states)
    terminations  = torch.tensor(terminations).float().to(device)
    
    with torch.no_grad():
      if self.enable_double_dqn:
        best_action_from_policy = policy_dqn(states).argmax(dim=1)

        target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1).gather(dim=1, index=best_action_from_policy(dim=1)).squeeze()
      else:
        target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
        '''
          target_dqn(new_states)  => tensor([ [1, 2, 3], [4, 5, 6] ])
            .max(dim=1)           => torch.return_types.max(values=tensor([3, 6]), indices=tensor([3, 0, 0, 1]))
              [0]                 => tensor([3, 6])
        '''
    
    #calculates Q values from current policy
    current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
    '''
    policy_dqn(states)    => tensor([[1, 2, 3], [4, 5, 6]])
      actions.unsqueeze(dim=1)
      .gather(1, )index actions.unsqueeze(dim=1) => 
        .squeeze => 
    '''

    # compute loss for the whole minibatch
    loss = self.loss_fn(current_q, target_q)

    # optimize the model
    self.optimizer.zero_grad()  # clear gradients
    loss.backward()            # compute gradients (backpropagation)
    self.optimizer.step()       # update network parameters i.e. weight and bias

  def save_graph(self, rewards_per_episode, epsilon_history):
    # Save plots
    fig = plt.figure(1)

    # Plot average rewards (Y-axis) vs episodes (X-axis)
    mean_rewards = np.zeros(len(rewards_per_episode))
    for x in range(len(mean_rewards)):
        mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
    
    plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
    # plt.xlabel('Episodes')
    plt.ylabel('Mean Rewards')
    plt.plot(mean_rewards)

    # Plot epsilon decay (Y-axis) vs episodes (X-axis)
    plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
    # plt.xlabel('Time Steps')
    plt.ylabel('Epsilon Decay')
    plt.plot(epsilon_history)

    plt.subplots_adjust(wspace=1.0, hspace=1.0)

    # Save plots
    fig.savefig(self.GRAPH_FILE)
    plt.close(fig)

if __name__ == "__main__":
  # Parse command line inputs
  parser = argparse.ArgumentParser(description='Train or test model.')
  parser.add_argument('hyperparameters', help='')
  parser.add_argument('--train', help='Training mode', action='store_true')
  args = parser.parse_args()

  dql = Agent(hyperparameter_set=args.hyperparameters)

  if args.train:
      dql.run(is_training=True)
  else:
      dql.run(is_training=False, render=True)