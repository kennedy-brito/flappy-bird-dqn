import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

  def __init__(self, state_dim, action_dim, hidden_dim=256):
    super(DQN, self).__init__()

    self.fc1 = nn.Linear(state_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, action_dim)

  def forward(self, state):

    x = F.relu(self.fc1(state))
    actions_q_values = self.fc2(x)

    return actions_q_values
  
if __name__ == "__main__":

  state_dim = 12
  action_dim = 2
  net = DQN(state_dim, action_dim)
  state = torch.randn(1, state_dim)
  output = net(state)
  print(output)