# TODO reward function, loss, model 
import numpy as np
import torch
from torch import nn

class DumbModel:
    def __init__(self, possible_actions: list) -> None:
        self.proba = torch.Tensor(np.full(len(possible_actions), 1 / len(possible_actions)))

    def choose_action(self, possible_actions: list):
        return np.random.choice(possible_actions)
    
class PolicyModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, actions_dim: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, actions_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU(x)
        x = self.fc2(x)

        return x

class CtiticModel(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU(x)
        x = self.fc2(x)

        return x
