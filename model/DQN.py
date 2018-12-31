import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        linear_input_size = 7*7
        self.head = nn.Linear(linear_input_size, 9) # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return self.head(x.view(-1, 49))