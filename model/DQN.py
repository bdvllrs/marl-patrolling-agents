import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):

    def __init__(self, h):
        super(DQN, self).__init__()
        self.linear_input_size = (h * h)
        # self.head = nn.Linear(self.linear_input_size, 100)  # 448 or 512
        # self.head2 = nn.Linear(100, 9)  # 448 or 512
        self.net = nn.Sequential(
            nn.Linear(self.linear_input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 9),
            nn.Softmax()
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # x = self.head2(self.head(x))
        # print(x.shape)
        return self.net(x)
