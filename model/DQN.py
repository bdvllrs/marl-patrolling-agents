import torch
from math import floor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class DQN(nn.Module):

    def __init__(self, h):
        super(DQN, self).__init__()
        self.linear_input_size = h

        # self.head = nn.Linear(self.linear_input_size, 100)  # 448 or 512
        # self.head2 = nn.Linear(100, 9)  # 448 or 512
        def get_conv_output_dim(w, kernel_size):
            return floor(w - kernel_size + 1)

        h = get_conv_output_dim(get_conv_output_dim(self.linear_input_size, 5), 5)
        w = get_conv_output_dim(get_conv_output_dim(self.linear_input_size, 5), 5)

        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.Conv2d(16, 1, 5),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(49, 9),
            nn.Softmax(dim=1)
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.unsqueeze(1)
        # x = self.head2(self.head(x))
        # print(x.shape)
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
