import torch
import torch.nn as nn
import torch.optim as optim

class PolicyValueNetwork(nn.Module):
    def __init__(self, board_size):
        super(PolicyValueNetwork, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc_policy = nn.Linear(128 * board_size * board_size, board_size * board_size)
        self.fc_value = nn.Linear(128 * board_size * board_size, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = x.view(-1, 128 * self.board_size * self.board_size)
        policy = self.fc_policy(x)
        value = torch.tanh(self.fc_value(x))
        return policy, value
