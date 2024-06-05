import torch
import torch.nn as nn

class AlphaZeroNet(nn.Module):
    def __init__(self, board_size):
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(256 * board_size * board_size, 1024)
        self.fc2_policy = nn.Linear(1024, board_size * board_size)
        self.fc2_value = nn.Linear(1024, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 256 * self.board_size * self.board_size)
        x = torch.relu(self.fc1(x))

        policy = torch.softmax(self.fc2_policy(x), dim=1)
        value = torch.tanh(self.fc2_value(x))

        return policy, value
