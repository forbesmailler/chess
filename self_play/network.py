# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import NUM_ACTIONS

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ChessNet(nn.Module):
    def __init__(self,
                 in_channels=13,
                 hidden_channels=64,
                 num_res_blocks=4,
                 board_size=8,
                 num_actions: int = NUM_ACTIONS):
        """
        in_channels=12 piece-planes + 1 side-to-move
        num_actions= max UCI moves (8x8x8x8 + promotions) ≈4672
        """
        super().__init__()
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(hidden_channels) for _ in range(num_res_blocks)]
        )
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, num_actions)
        )
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.conv_init(x))
        for blk in self.res_blocks:
            x = blk(x)
        policy = self.policy_head(x)
        value  = self.value_head(x)
        return policy, value
