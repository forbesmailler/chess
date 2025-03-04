import torch
import torch.nn as nn
import torch.nn.functional as F

# Configurable hyperparameters
CONV_CHANNELS = [4, 8]         # For faster training you might try fewer channels (e.g. [16, 32])
KERNEL_SIZE = 3                # Common choice is 3, try larger sizes if needed
PADDING = 1                    # With kernel size 3, padding of 1 preserves the spatial dimensions
FC_HIDDEN_LAYERS = [64]        # You can reduce to [256] for a lighter model and faster training

# The ChessCNN model with configurable hyperparameters
class ChessCNN(nn.Module):
    def __init__(self, conv_channels=CONV_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, fc_hidden_layers=FC_HIDDEN_LAYERS):
        super(ChessCNN, self).__init__()
        # The board state is 768 features reshaped into (12, 8, 8)
        self.conv_layers = nn.ModuleList()
        in_channels = 12  # 6 piece types * 2 colors
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding))
            in_channels = out_channels
        # Assuming the board remains 8x8 after conv layers (if kernel_size=3 and padding=1)
        conv_output_size = 8 * 8 * conv_channels[-1]
        # Extra 12 features are concatenated
        fc_input_size = conv_output_size + 12

        self.fc_layers = nn.ModuleList()
        for hidden_size in fc_hidden_layers:
            self.fc_layers.append(nn.Linear(fc_input_size, hidden_size))
            fc_input_size = hidden_size
        self.output_layer = nn.Linear(fc_input_size, 3)  # 3 classes: win, draw, loss

    def forward(self, x):
        # x shape: (N, 780)
        board = x[:, :768]   # First 768 features for board state
        extras = x[:, 768:]  # Next 12 features (en passant + castling)
        
        # Reshape board to (N, 12, 8, 8)
        board = board.view(-1, 12, 8, 8)
        for conv in self.conv_layers:
            board = F.relu(conv(board))
        board = board.view(board.size(0), -1)
        
        # Concatenate flattened board features with extra features
        x_combined = torch.cat([board, extras], dim=1)
        for fc in self.fc_layers:
            x_combined = F.relu(fc(x_combined))
        logits = self.output_layer(x_combined)
        return logits