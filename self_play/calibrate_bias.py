# --- calibrate_bias.py ---
import torch
import chess
from train_bot import ChessNet, state_to_tensor

# Load model
model = ChessNet()
state_dict = torch.load('best.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# Prepare initial position
board = chess.Board()
feat = state_to_tensor(board).unsqueeze(0)

# Compute current evaluation
with torch.no_grad():
    raw_out = model(feat).item()
# Compute pre-activation adjustment
pre_act = torch.atanh(torch.tensor(raw_out))

# Adjust final layer bias so that eval becomes 0
model.fc3.bias.data -= pre_act

# Save calibrated model
torch.save(model.state_dict(), 'best_calibrated.pth')
print(f"Adjusted final bias by {pre_act.item():.6f}")
# Verify new evaluation
with torch.no_grad():
    new_out = model(feat).item()
print(f"New eval of initial position: {new_out:.6f}")