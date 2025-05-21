# --- train_bot.py ---
import os
import math
import logging
import random

import torch
import torch.nn as nn
import torch.optim as optim
import chess

# -------------------- Constants --------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 0.01  # learning rate
MCTS_SIMS = 200

# -------------------- Logging Setup --------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# -------------------- Feature Extraction --------------------
def state_to_tensor(board: chess.Board) -> torch.Tensor:
    b = board.copy()
    if b.turn == chess.BLACK:
        b = b.mirror()
    arr = torch.zeros(12 * 64, dtype=torch.float32)
    for sq, piece in b.piece_map().items():
        ch = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        idx = ch * 64 + sq
        arr[idx] = 1.0
    return arr

# -------------------- Neural Network --------------------
class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12 * 64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self._init_weights()

    def _init_weights(self):
        self.fc1.weight.data.normal_(0.0, 1e-3)
        self.fc1.bias.data.zero_()
        # init first two nodes with material sums
        values = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
        w0 = torch.zeros(12 * 64)
        for ch in range(6):
            w0[ch*64:(ch+1)*64] = values[ch]
        self.fc1.weight.data[0] = w0
        self.fc1.bias.data[0] = 0.0
        w1 = torch.zeros(12 * 64)
        for ch in range(6, 12):
            w1[ch*64:(ch+1)*64] = values[ch-6]
        self.fc1.weight.data[1] = w1
        self.fc1.bias.data[1] = 0.0

        self.fc2.weight.data.normal_(0.0, 1e-3)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data[0, 0] = 1.0
        self.fc2.weight.data[1, 1] = 1.0

        self.fc3.weight.data.normal_(0.0, 1e-3)
        self.fc3.bias.data.zero_()
        self.fc3.weight.data[0, 0] = 1.0
        self.fc3.weight.data[0, 1] = -1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        out = self.fc3(h2)
        return torch.tanh(out)

# -------------------- MCTS --------------------
class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0

    def is_leaf(self):
        return not self.children

    def expand(self):
        for mv in self.board.legal_moves:
            nb = self.board.copy()
            nb.push(mv)
            self.children[mv] = MCTSNode(nb, self)

def uct_score(parent: MCTSNode, child: MCTSNode, c_puct):
    return child.Q + c_puct * math.sqrt(parent.N) / (1 + child.N)

def _get_path(node: MCTSNode):
    path = []
    while node:
        path.append(node)
        node = node.parent
    return path

class MCTS:
    def __init__(self, net: ChessNet, sims=MCTS_SIMS, c_puct=math.sqrt(2), device=DEVICE):
        self.net = net.to(device)
        self.sims = sims
        self.c_puct = c_puct
        self.device = device

    def search(self, root_board: chess.Board) -> MCTSNode:
        root = MCTSNode(root_board)
        root.expand()
        for _ in range(self.sims):
            node = root
            while not node.is_leaf():
                mv, node = max(node.children.items(),
                               key=lambda kv: uct_score(node, kv[1], self.c_puct))
            value = self._evaluate_and_expand(node)
            for nd in reversed(_get_path(node)):
                nd.N += 1
                nd.W += value
                nd.Q = nd.W / nd.N
                value = -value
        return root

    def _evaluate_and_expand(self, node: MCTSNode) -> float:
        board = node.board
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            if outcome.termination == chess.Termination.CHECKMATE:
                return 1.0 if outcome.winner else -1.0
            return 0.0

        node.expand()
        feat = state_to_tensor(board).to(self.device).unsqueeze(0)
        with torch.no_grad():
            raw_val = self.net(feat).cpu().item()

        # convert to "white-perspective" eval: multiply by -1 if black to move
        adj_val = raw_val if board.turn == chess.WHITE else -raw_val
        logger.info(f"Evaluated position (white-perspective): {adj_val:.4f}")

        # but for MCTS update we return raw_val (perspective of current player)
        return raw_val

# -------------------- Training Helpers --------------------
def train_on_batch(model, optimizer, batch):
    fens, zs = zip(*batch)
    states = torch.stack([state_to_tensor(chess.Board(f)) for f in fens]).to(DEVICE)
    target_vals = torch.tensor(zs, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    model.train()
    optimizer.zero_grad()
    pred = model(states)
    loss = nn.MSELoss()(pred, target_vals)
    loss.backward()
    optimizer.step()
    return loss.item()

# -------------------- Self-Play Loop --------------------
def selfplay_train_loop():
    model = ChessNet().to(DEVICE)
    if os.path.exists('best.pth'):
        model.load_state_dict(torch.load('best.pth', map_location=DEVICE))
        logger.info("Loaded initial model from best.pth")
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    while True:
        board = chess.Board()
        mcts = MCTS(model, sims=MCTS_SIMS, device=DEVICE)
        history = []

        while not board.is_game_over(claim_draw=True):
            root = mcts.search(board)

            # ε-greedy: pick second-best 20% of the time
            sorted_children = sorted(root.children.items(),
                                     key=lambda kv: kv[1].N,
                                     reverse=True)
            best_move = sorted_children[0][0]
            second_move = sorted_children[1][0] if len(sorted_children) > 1 else best_move
            if random.random() < 0.20:
                move = second_move
                logger.debug(f"Played second-best move: {move}")
            else:
                move = best_move

            history.append(board.fen())
            board.push(move)

        outcome = board.outcome(claim_draw=True)
        term = outcome.termination.name

        # base_z is +1 if white won, -1 if black won, 0 on draw
        if outcome.termination == chess.Termination.CHECKMATE:
            base_z = 1.0 if outcome.winner else -1.0
        else:
            base_z = 0.0

        result_str = ("win" if base_z == 1.0
                      else "loss" if base_z == -1.0
                      else "draw")
        logger.info(f"Game ended in {result_str} (base_z={base_z}), termination reason: {term}")

        # build batch with alternating targets: [base_z, -base_z, base_z, …]
        batch = []
        for i, fen in enumerate(history):
            z_i = base_z * ((-1) ** i)
            batch.append((fen, z_i))

        loss = train_on_batch(model, optimizer, batch)
        logger.info(f"Trained on {len(batch)} positions, loss={loss:.4f}")

        torch.save(model.state_dict(), 'best.pth')
        logger.info("Saved updated best.pth")



if __name__ == '__main__':
    selfplay_train_loop()