# --- train_bot.py ---
import os
import math
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
import chess
from collections import deque

# -------------------- Constants --------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 0.01
MCTS_SIMS = 200

# -------------------- Logging Setup --------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# -------------------- Feature Extraction --------------------
def state_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Returns flattened 12x64 binary representation (no mirroring);
    evaluation sign handled in MCTS based on player_color.
    """
    arr = torch.zeros(12 * 64, dtype=torch.float32)
    for sq, piece in board.piece_map().items():
        ch = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        arr[ch * 64 + sq] = 1.0
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
        vals = torch.tensor([1.0,3.0,3.0,5.0,9.0,0.0])
        w0 = vals.repeat_interleave(64)
        w1 = vals.repeat_interleave(64)
        w1 = torch.cat([torch.zeros(64*6), w1])
        self.fc1.weight.data[0] = w0
        self.fc1.bias.data[0] = 0.0
        self.fc1.weight.data[1] = w1
        self.fc1.bias.data[1] = 0.0

        self.fc2.weight.data.normal_(0.0, 1e-3)
        self.fc2.bias.data.zero_()
        self.fc2.weight.data[0,0] = 1.0
        self.fc2.weight.data[1,1] = 1.0

        self.fc3.weight.data.normal_(0.0, 1e-3)
        self.fc3.bias.data.zero_()
        self.fc3.weight.data[0,0] = 1.0
        self.fc3.weight.data[0,1] = -1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        return torch.tanh(self.fc3(h2))

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
    def __init__(self, net: ChessNet, player_color: bool, sims=MCTS_SIMS, c_puct=math.sqrt(2), device=DEVICE):
        self.net = net.to(device)
        self.player_color = player_color
        self.sims = sims
        self.c_puct = c_puct
        self.device = device

    def search(self, root_board: chess.Board, root: MCTSNode=None) -> MCTSNode:
        if root is None or root.board.fen() != root_board.fen():
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
            raw = self.net(feat).cpu().item()
        # Flip sign if side to move != mcts player
        return raw if board.turn == self.player_color else -raw

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

    # Two MCTS instances, one per color
    white_mcts = MCTS(model, player_color=chess.WHITE, sims=MCTS_SIMS, c_puct=math.sqrt(2), device=DEVICE)
    black_mcts = MCTS(model, player_color=chess.BLACK, sims=MCTS_SIMS, c_puct=math.sqrt(2), device=DEVICE)
    white_root = None
    black_root = None

    while True:
        board = chess.Board()
        history = []
        ply = 0
        while not board.is_game_over(claim_draw=True):
            # Log current eval (white perspective)
            feat = state_to_tensor(board).to(DEVICE).unsqueeze(0)
            with torch.no_grad(): raw_val = model(feat).cpu().item()
            adj = raw_val if board.turn == chess.WHITE else -raw_val
            logger.info(f"Self-play eval after move {ply+1}: {adj:.4f}")

            sims = max(50, int(MCTS_SIMS * (1 - ply / 200)))
            if board.turn == chess.WHITE:
                mcts = white_mcts
                white_mcts.sims = sims
                white_root = mcts.search(board, white_root)
                root = white_root
            else:
                mcts = black_mcts
                black_mcts.sims = sims
                black_root = mcts.search(board, black_root)
                root = black_root

            # Choose move
            children = sorted(root.children.items(), key=lambda kv: kv[1].N, reverse=True)
            best_move = children[0][0]
            second_move = children[1][0] if len(children) > 1 else best_move
            move = second_move if random.random() < 0.40 else best_move
            if move == second_move:
                logger.debug(f"Played second-best move: {move}")

            history.append(board.fen())
            board.push(move)
            ply += 1
            # Advance the corresponding root
            if board.turn == chess.BLACK:
                white_root = white_root.children.get(move)
                if white_root: white_root.parent = None
            else:
                black_root = black_root.children.get(move)
                if black_root: black_root.parent = None

        outcome = board.outcome(claim_draw=True)
        term = outcome.termination.name
        base_z = 1.0 if (outcome.termination == chess.Termination.CHECKMATE and outcome.winner) else (-1.0 if outcome.termination == chess.Termination.CHECKMATE else 0.0)
        logger.info(f"Game ended: {term}, base_z={base_z}")

        batch = [(fen, base_z * ((-1) ** i)) for i, fen in enumerate(history)]
        loss = train_on_batch(model, optimizer, batch)
        logger.info(f"Trained on {len(batch)} positions, loss={loss:.4f}")

        # Calibrate final bias
        init_board = chess.Board()
        feat0 = state_to_tensor(init_board).to(DEVICE).unsqueeze(0)
        with torch.no_grad(): raw0 = model(feat0).cpu().item()
        pre_act = torch.atanh(torch.tensor(raw0, device=DEVICE))
        model.fc3.bias.data -= pre_act
        torch.save(model.state_dict(), 'best.pth')
        logger.info("Saved updated best.pth")

if __name__ == '__main__':
    selfplay_train_loop()
