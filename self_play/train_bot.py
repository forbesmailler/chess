import os
import math
import time
import random
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import chess
import torch.cuda.amp as amp

# -------------------- Constants --------------------
FILES = "abcdefgh"
RANKS = "12345678"
ALL_UCIS = []
for f1 in FILES:
    for r1 in RANKS:
        for f2 in FILES:
            for r2 in RANKS:
                u = f1 + r1 + f2 + r2
                ALL_UCIS.append(u)
                if r2 in ("8", "1"):
                    for p in ("q", "r", "b", "n"):
                        ALL_UCIS.append(u + p)
# dedupe promotions
ALL_UCIS = list(dict.fromkeys(ALL_UCIS))
UCI_TO_IDX = {u: i for i, u in enumerate(ALL_UCIS)}
NUM_ACTIONS = len(ALL_UCIS)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-3
EPOCHS = 50
GAMES_PER_EPOCH = 50
MCTS_SIMS = 100
MAX_MOVES = 200
TEMP_MOVES = 30

# logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# -------------------- Neural Network --------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChessNet(nn.Module):
    def __init__(self, in_channels=13, hidden_channels=64, num_res_blocks=4,
                 board_size=8, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.conv_init = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels) for _ in range(num_res_blocks)
        ])
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 2, 1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, num_actions)
        )
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
        value = self.value_head(x)
        return policy, value

# -------------------- MCTS --------------------
def state_to_tensor(board: chess.Board) -> torch.Tensor:
    arr = torch.zeros(13, 8, 8, dtype=torch.float32)
    for sq, piece in board.piece_map().items():
        pt = {chess.PAWN:0, chess.KNIGHT:1, chess.BISHOP:2,
              chess.ROOK:3, chess.QUEEN:4, chess.KING:5}[piece.piece_type]
        offset = 0 if piece.color == chess.WHITE else 6
        row, col = divmod(sq, 8)
        arr[pt + offset, row, col] = 1.0
    if board.turn == chess.WHITE:
        arr[12].fill_(1.0)
    return arr

class MCTSNode:
    def __init__(self, board, parent=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior
    def expand(self, priors):
        for mv, p in priors:
            if mv not in self.children:
                nb = self.board.copy()
                nb.push(mv)
                self.children[mv] = MCTSNode(nb, self, p)
    def is_leaf(self):
        return not self.children

def uct(node: MCTSNode, child: MCTSNode, c_puct: float):
    return child.Q + c_puct * child.P * math.sqrt(node.N) / (1 + child.N)

def _get_path(node: MCTSNode):
    path = []
    while node:
        path.append(node)
        node = node.parent
    return path

class MCTS:
    def __init__(self, net: ChessNet, sims=MCTS_SIMS, c_puct=1.0,
                 time_limit=None, device=DEVICE,
                 epsilon=0.25, dir_alpha=0.3):
        self.net = net.to(device)
        self.sims = sims
        self.c_puct = c_puct
        self.time_limit = time_limit
        self.device = device
        self.epsilon = epsilon
        self.dir_alpha = dir_alpha
        self._infer_cache = {}

    def search(self, root_board: chess.Board) -> MCTSNode:
        self._infer_cache.clear()
        root = MCTSNode(root_board)

        policy, _ = self._infer(root_board)
        legal = list(root_board.legal_moves)
        probs = [policy[UCI_TO_IDX[mv.uci()]] for mv in legal]
        total = sum(probs) or len(probs)
        priors = [(mv, p/total) for mv, p in zip(legal, probs)]

        noise = np.random.dirichlet([self.dir_alpha] * len(priors))
        priors = [
            (mv, (1 - self.epsilon) * p + self.epsilon * n)
            for (mv, p), n in zip(priors, noise)
        ]
        root.expand(priors)

        start = time.time()
        for _ in range(self.sims):
            if self.time_limit and (time.time() - start) > self.time_limit:
                break
            node = root
            while not node.is_leaf():
                mv, node = max(node.children.items(),
                               key=lambda kv: uct(node, kv[1], self.c_puct))
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
            if outcome.winner is None:
                return 0.0
            return 1.0 if outcome.winner else -1.0
        if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
            return 0.0

        policy, value = self._infer(board)
        legal = list(board.legal_moves)
        probs = [policy[UCI_TO_IDX[mv.uci()]] for mv in legal]
        total = sum(probs) or len(probs)
        priors = [(mv, p/total) for mv, p in zip(legal, probs)]
        node.expand(priors)
        return value

    def _infer(self, board: chess.Board):
        fen = board.fen()
        if fen in self._infer_cache:
            return self._infer_cache[fen]
        t = state_to_tensor(board).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if self.device.startswith('cuda'):
                with amp.autocast():
                    logits, v = self.net(t)
            else:
                logits, v = self.net(t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        val = v.cpu().item()
        self._infer_cache[fen] = (probs, val)
        return probs, val

# -------------------- Training Helpers --------------------
def tensor_from_pi(pi_dict):
    pi = torch.zeros(len(ALL_UCIS), dtype=torch.float32)
    for uci, p in pi_dict.items():
        pi[UCI_TO_IDX[uci]] = p
    return pi

def train_on_batch(model, optimizer, batch):
    states = torch.stack([state_to_tensor(chess.Board(fen)) for fen, _, _ in batch]).to(DEVICE)
    pis = torch.stack([tensor_from_pi(pi) for _, pi, _ in batch]).to(DEVICE)
    zs = torch.tensor([z for *_, z in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    model.train()
    optimizer.zero_grad()
    logits, values = model(states)
    loss = nn.MSELoss()(values, zs) + nn.CrossEntropyLoss()(logits, pis.argmax(dim=1))
    loss.backward()
    optimizer.step()
    return loss.item()

# -------------------- Self-Play Training Loop --------------------
def selfplay_train_loop():
    model = ChessNet().to(DEVICE)
    if os.path.exists('best.pth'):
        model.load_state_dict(torch.load('best.pth', map_location=DEVICE))
        logger.info("Loaded initial model from best.pth")

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{EPOCHS} start")
        for game_num in range(1, GAMES_PER_EPOCH + 1):
            board = chess.Board()
            mcts = MCTS(model, sims=MCTS_SIMS, device=DEVICE)
            examples = []
            move_count = 0

            while True:
                if board.is_game_over(claim_draw=True):
                    break
                if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
                    break
                if move_count >= MAX_MOVES:
                    break

                root = mcts.search(board)
                counts = {mv.uci(): nd.N for mv, nd in root.children.items()}
                total = sum(counts.values())
                pi = {u: n/total for u, n in counts.items()}
                examples.append((board.fen(), pi, None))

                if move_count < TEMP_MOVES:
                    moves, weights = zip(*[(mv, nd.N) for mv, nd in root.children.items()])
                    best_move = random.choices(moves, weights=weights)[0]
                else:
                    best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]

                board.push(best_move)
                move_count += 1

            outcome = board.outcome(claim_draw=True)
            # assign z
            if outcome.winner is None:
                z = 0.0
            else:
                z = 1.0 if outcome.winner else -1.0
            # determine ending reason
            if board.is_checkmate():
                end = 'checkmate'
            elif board.is_stalemate():
                end = 'stalemate'
            elif board.is_insufficient_material():
                end = 'insufficient_material'
            elif board.can_claim_fifty_moves():
                end = '50_move'
            elif board.can_claim_threefold_repetition():
                end = 'repetition'
            else:
                end = 'draw'

            game_data = [(fen, pi, z) for fen, pi, _ in examples]

            loss = train_on_batch(model, optimizer, game_data)
            logger.info(f"Game {game_num}/{GAMES_PER_EPOCH}: result={board.result()} ({end}), "
                        f"{len(game_data)} examples, loss={loss:.4f}")
            torch.save(model.state_dict(), 'best.pth')
            logger.info("  Saved updated best.pth")

        logger.info(f"Finished epoch {epoch}/{EPOCHS}")
    logger.info("Training complete.")

if __name__ == '__main__':
    selfplay_train_loop()