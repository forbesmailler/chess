# constants.py
files  = "abcdefgh"
ranks  = "12345678"
ALL_UCIS = []
for f1 in files:
    for r1 in ranks:
        for f2 in files:
            for r2 in ranks:
                u = f1 + r1 + f2 + r2
                ALL_UCIS.append(u)
                # promotions on the last rank for white (8) or black (1)
                if r2 in ("8", "1"):
                    for p in ("q","r","b","n"):
                        ALL_UCIS.append(u + p)
# dedupe
ALL_UCIS = list(dict.fromkeys(ALL_UCIS))
UCI_TO_IDX = {u:i for i,u in enumerate(ALL_UCIS)}
NUM_ACTIONS = len(ALL_UCIS)

# mcts.py
import math
import numpy as np
import torch
import chess
from network import ChessNet
from typing import Dict, Tuple, List
from constants import ALL_UCIS, UCI_TO_IDX

def state_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    12 planes for {P,N,B,R,Q,K}×{white,black}, plus 1 plane for side-to-move.
    Output shape: (13,8,8)
    """
    arr = torch.zeros(13, 8, 8, dtype=torch.float32)
    piece_map = board.piece_map()
    for sq, piece in piece_map.items():
        plane = {
            chess.PAWN:   0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK:   3,
            chess.QUEEN:  4,
            chess.KING:   5
        }[piece.piece_type]
        color_offset = 0 if piece.color == chess.WHITE else 6
        row, col = divmod(sq, 8)
        arr[plane + color_offset, row, col] = 1.0
    # side to move
    if board.turn == chess.WHITE:
        arr[12].fill_(1.0)
    return arr

class MCTSNode:
    def __init__(self, board: chess.Board, parent=None, prior: float=0.0):
        self.board = board
        self.parent = parent
        self.children: Dict[chess.Move, MCTSNode] = {}
        self.N = 0        # visit count
        self.W = 0.0      # total value
        self.Q = 0.0      # mean value
        self.P = prior   # prior probability

    def expand(self, priors: List[Tuple[chess.Move,float]]):
        for move, p in priors:
            if move not in self.children:
                nb = self.board.copy()
                nb.push(move)
                self.children[move] = MCTSNode(nb, parent=self, prior=p)

    def is_leaf(self):
        return len(self.children) == 0

def uct(node: MCTSNode, child: MCTSNode, c_puct: float):
    return child.Q + c_puct * child.P * math.sqrt(node.N) / (1 + child.N)

class MCTS:
    def __init__(self,
                 net: ChessNet,
                 sims: int = 400,
                 c_puct: float = 1.0,
                 device: str = 'cpu'):
        self.net = net.to(device)
        self.sims = sims
        self.c_puct = c_puct
        self.device = device

    def search(self, root_board: chess.Board) -> MCTSNode:
        root = MCTSNode(root_board)
        # get root priors
        policy, _ = self._infer(root_board)
        legal = list(root_board.legal_moves)
        priors = []
        probs = []
        for mv in legal:
            idx = UCI_TO_IDX[mv.uci()]
            probs.append(policy[idx])
        total = sum(probs)
        for mv, p in zip(legal, probs):
            priors.append((mv, p/total if total>0 else 1/len(probs)))
        root.expand(priors)

        for _ in range(self.sims):
            node = root
            path = [node]
            # ➊ Selection
            while not node.is_leaf():
                mv, node = max(node.children.items(),
                               key=lambda itm: uct(node, itm[1], self.c_puct))
                path.append(node)
            # ➋ Evaluation & Expansion
            value = self._evaluate_and_expand(node)
            # ➌ Backpropagation
            for nd in reversed(path):
                nd.N += 1
                nd.W += value
                nd.Q = nd.W / nd.N
                value = -value
        return root

    def _evaluate_and_expand(self, node: MCTSNode) -> float:
        if node.board.is_game_over():
            result = node.board.result()
            if result == '1-0': return 1.0
            if result == '0-1': return -1.0
            return 0.0
        policy, value = self._infer(node.board)
        legal = list(node.board.legal_moves)
        priors = []
        probs = [policy[UCI_TO_IDX[mv.uci()]] for mv in legal]
        total = sum(probs)
        for mv, p in zip(legal, probs):
            priors.append((mv, p/total if total>0 else 1/len(probs)))
        node.expand(priors)
        return value

    def _infer(self, board: chess.Board) -> Tuple[np.ndarray,float]:
        t = state_to_tensor(board).to(self.device).unsqueeze(0)
        with torch.no_grad():
            logits, v = self.net(t)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            value = v.cpu().item()
        return probs, value
    
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

# selfplay_train.py
# Integrated self-play and online training with logging

import os
import random
import logging
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import chess
from network import ChessNet
from mcts import MCTS, state_to_tensor, UCI_TO_IDX, ALL_UCIS

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Hyperparameters
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
REPLAY_BUFFER_SIZE = 100_000
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 50
GAMES_PER_EPOCH = 50
MCTS_SIMS = 100


def tensor_from_pi(pi_dict):
    """
    Convert a dict of move->prob into a tensor of length len(ALL_UCIS).
    """
    pi = torch.zeros(len(ALL_UCIS), dtype=torch.float32)
    for uci, p in pi_dict.items():
        pi[UCI_TO_IDX[uci]] = p
    return pi


def train_on_batch(model, optimizer, batch):
    """
    Perform a single optimization step on the given batch.
    Batch is a list of (fen, pi_dict, z).
    """
    # Prepare tensors
    states = torch.stack([state_to_tensor(chess.Board(fen)) for fen, pi, z in batch]).to(DEVICE)
    pis = torch.stack([tensor_from_pi(pi) for fen, pi, z in batch]).to(DEVICE)
    zs = torch.tensor([z for fen, pi, z in batch], dtype=torch.float32, device=DEVICE).unsqueeze(1)

    model.train()
    optimizer.zero_grad()
    logits, values = model(states)
    loss_v = nn.MSELoss()(values, zs)
    loss_p = nn.CrossEntropyLoss()(logits, pis.argmax(dim=1))
    loss = loss_v + loss_p
    loss.backward()
    optimizer.step()
    return loss.item()


def selfplay_train_loop():
    """
    Main loop: for each epoch, generate games by self-play,
    add examples to replay buffer, and perform online training.
    """
    # Initialize model and optimizer
    model = ChessNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Replay buffer
    buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    for epoch in range(1, EPOCHS + 1):
        logger.info(f"Epoch {epoch}/{EPOCHS} start")
        for game_num in range(1, GAMES_PER_EPOCH + 1):
            # Self-play one game
            board = chess.Board()
            mcts = MCTS(model, sims=MCTS_SIMS, device=DEVICE)
            game_examples = []

            while not board.is_game_over():
                root = mcts.search(board)
                # Extract visit counts -> move probabilities
                counts = {mv.uci(): child.N for mv, child in root.children.items()}
                total = sum(counts.values())
                pi = {uci: n / total for uci, n in counts.items()}
                game_examples.append((board.fen(), pi, None))
                # Select best move
                best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
                board.push(best_move)

            # Determine game outcome
            result = board.result()  # '1-0', '0-1', or '1/2-1/2'
            z = 1.0 if result == '1-0' else (-1.0 if result == '0-1' else 0.0)
            game_data = [(fen, pi, z) for fen, pi, _ in game_examples]

            # Add to replay buffer
            buffer.extend(game_data)
            logger.info(
                f"Game {game_num}/{GAMES_PER_EPOCH}: generated {len(game_data)} examples, "
                f"buffer size={len(buffer)}"
            )

            # Online training step if enough data
            if len(buffer) >= BATCH_SIZE:
                batch = random.sample(buffer, BATCH_SIZE)
                loss = train_on_batch(model, optimizer, batch)
                logger.info(f"  Training step loss={loss:.4f}")

        # Save checkpoint at end of epoch
        ckpt_path = f'checkpoint_epoch{epoch}.pth'
        torch.save(model.state_dict(), ckpt_path)
        logger.info(f"Saved checkpoint: {ckpt_path}")

    # Final save
    best_path = 'best.pth'
    torch.save(model.state_dict(), best_path)
    logger.info(f"Training complete. Model saved to {best_path}")


if __name__ == '__main__':
    selfplay_train_loop()
