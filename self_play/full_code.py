# lichess.py
# A Lichess bot that plays, learns from its games, and saves its model after each game.

import os
import logging
import torch
import berserk
import chess
from berserk.exceptions import ResponseError, ApiError
from train_bot import (
    ChessNet,
    MCTS,
    state_to_tensor,
    UCI_TO_IDX,
    ALL_UCIS,
    LR,
    MCTS_SIMS,
    DEVICE
)

# ------------------------- Setup -------------------------
logging.basicConfig(level=logging.INFO)

# load or initialize model
model = ChessNet().to(DEVICE)
if os.path.exists('best.pth'):
    model.load_state_dict(torch.load('best.pth', map_location=DEVICE))
else:
    logging.info("No existing best.pth found. Starting from scratch.")
model.eval()

# optimizer for online training
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Lichess client
with open('token.txt', 'r') as f:
    token = f.read().strip()
session = berserk.TokenSession(token)
client = berserk.Client(session=session)
MY_ID = client.account.get()['id']

# ------------------------- Training utils -------------------------
def tensor_from_pi(pi_dict):
    pi = torch.zeros(len(ALL_UCIS), dtype=torch.float32)
    for uci, p in pi_dict.items():
        pi[UCI_TO_IDX[uci]] = p
    return pi


def train_on_batch(batch):
    st_fens, pis, zs = zip(*batch)
    states = torch.stack([state_to_tensor(chess.Board(fen)) for fen in st_fens]).to(DEVICE)
    target_pis = torch.stack([tensor_from_pi(pi) for pi in pis]).to(DEVICE)
    target_vals = torch.tensor(zs, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    model.train()
    optimizer.zero_grad()
    logits, values = model(states)
    loss = (
        torch.nn.functional.mse_loss(values, target_vals)
        + torch.nn.functional.cross_entropy(logits, target_pis.argmax(dim=1))
    )
    loss.backward()
    optimizer.step()
    model.eval()
    return loss.item()

# ------------------------- Helper functions -------------------------
def _extract_player_id(p):
    if not isinstance(p, dict):
        return None
    user = p.get('user')
    if isinstance(user, dict) and 'id' in user:
        return user['id']
    if 'id' in p:
        return p['id']
    return p.get('name')


def _try_make_move(game_id: str, uci: str) -> bool:
    try:
        client.bots.make_move(game_id, uci)
        logging.info(f"Game {game_id}: played move {uci}")
        return True
    except (ResponseError, ApiError) as e:
        logging.warning(f"Game {game_id}: could not play move {uci}: {e}")
    except Exception as e:
        logging.error(f"Game {game_id}: unexpected error on move {uci}: {e}")
    return False

# ------------------------- Game handler -------------------------
def handle_game(game_id: str):
    stream = client.bots.stream_game_state(game_id)
    try:
        first = next(stream)
    except StopIteration:
        logging.error(f"No events for game {game_id}")
        return

    if first.get('type') != 'gameFull':
        logging.error(f"Expected gameFull, got {first.get('type')}")
        return

    # determine players
    if 'players' in first:
        raw_w = first['players'].get('white')
        raw_b = first['players'].get('black')
    else:
        raw_w = first.get('white')
        raw_b = first.get('black')

    white_id = _extract_player_id(raw_w)
    black_id = _extract_player_id(raw_b)
    our_white = (white_id == MY_ID)

    logging.info(f"Game {game_id}: we are {'White' if our_white else 'Black'}")

    # collect self-play examples
    game_examples = []  # (fen, pi_dict, z)
    board = chess.Board()

    # apply any initial moves
    init_moves = first.get('state', {}).get('moves', '')
    for uci in init_moves.split():
        board.push(chess.Move.from_uci(uci))

    # if it's our turn from the start, play and record
    if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
        root = MCTS(model, sims=MCTS_SIMS, device=DEVICE).search(board)
        counts = {mv.uci(): nd.N for mv, nd in root.children.items()}
        total = sum(counts.values())
        pi = {u: n/total for u, n in counts.items()}
        game_examples.append((board.fen(), pi, None))
        best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        _try_make_move(game_id, best_move.uci())
        board.push(best_move)

    # play until end
    result = None
    for event in stream:
        if event.get('type') != 'gameState':
            continue
        if event.get('status') != 'started':
            result = event.get('winner')
            break

        moves = event.get('moves', '')
        board = chess.Board()
        for uci in moves.split():
            board.push(chess.Move.from_uci(uci))

        if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
            root = MCTS(model, sims=MCTS_SIMS, device=DEVICE).search(board)
            counts = {mv.uci(): nd.N for mv, nd in root.children.items()}
            total = sum(counts.values())
            pi = {u: n/total for u, n in counts.items()}
            game_examples.append((board.fen(), pi, None))
            best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            if not _try_make_move(game_id, best_move.uci()):
                break
            board.push(best_move)

    # determine outcome z
    z = 0.0
    if result == 'white':
        z = 1.0
    elif result == 'black':
        z = -1.0

    # train on all moves from this game
    training_batch = [(fen, pi, z) for fen, pi, _ in game_examples]
    if training_batch:
        loss = train_on_batch(training_batch)
        logging.info(f"Training on {len(training_batch)} examples: loss={loss:.4f}")
        torch.save(model.state_dict(), 'best.pth')
        logging.info("Saved best.pth")

# ------------------------- Main loop -------------------------
def main():
    for ev in client.bots.stream_incoming_events():
        if ev['type'] == 'challenge':
            client.bots.accept_challenge(ev['challenge']['id'])
        elif ev['type'] == 'gameStart':
            handle_game(ev['game']['id'])

if __name__ == '__main__':
    main()

# train_bot.py

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
LR = 0.1
EPOCHS = 50
GAMES_PER_EPOCH = 50
MCTS_SIMS = 200
TEMP_MOVES = 20

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
        # full game-over with draw claims
        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            term = outcome.termination
            if term == chess.Termination.CHECKMATE:
                return 1.0 if outcome.winner else -1.0
            return -0.5
        # pre-claim draws
        if board.can_claim_threefold_repetition() or board.can_claim_fifty_moves():
            return -0.5

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
            term = outcome.termination
            # assign z with draw penalties
            if term == chess.Termination.CHECKMATE:
                z = 1.0 if outcome.winner else -1.0
            else:
                z = -0.5
            # determine end reason string
            if term == chess.Termination.CHECKMATE:
                end = 'checkmate'
            elif term == chess.Termination.STALEMATE:
                end = 'stalemate'
            elif term == chess.Termination.INSUFFICIENT_MATERIAL:
                end = 'insufficient_material'
            elif term == chess.Termination.FIFTY_MOVES:
                end = '50_move'
            elif term == chess.Termination.THREEFOLD_REPETITION:
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