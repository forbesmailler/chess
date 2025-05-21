# --- lichess.py ---
import os
import logging
import math
import torch
import chess
import berserk
from berserk.exceptions import ResponseError, ApiError
from train_bot import (
    ChessNet,
    MCTS,
    state_to_tensor,
    DEVICE,
    LR,
    MCTS_SIMS,
    train_on_batch
)

# ------------------------- Setup -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = ChessNet().to(DEVICE)
if os.path.exists('best.pth'):
    model.load_state_dict(torch.load('best.pth', map_location=DEVICE))
    logger.info("Loaded existing best.pth")
model.eval()

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Lichess client
with open('token.txt', 'r') as f:
    token = f.read().strip()
session = berserk.TokenSession(token)
client = berserk.Client(session=session)
MY_ID = client.account.get()['id']


def _try_make_move(game_id: str, uci: str) -> bool:
    try:
        client.bots.make_move(game_id, uci)
        return True
    except (ResponseError, ApiError) as e:
        logger.warning(f"Game {game_id}: could not play move {uci}: {e}")
    except Exception as e:
        logger.error(f"Game {game_id}: unexpected error on move {uci}: {e}")
    return False


def handle_game(game_id: str):
    stream = client.bots.stream_game_state(game_id)
    first = next(stream, None)
    if not first or first.get('type') != 'gameFull':
        logger.error(f"No gameFull for {game_id}")
        return

    our_white = (first.get('white', {}).get('id') == MY_ID)
    logger.info(f"Game {game_id}: we are {'White' if our_white else 'Black'}")

    # Initialize board and MCTS root
    moves = first.get('state', {}).get('moves', '').split()
    board = chess.Board()
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    ply_count = len(moves)
    root = None
    examples = []

    # If it's our turn before any move
    if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
        feat = state_to_tensor(board).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            raw_val = model(feat).cpu().item()
        adj_val = raw_val if board.turn == chess.WHITE else -raw_val
        logger.info(f"Eval after ply {ply_count} (white-persp): {adj_val:.4f}")

        sims = max(50, int(MCTS_SIMS * (1 - ply_count / 200)))
        mcts = MCTS(model, sims=sims, c_puct=math.sqrt(2), device=DEVICE)
        root = mcts.search(board)

        examples.append(board.fen())
        best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
        _try_make_move(game_id, best_move.uci())
        board.push(best_move)
        ply_count += 1
        root = root.children.get(best_move)
        if root:
            root.parent = None

    # stream events
    result = None
    for event in stream:
        if event.get('type') != 'gameState':
            continue
        if event.get('status') != 'started':
            result = event.get('winner')
            break

        new_moves = event.get('moves', '').split()
        for uci in new_moves[ply_count:]:
            actor = "Bot" if ((ply_count % 2 == 0 and our_white) or (ply_count % 2 == 1 and not our_white)) else "Opponent"
            logger.info(f"Game {game_id}: {actor} played move {uci}")
            board.push(chess.Move.from_uci(uci))
            if root and chess.Move.from_uci(uci) in root.children:
                root = root.children[chess.Move.from_uci(uci)]
                root.parent = None
            else:
                root = None
            ply_count += 1

        feat = state_to_tensor(board).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            raw_val = model(feat).cpu().item()
        adj_val = raw_val if board.turn == chess.WHITE else -raw_val
        logger.info(f"Eval after ply {ply_count} (white-persp): {adj_val:.4f}")

        if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
            sims = max(50, int(MCTS_SIMS * (1 - ply_count / 200)))
            mcts = MCTS(model, sims=sims, c_puct=math.sqrt(2), device=DEVICE)
            if root:
                root = mcts.search(board, root)
            else:
                root = mcts.search(board)
            examples.append(board.fen())

            best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            if not _try_make_move(game_id, best_move.uci()):
                break
            board.push(best_move)
            ply_count += 1
            root = root.children.get(best_move)
            if root:
                root.parent = None

    # Training
    base_z = 1.0 if result == 'white' else -1.0 if result == 'black' else 0.0
    batch = [(fen, base_z * ((-1) ** i)) for i, fen in enumerate(examples)]
    if batch:
        loss = train_on_batch(model, optimizer, batch)
        logger.info(f"Training on {len(batch)} examples: loss={loss:.4f}")

        # Calibrate final bias so initial position evaluates to zero
        # Prepare initial position
        init_board = chess.Board()
        feat0 = state_to_tensor(init_board).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            raw0 = model(feat0).cpu().item()
        pre_act = torch.atanh(torch.tensor(raw0, device=DEVICE))
        model.fc3.bias.data -= pre_act
        torch.save(model.state_dict(), 'best.pth')
        logger.info("Saved best.pth")


def main():
    for ev in client.bots.stream_incoming_events():
        if ev['type'] == 'challenge':
            client.bots.accept_challenge(ev['challenge']['id'])
        elif ev['type'] == 'gameStart':
            handle_game(ev['game']['id'])


if __name__ == '__main__':
    main()

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
    def __init__(self, net: ChessNet, sims=MCTS_SIMS, c_puct=math.sqrt(2), device=DEVICE):
        self.net = net.to(device)
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
            return self.net(feat).cpu().item()

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
        history = []
        ply = 0
        while not board.is_game_over(claim_draw=True):
            feat = state_to_tensor(board).to(DEVICE).unsqueeze(0)
            with torch.no_grad():
                raw_val = model(feat).cpu().item()
            adj_val = raw_val if board.turn == chess.WHITE else -raw_val
            logger.info(f"Self-play eval after move {ply+1}: {adj_val:.4f}")

            sims = max(50, int(MCTS_SIMS * (1 - ply / 200)))
            mcts = MCTS(model, sims=sims, c_puct=math.sqrt(2), device=DEVICE)
            root = mcts.search(board)
            sorted_children = sorted(root.children.items(),
                                     key=lambda kv: kv[1].N,
                                     reverse=True)
            best_move = sorted_children[0][0]
            second_move = sorted_children[1][0] if len(sorted_children) > 1 else best_move
            move = second_move if random.random() < 0.40 else best_move
            if move == second_move:
                logger.debug(f"Played second-best move: {move}")

            history.append(board.fen())
            board.push(move)
            ply += 1

        outcome = board.outcome(claim_draw=True)
        term = outcome.termination.name
        if outcome.termination == chess.Termination.CHECKMATE:
            base_z = 1.0 if outcome.winner else -1.0
        else:
            base_z = 0.0
        result_str = "win" if base_z == 1.0 else "loss" if base_z == -1.0 else "draw"
        logger.info(f"Game ended in {result_str} (base_z={base_z}), termination reason: {term}")

        batch = [(fen, base_z * ((-1) ** i)) for i, fen in enumerate(history)]
        loss = train_on_batch(model, optimizer, batch)
        logger.info(f"Trained on {len(batch)} positions, loss={loss:.4f}")


        # Calibrate final bias
        init_board = chess.Board()
        feat0 = state_to_tensor(init_board).to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            raw0 = model(feat0).cpu().item()
        pre_act = torch.atanh(torch.tensor(raw0, device=DEVICE))
        model.fc3.bias.data -= pre_act
        torch.save(model.state_dict(), 'best.pth')
        logger.info("Saved updated best.pth")

if __name__ == '__main__':
    selfplay_train_loop()