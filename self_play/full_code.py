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
    MCTS_SIMS
)

# ------------------------- Setup -------------------------
logging.basicConfig(level=logging.INFO)
model = ChessNet().to(DEVICE)
if os.path.exists('best.pth'):
    model.load_state_dict(torch.load('best.pth', map_location=DEVICE))
    logging.info("Loaded existing best.pth")
model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Lichess client
with open('token.txt', 'r') as f:
    token = f.read().strip()
session = berserk.TokenSession(token)
client = berserk.Client(session=session)
MY_ID = client.account.get()['id']

# ------------------------- Helper functions -------------------------
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
    first = next(stream, None)
    if not first or first.get('type') != 'gameFull':
        logging.error(f"No gameFull for {game_id}")
        return

    raw_w = first['players'].get('white')
    raw_b = first['players'].get('black')
    our_white = (raw_w.get('user', {}).get('id') == MY_ID)
    logging.info(f"Game {game_id}: we are {'White' if our_white else 'Black'}")

    board = chess.Board()
    for uci in first.get('state', {}).get('moves', '').split():
        board.push(chess.Move.from_uci(uci))

    examples = []
    result = None
    for event in stream:
        if event.get('type') != 'gameState':
            continue
        if event.get('status') != 'started':
            result = event.get('winner')
            break
        moves = event.get('moves', '').split()
        board = chess.Board()
        for uci in moves:
            board.push(chess.Move.from_uci(uci))
        if (board.turn == chess.WHITE and our_white) or (board.turn == chess.BLACK and not our_white):
            root = MCTS(model, sims=MCTS_SIMS, c_puct=math.sqrt(2), device=DEVICE).search(board)
            counts = {mv.uci(): nd.N for mv, nd in root.children.items()}
            total = sum(counts.values())
            pi = {u: n/total for u, n in counts.items()}
            examples.append((board.fen(), None, result))  # z to be filled
            best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            if not _try_make_move(game_id, best_move.uci()):
                break
    # determine z
    z = 0.0
    if result == 'white':
        z = 1.0
    elif result == 'black':
        z = -1.0

    # train on this game's data
    # examples list of (fen, None, z)
    batch = [(fen, z) for fen, _, _ in examples]
    if batch:
        from train_bot import train_on_batch  # import here to avoid circular
        loss = train_on_batch(model, optimizer, batch)
        logging.info(f"Training on {len(batch)} examples: loss={loss:.4f}")
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


# --- train_bot.py ---
import os
import math
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import chess

# -------------------- Constants --------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 0.01  # updated learning rate
MCTS_SIMS = 200

# logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# -------------------- Feature extraction --------------------
def state_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Returns flattened 12x64 binary representation, flipped for active color.
    """
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
        # input: 12*64, hidden1=32, hidden2=16, output=1
        self.fc1 = nn.Linear(12 * 64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self._init_weights()

    def _init_weights(self):
        # small random weights
        for m in (self.fc1, self.fc2):
            m.weight.data.normal_(0.0, 1e-3)
            m.bias.data.zero_()
        # initialize fc1 node 0 and 1 with material sums
        values = torch.tensor([1.0, 3.0, 3.0, 5.0, 9.0, 0.0])
        # node0: current player material
        w0 = torch.zeros(12 * 64)
        for ch in range(6):
            w0[ch * 64:(ch + 1) * 64] = values[ch]
        self.fc1.weight.data[0] = w0
        self.fc1.bias.data[0] = 0.0
        # node1: opponent material
        w1 = torch.zeros(12 * 64)
        for ch in range(6, 12):
            w1[ch * 64:(ch + 1) * 64] = values[ch - 6]
        self.fc1.weight.data[1] = w1
        self.fc1.bias.data[1] = 0.0
        # fc2 and fc3 small random
        self.fc2.weight.data.normal_(0.0, 1e-3)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.normal_(0.0, 1e-3)
        self.fc3.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        out = self.fc3(h2)
        return out  # (...,1)

# -------------------- MCTS --------------------
class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = 1.0

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
            return 0.0  # draw
        node.expand()
        feat = state_to_tensor(board).to(self.device).unsqueeze(0)
        with torch.no_grad():
            val = self.net(feat).cpu().item()
        return val

# -------------------- Training Helpers --------------------
def train_on_batch(model, optimizer, batch):
    # batch: list of (fen, z)
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
            best_move = max(root.children.items(), key=lambda kv: kv[1].N)[0]
            history.append(board.fen())
            board.push(best_move)
        outcome = board.outcome(claim_draw=True)
        if outcome.termination == chess.Termination.CHECKMATE:
            z = 1.0 if outcome.winner else -1.0
        else:
            z = 0.0
        batch = [(fen, z) for fen in history]
        loss = train_on_batch(model, optimizer, batch)
        logger.info(f"Trained on {len(batch)} positions, loss={loss:.4f}")
        torch.save(model.state_dict(), 'best.pth')
        logger.info("Saved updated best.pth")

if __name__ == '__main__':
    selfplay_train_loop()