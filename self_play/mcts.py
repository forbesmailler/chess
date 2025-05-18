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
