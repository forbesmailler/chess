import chess
import numpy as np

PIECE_TO_INDEX = {
    (chess.WHITE, chess.PAWN):   0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK):   3,
    (chess.WHITE, chess.QUEEN):  4,
    (chess.WHITE, chess.KING):   5,
    (chess.BLACK, chess.PAWN):   6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK):   9,
    (chess.BLACK, chess.QUEEN):  10,
    (chess.BLACK, chess.KING):   11
}

def fen_to_binary_features(fen: str) -> np.ndarray:
    """
    Convert a FEN string into a 780-dimensional feature vector
    FROM THE PERSPECTIVE OF THE SIDE TO MOVE.
      - If it's White to move, no change.
      - If it's Black to move, mirror the board so that 
        the position is treated as White to move.
      - Then produce:
         768 features for piece placement (64 squares × 12),
         4 for castling rights,
         8 for en passant rights.
    """
    board = chess.Board(fen=fen)
    side_to_move = board.turn
    if not side_to_move:  # Black to move
        board = board.mirror()

    # (1) Piece placement (768 = 64 squares × 12 piece types)
    piece_placement = np.zeros((64, 12), dtype=int)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            idx = PIECE_TO_INDEX[(piece.color, piece.piece_type)]
            piece_placement[square, idx] = 1
    piece_placement_flat = piece_placement.flatten()

    # (2) Castling rights (4 bits: WhiteK, WhiteQ, BlackK, BlackQ)
    castling_rights = np.zeros(4, dtype=int)
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_rights[0] = 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_rights[1] = 1
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_rights[2] = 1
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_rights[3] = 1

    # (3) En passant (8 bits, one per file 'a'..'h')
    en_passant = np.zeros(8, dtype=int)
    ep_square = board.ep_square  # None or [0..63]
    if ep_square is not None:
        file_index = chess.square_file(ep_square)
        en_passant[file_index] = 1

    # Combine into a single 1D array
    features = np.concatenate([piece_placement_flat, castling_rights, en_passant])
    return features