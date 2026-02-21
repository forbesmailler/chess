"""Read book.bin and generate opening_tree.md showing first 2 plies."""

import struct
from pathlib import Path

import chess
import chess.polyglot

BOOK_PATH = Path(__file__).resolve().parent.parent / "book.bin"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "opening_tree.md"


def read_book(path):
    """Parse binary book into dict: zobrist_hash -> [(from_sq, to_sq, promo, weight)]."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"BOOK", f"Bad magic: {magic}"
        version, num_positions, num_moves = struct.unpack("<III", f.read(12))
        assert version == 1

        positions = []
        for _ in range(num_positions):
            h, offset, count = struct.unpack("<QIHxx", f.read(16))
            positions.append((h, offset, count))

        moves = []
        for _ in range(num_moves):
            fs, ts, pr, wt = struct.unpack("<BBBB", f.read(4))
            moves.append((fs, ts, pr, wt))

    book = {}
    for h, offset, count in positions:
        book[h] = moves[offset : offset + count]
    return book


def find_move(board, from_sq, to_sq, promo):
    """Match book move against legal moves (handles castling variants)."""
    for move in board.legal_moves:
        if move.from_square != from_sq or move.to_square != to_sq:
            continue
        if promo == 0 and move.promotion is None:
            return move
        if promo != 0 and move.promotion is not None:
            promo_map = {
                2: chess.KNIGHT,
                3: chess.BISHOP,
                4: chess.ROOK,
                5: chess.QUEEN,
            }
            if move.promotion == promo_map.get(promo):
                return move
    return None


def main():
    book = read_book(BOOK_PATH)
    board = chess.Board()
    root_hash = chess.polyglot.zobrist_hash(board)

    root_moves = book.get(root_hash, [])
    if not root_moves:
        print("No moves found for starting position!")
        return

    total_weight_ply1 = sum(w for _, _, _, w in root_moves)
    ply1_entries = sorted(root_moves, key=lambda x: x[3], reverse=True)

    lines = ["# Opening Book -- First 2 Plies", ""]

    for from_sq, to_sq, promo, weight in ply1_entries:
        move = find_move(board, from_sq, to_sq, promo)
        if move is None:
            continue
        san = board.san(move)
        pct = weight / total_weight_ply1 * 100

        lines.append(f"## 1. {san} (weight {weight}, {pct:.1f}%)")
        lines.append("")

        board.push(move)
        ply2_hash = chess.polyglot.zobrist_hash(board)
        ply2_moves = book.get(ply2_hash, [])

        if ply2_moves:
            total_weight_ply2 = sum(w for _, _, _, w in ply2_moves)
            ply2_sorted = sorted(ply2_moves, key=lambda x: x[3], reverse=True)

            lines.append("| Response | Weight | Probability |")
            lines.append("|----------|--------|-------------|")

            for fs2, ts2, pr2, wt2 in ply2_sorted:
                m2 = find_move(board, fs2, ts2, pr2)
                if m2 is None:
                    continue
                san2 = board.san(m2)
                pct2 = wt2 / total_weight_ply2 * 100
                lines.append(f"| 1...{san2} | {wt2} | {pct2:.1f}% |")

            lines.append("")
        else:
            lines.append("*No responses in book.*")
            lines.append("")

        board.pop()

    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
