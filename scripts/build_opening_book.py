"""Build a binary opening book from a PGN file.

Accepts .pgn or .pgn.zst files. Streams games without loading the full file.

Phase 1 — Depth analysis: finds maximum depth D where the top N opening
sequences cover >threshold of games. Uses running hashes to avoid storing
all game sequences in memory.

Phase 2 — Book building: replays all games to depth D, recording
(polyglot_hash, from_sq, to_sq, promotion) → count for each position-move.

Binary format (little-endian):
  Header (16 bytes): magic "BOOK", version u32, num_positions u32, num_moves u32
  Position table (sorted by hash, 16 bytes each):
    hash u64, moves_offset u32, num_moves u16, reserved u16
  Moves array (4 bytes each):
    from_sq u8, to_sq u8, promotion u8, weight u8
"""

import argparse
import io
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chess
import chess.pgn
import chess.polyglot
from tqdm import tqdm


def open_pgn(path: Path):
    """Open a PGN file, handling .zst compression."""
    if path.suffix == ".zst" or path.name.endswith(".pgn.zst"):
        import zstandard

        fh = open(path, "rb")
        dctx = zstandard.ZstdDecompressor()
        reader = dctx.stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def stream_games(pgn_handle, min_elo: int = 0):
    """Yield chess.pgn.Game objects, optionally filtering by minimum Elo."""
    while True:
        game = chess.pgn.read_game(pgn_handle)
        if game is None:
            break
        if min_elo > 0:
            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
                if white_elo < min_elo or black_elo < min_elo:
                    continue
            except ValueError:
                continue
        yield game


def extract_moves(game, max_depth: int) -> list[chess.Move]:
    """Extract the first max_depth moves from a game."""
    moves = []
    node = game
    for _ in range(max_depth):
        node = node.next()
        if node is None:
            break
        moves.append(node.move)
    return moves


def find_optimal_depth(
    pgn_path: Path,
    max_lines: int,
    coverage: float,
    max_depth: int,
    min_elo: int,
) -> tuple[int, int]:
    """Find the max depth where top N lines cover >coverage of games.

    Uses running hashes instead of storing all sequences — O(unique_lines)
    memory per depth instead of O(total_games).

    Returns (optimal_depth, total_games).
    """
    print("Phase 1: Analyzing opening depth...")

    depth_counters = [Counter() for _ in range(max_depth + 1)]
    total_games = 0

    handle = open_pgn(pgn_path)
    try:
        for game in tqdm(
            stream_games(handle, min_elo),
            desc="  Scanning",
            unit=" games",
            unit_scale=True,
        ):
            moves = extract_moves(game, max_depth)
            if not moves:
                continue
            total_games += 1
            h = 0
            for d, move in enumerate(moves, 1):
                h = hash((h, move.uci()))
                depth_counters[d][h] += 1
    finally:
        handle.close()

    if total_games == 0:
        print("No games found!")
        return 0, 0

    print(f"  Total games: {total_games:,}")

    optimal_depth = 1
    for depth in range(1, max_depth + 1):
        counter = depth_counters[depth]
        if not counter:
            break
        games_at_depth = sum(counter.values())

        if games_at_depth < total_games * 0.5:
            break

        top_lines = counter.most_common(max_lines)
        top_coverage = sum(c for _, c in top_lines) / total_games

        print(
            f"  Depth {depth:2d}: {len(counter):>8,d} unique lines,"
            f" top {min(max_lines, len(counter)):>6,d} cover {top_coverage:.1%}"
            f" of {games_at_depth:,} games"
        )

        if top_coverage >= coverage:
            optimal_depth = depth
        else:
            break

    del depth_counters
    print(f"  Optimal depth: {optimal_depth}")
    return optimal_depth, total_games


def build_book(
    pgn_path: Path,
    depth: int,
    min_elo: int,
) -> dict[tuple[int, int, int, int], int]:
    """Replay games to depth, collecting (hash, from, to, promo) → count."""
    print(f"Phase 2: Building book at depth {depth}...")

    move_counts: dict[tuple[int, int, int, int], int] = defaultdict(int)
    game_count = 0

    handle = open_pgn(pgn_path)
    try:
        for game in tqdm(
            stream_games(handle, min_elo),
            desc="  Building",
            unit=" games",
            unit_scale=True,
        ):
            board = game.board()
            node = game
            for ply in range(depth):
                node = node.next()
                if node is None:
                    break

                move = node.move
                h = chess.polyglot.zobrist_hash(board)
                from_sq = move.from_square
                to_sq = move.to_square
                promo = 0
                if move.promotion is not None:
                    promo = move.promotion  # chess.KNIGHT=2..QUEEN=5

                move_counts[(h, from_sq, to_sq, promo)] += 1
                board.push(move)

            game_count += 1
    finally:
        handle.close()

    print(
        f"  Processed {game_count:,} games, {len(move_counts):,} unique position-moves"
    )
    return dict(move_counts)


def write_book(
    move_counts: dict[tuple[int, int, int, int], int],
    output: Path,
) -> None:
    """Write the binary book file."""
    # Group moves by position hash
    positions: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for (h, from_sq, to_sq, promo), count in move_counts.items():
        positions[h].append((from_sq, to_sq, promo, count))

    # Sort positions by hash for binary search
    sorted_hashes = sorted(positions.keys())

    # Build moves array and position table
    pos_table = []
    moves_array = []

    for h in sorted_hashes:
        entries = positions[h]
        # Sort moves by count descending for better weight distribution
        entries.sort(key=lambda x: x[3], reverse=True)

        max_count = max(c for _, _, _, c in entries)
        offset = len(moves_array)

        for from_sq, to_sq, promo, count in entries:
            weight = max(1, round(count / max_count * 255))
            moves_array.append((from_sq, to_sq, promo, weight))

        pos_table.append((h, offset, len(entries)))

    # Write binary
    with open(output, "wb") as f:
        # Header
        f.write(b"BOOK")
        f.write(struct.pack("<III", 1, len(pos_table), len(moves_array)))

        # Position table
        for h, offset, count in pos_table:
            f.write(struct.pack("<QIHxx", h, offset, count))

        # Moves array
        for from_sq, to_sq, promo, weight in moves_array:
            f.write(struct.pack("<BBBB", from_sq, to_sq, promo, weight))

    size = output.stat().st_size
    unit = "MB" if size > 1024 * 1024 else "KB"
    val = size / (1024 * 1024) if size > 1024 * 1024 else size / 1024
    print(f"  Wrote {output}: {len(pos_table):,} positions, {len(moves_array):,} moves")
    print(f"  File size: {val:.1f} {unit}")


def main():
    parser = argparse.ArgumentParser(description="Build opening book from PGN")
    parser.add_argument("pgn", type=Path, help="PGN file (.pgn or .pgn.zst)")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("book.bin"), help="Output file"
    )
    parser.add_argument(
        "--lines", type=int, default=250000, help="Max opening lines to cover"
    )
    parser.add_argument(
        "--coverage", type=float, default=0.5, help="Min coverage threshold"
    )
    parser.add_argument("--max-depth", type=int, default=30, help="Max search depth")
    parser.add_argument("--min-elo", type=int, default=0, help="Min player Elo filter")
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="Skip depth analysis, use this depth directly",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum times a position-move must appear (filters noise)",
    )
    args = parser.parse_args()

    if not args.pgn.exists():
        print(f"Error: {args.pgn} not found", file=sys.stderr)
        sys.exit(1)

    if args.depth > 0:
        depth = args.depth
        print(f"Using fixed depth: {depth}")
    else:
        depth, _ = find_optimal_depth(
            args.pgn, args.lines, args.coverage, args.max_depth, args.min_elo
        )
        if depth == 0:
            print("No suitable depth found.", file=sys.stderr)
            sys.exit(1)

    move_counts = build_book(args.pgn, depth, args.min_elo)

    if args.min_count > 1:
        before = len(move_counts)
        move_counts = {k: v for k, v in move_counts.items() if v >= args.min_count}
        print(
            f"  Filtered {before:,} → {len(move_counts):,} (min_count={args.min_count})"
        )

    write_book(move_counts, args.output)


if __name__ == "__main__":
    main()
