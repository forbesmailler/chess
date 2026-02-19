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
import multiprocessing as mp
import os
import queue
import struct
import sys
import threading
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import chess
import chess.polyglot
from tqdm import tqdm

_CHUNK_SIZE = 1 << 24  # 16 MB
_GAME_SEP = b"\n[Event "
_GAME_PREFIX = b"[Event "
_SENTINEL = None


def open_pgn(path: Path):
    """Open a PGN file as binary reader, handling .zst compression."""
    if path.suffix == ".zst" or path.name.endswith(".pgn.zst"):
        import zstandard

        fh = open(path, "rb")
        dctx = zstandard.ZstdDecompressor()
        return dctx.stream_reader(fh, read_size=1 << 22)
    return open(path, "rb")


def _prefetch_chunks(pgn_path, q):
    """Producer thread: read raw byte chunks split at game boundaries."""
    reader = open_pgn(pgn_path)
    try:
        tail = b""
        while True:
            raw = reader.read(_CHUNK_SIZE)
            if not raw:
                break
            data = tail + raw.replace(b"\r", b"")
            last = data.rfind(_GAME_SEP)
            if last == -1:
                tail = data
                continue
            q.put(data[:last])
            tail = data[last + 1:]  # starts with "[Event ..."
        if tail.strip():
            q.put(tail)
    finally:
        reader.close()
    q.put(_SENTINEL)


def _split_chunk(chunk):
    """Split a byte chunk into individual game byte strings."""
    parts = chunk.split(_GAME_SEP)
    result = [parts[0]]
    for p in parts[1:]:
        result.append(_GAME_PREFIX + p)
    return result


def _seq_hash(prev: int, move_san: str) -> int:
    """Deterministic running hash for move sequences (cross-process safe)."""
    return zlib.crc32(move_san.encode(), prev & 0xFFFFFFFF)


def _parse_game_bytes(game_data):
    """Extract (white_elo, black_elo, movetext_str) from PGN game bytes."""
    idx = game_data.find(b"\n\n")
    if idx == -1:
        return 0, 0, ""
    headers = game_data[:idx]
    movetext = game_data[idx + 2 :].decode("utf-8", errors="replace")

    white_elo = black_elo = 0
    i = headers.find(b'[WhiteElo "')
    if i != -1:
        j = headers.index(b'"', i + 11)
        try:
            white_elo = int(headers[i + 11 : j])
        except ValueError:
            pass
    i = headers.find(b'[BlackElo "')
    if i != -1:
        j = headers.index(b'"', i + 11)
        try:
            black_elo = int(headers[i + 11 : j])
        except ValueError:
            pass
    return white_elo, black_elo, movetext


def _extract_san_moves(movetext, max_depth):
    """Extract SAN move tokens from movetext, skipping comments efficiently."""
    moves = []
    i = 0
    n = len(movetext)
    while i < n and len(moves) < max_depth:
        c = movetext[i]
        if c <= " ":  # whitespace
            i += 1
            continue
        if c == "{":  # skip comment
            j = movetext.find("}", i + 1)
            i = n if j == -1 else j + 1
            continue
        if c == "(":  # skip variation
            depth = 1
            i += 1
            while i < n and depth > 0:
                if movetext[i] == "(":
                    depth += 1
                elif movetext[i] == ")":
                    depth -= 1
                i += 1
            continue
        # read token
        j = i
        while j < n and movetext[j] > " " and movetext[j] not in "{}()":
            j += 1
        token = movetext[i:j]
        i = j
        if not token:
            continue
        fc = token[0]
        if fc.isdigit():
            if "-" in token or "/" in token:
                break  # result
            continue  # move number
        if fc == "*":
            break  # result
        if fc == "$":
            continue  # NAG
        moves.append(token)
    return moves


_P1_ENTRY = struct.Struct("<BII")  # depth_u8, hash_u32, count_u32


def _phase1_worker(args):
    """Process a raw byte chunk for depth analysis. Returns packed bytes."""
    chunk, max_depth, min_elo = args
    games = _split_chunk(chunk)
    depth_counters = [Counter() for _ in range(max_depth + 1)]
    count = 0
    seq_hash = _seq_hash

    for game_data in games:
        if not game_data.strip():
            continue
        white_elo, black_elo, movetext = _parse_game_bytes(game_data)
        if min_elo > 0 and (white_elo < min_elo or black_elo < min_elo):
            continue
        san_moves = _extract_san_moves(movetext, max_depth)
        if not san_moves:
            continue
        count += 1
        h = 0
        for d, san in enumerate(san_moves, 1):
            h = seq_hash(h, san)
            depth_counters[d][h] += 1

    pack = _P1_ENTRY.pack
    parts = []
    for d in range(1, max_depth + 1):
        for h, c in depth_counters[d].items():
            parts.append(pack(d, h, c))
    return b"".join(parts), count


_P2_ENTRY = struct.Struct("<QBBBBI")  # hash_u64, from, to, promo, pad, count_u32


def _phase2_worker(args):
    """Process a raw byte chunk for book building. Returns packed bytes."""
    chunk, depth, min_elo = args
    games = _split_chunk(chunk)
    move_counts: dict[tuple[int, int, int, int], int] = defaultdict(int)
    count = 0

    for game_data in games:
        if not game_data.strip():
            continue
        white_elo, black_elo, movetext = _parse_game_bytes(game_data)
        if min_elo > 0 and (white_elo < min_elo or black_elo < min_elo):
            continue
        san_moves = _extract_san_moves(movetext, depth)
        if not san_moves:
            continue

        board = chess.Board()
        try:
            for san in san_moves:
                h = chess.polyglot.zobrist_hash(board)
                move = board.parse_san(san)
                promo = move.promotion if move.promotion is not None else 0
                move_counts[(h, move.from_square, move.to_square, promo)] += 1
                board.push(move)
        except (chess.IllegalMoveError, chess.InvalidMoveError,
                chess.AmbiguousMoveError):
            pass
        count += 1

    pack = _P2_ENTRY.pack
    data = b"".join(pack(h, f, t, p, 0, c) for (h, f, t, p), c in move_counts.items())
    return data, count


def find_optimal_depth(
    pgn_path: Path,
    max_lines: int,
    coverage: float,
    max_depth: int,
    min_elo: int,
    workers: int,
) -> tuple[int, int]:
    """Find the max depth where top N lines cover >coverage of games.

    Returns (optimal_depth, total_games).
    """
    print(f"Phase 1: Analyzing opening depth... ({workers} workers)")

    depth_counters = [Counter() for _ in range(max_depth + 1)]
    total_games = 0

    q = queue.Queue(maxsize=workers * 2)
    producer = threading.Thread(
        target=_prefetch_chunks, args=(pgn_path, q), daemon=True
    )
    producer.start()

    def chunk_iter():
        while True:
            chunk = q.get()
            if chunk is _SENTINEL:
                break
            yield (chunk, max_depth, min_elo)

    with mp.Pool(workers) as pool:
        pbar = tqdm(desc="  Scanning", unit=" games", unit_scale=True)
        for data, count in pool.imap_unordered(_phase1_worker, chunk_iter()):
            total_games += count
            pbar.update(count)
            for d, h, c in _P1_ENTRY.iter_unpack(data):
                depth_counters[d][h] += c
        pbar.close()

    producer.join()

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
    workers: int,
) -> dict[tuple[int, int, int, int], int]:
    """Replay games to depth, collecting (hash, from, to, promo) → count."""
    print(f"Phase 2: Building book at depth {depth}... ({workers} workers)")

    move_counts: dict[tuple[int, int, int, int], int] = defaultdict(int)
    game_count = 0

    q = queue.Queue(maxsize=workers * 2)
    producer = threading.Thread(
        target=_prefetch_chunks, args=(pgn_path, q), daemon=True
    )
    producer.start()

    def chunk_iter():
        while True:
            chunk = q.get()
            if chunk is _SENTINEL:
                break
            yield (chunk, depth, min_elo)

    with mp.Pool(workers) as pool:
        pbar = tqdm(desc="  Building", unit=" games", unit_scale=True)
        for data, count in pool.imap_unordered(_phase2_worker, chunk_iter()):
            game_count += count
            pbar.update(count)
            for h, f, t, p, _, c in _P2_ENTRY.iter_unpack(data):
                move_counts[(h, f, t, p)] += c
        pbar.close()

    producer.join()

    print(
        f"  Processed {game_count:,} games, {len(move_counts):,} unique position-moves"
    )
    return dict(move_counts)


def write_book(
    move_counts: dict[tuple[int, int, int, int], int],
    output: Path,
) -> None:
    """Write the binary book file."""
    positions: dict[int, list[tuple[int, int, int, int]]] = defaultdict(list)
    for (h, from_sq, to_sq, promo), count in move_counts.items():
        positions[h].append((from_sq, to_sq, promo, count))

    sorted_hashes = sorted(positions.keys())

    pos_table = []
    moves_array = []

    for h in sorted_hashes:
        entries = positions[h]
        entries.sort(key=lambda x: x[3], reverse=True)

        max_count = max(c for _, _, _, c in entries)
        offset = len(moves_array)

        for from_sq, to_sq, promo, count in entries:
            weight = max(1, round(count / max_count * 255))
            moves_array.append((from_sq, to_sq, promo, weight))

        pos_table.append((h, offset, len(entries)))

    with open(output, "wb") as f:
        f.write(b"BOOK")
        f.write(struct.pack("<III", 1, len(pos_table), len(moves_array)))

        for h, offset, count in pos_table:
            f.write(struct.pack("<QIHxx", h, offset, count))

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
    parser.add_argument(
        "--max-depth", type=int, default=20, help="Max search depth for Phase 1"
    )
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
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes (default: cpu_count)",
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
            args.pgn, args.lines, args.coverage, args.max_depth, args.min_elo,
            args.workers,
        )
        if depth == 0:
            print("No suitable depth found.", file=sys.stderr)
            sys.exit(1)

    move_counts = build_book(args.pgn, depth, args.min_elo, args.workers)

    if args.min_count > 1:
        before = len(move_counts)
        move_counts = {k: v for k, v in move_counts.items() if v >= args.min_count}
        print(
            f"  Filtered {before:,} → {len(move_counts):,} (min_count={args.min_count})"
        )

    write_book(move_counts, args.output)


if __name__ == "__main__":
    main()
