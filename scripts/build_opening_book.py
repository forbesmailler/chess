"""Build a binary opening book from a PGN file.

Accepts .pgn or .pgn.zst files. For .zst, streams decompression in ~256 MB
chunks to avoid needing disk space for the full decompressed file.

Depth-agnostic: follows the game tree as deep as games go, keeping only moves
that are popular (>min_weight_pct at their position) and well-supported
(at least min_count games). Bullet games are excluded by default.

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
import struct
import sys
import tempfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.polyglot
from tqdm import tqdm

_GAME_SEP = b"\n[Event "
_GAME_PREFIX = b"[Event "


def _iter_zst_chunks(pgn_path, chunk_size=256 << 20):
    """Stream decompress .zst with one-chunk-ahead prefetching.

    Decompresses the next chunk in a background thread while the current
    chunk is being processed, overlapping I/O with worker computation.
    Peak temp disk usage: ~2 × chunk_size.
    """
    import zstandard

    fh = open(pgn_path, "rb")
    dctx = zstandard.ZstdDecompressor()
    reader = dctx.stream_reader(fh, read_size=1 << 22)
    leftover = [b""]
    live_paths = []

    def _decompress_next():
        parts = [leftover[0]] if leftover[0] else []
        total = len(leftover[0])
        eof = False
        while total < chunk_size:
            data = reader.read(1 << 22)
            if not data:
                eof = True
                break
            clean = data.replace(b"\r", b"")
            parts.append(clean)
            total += len(clean)
        if total == 0:
            leftover[0] = b""
            return None
        chunk = b"".join(parts)
        if not eof:
            boundary = chunk.rfind(_GAME_SEP)
            if boundary > 0:
                leftover[0] = chunk[boundary + 1 :]
                chunk = chunk[: boundary + 1]
            else:
                leftover[0] = b""
        else:
            leftover[0] = b""
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pgn", mode="wb")
        tmp.write(chunk)
        tmp.close()
        live_paths.append(tmp.name)
        return tmp.name

    try:
        with ThreadPoolExecutor(max_workers=1) as tex:
            future = tex.submit(_decompress_next)
            while True:
                path = future.result()
                if path is None:
                    break
                # Start next decompression while caller processes current
                future = tex.submit(_decompress_next)
                yield path
                try:
                    os.unlink(path)
                except OSError:
                    pass
    finally:
        reader.close()
        fh.close()
        for p in live_paths:
            try:
                os.unlink(p)
            except OSError:
                pass


def _iter_files(pgn_path):
    """Yield file paths for processing. .zst streams chunks; .pgn yields once."""
    if str(pgn_path).endswith(".pgn.zst") or Path(pgn_path).suffix == ".zst":
        yield from _iter_zst_chunks(pgn_path)
    else:
        yield str(pgn_path)


def _find_boundaries(filepath, num_chunks):
    """Find game boundaries for parallel processing."""
    size = os.path.getsize(filepath)
    if size == 0:
        return []

    offsets = [0]
    with open(filepath, "rb") as f:
        for i in range(1, num_chunks):
            target = size * i // num_chunks
            f.seek(target)
            buf = f.read(1 << 16)  # 64 KB search window
            idx = buf.find(_GAME_SEP)
            if idx != -1:
                offsets.append(target + idx + 1)  # start of "[Event ..."
    offsets.append(size)

    offsets = sorted(set(offsets))
    return [
        (offsets[i], offsets[i + 1])
        for i in range(len(offsets) - 1)
        if offsets[i] < offsets[i + 1]
    ]


def _read_chunk(filepath, start, end):
    """Read a byte range from a file."""
    with open(filepath, "rb") as f:
        f.seek(start)
        return f.read(end - start)


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


def _check_elo(game_data, min_elo):
    """Fast ELO check on raw bytes. Returns False if either player < min_elo."""
    i = game_data.find(b'[WhiteElo "')
    if i == -1:
        return False
    j = game_data.find(b'"', i + 11)
    if j == -1:
        return False
    try:
        if int(game_data[i + 11 : j]) < min_elo:
            return False
    except ValueError:
        return False
    i = game_data.find(b'[BlackElo "')
    if i == -1:
        return False
    j = game_data.find(b'"', i + 11)
    if j == -1:
        return False
    try:
        if int(game_data[i + 11 : j]) < min_elo:
            return False
    except ValueError:
        return False
    return True


def _check_time_control(game_data, min_seconds):
    """Fast time control check on raw bytes. Returns False if initial time < min_seconds."""
    i = game_data.find(b'[TimeControl "')
    if i == -1:
        return False
    j = game_data.find(b'"', i + 14)
    if j == -1:
        return False
    tc = game_data[i + 14 : j]
    plus = tc.find(b"+")
    try:
        base = int(tc[:plus]) if plus != -1 else int(tc)
    except ValueError:
        return False
    return base >= min_seconds


def _get_movetext(game_data):
    """Extract movetext string from game bytes (decode only the movetext)."""
    idx = game_data.find(b"\n\n")
    if idx == -1:
        return ""
    return game_data[idx + 2 :].decode("utf-8", errors="replace")


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
        if c <= " ":
            i += 1
            continue
        if c == "{":
            j = movetext.find("}", i + 1)
            i = n if j == -1 else j + 1
            continue
        if c == "(":
            depth = 1
            i += 1
            while i < n and depth > 0:
                if movetext[i] == "(":
                    depth += 1
                elif movetext[i] == ")":
                    depth -= 1
                i += 1
            continue
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
                break
            continue
        if fc == "*":
            break
        if fc == "$":
            continue
        moves.append(token)
    return moves


_LC_ENTRY = struct.Struct("<II")  # hash_u32, count_u32


def _count_lines_worker(args):
    """Count opening line frequencies and capture one example per hash."""
    filepath, start, end, min_elo, min_time = args
    chunk = _read_chunk(filepath, start, end).replace(b"\r", b"")
    games = _split_chunk(chunk)
    line_counts = Counter()
    sequences = {}
    matched = 0
    scanned = 0
    seq_hash = _seq_hash

    for game_data in games:
        if len(game_data) < 20:
            continue
        scanned += 1
        if min_elo > 0 and not _check_elo(game_data, min_elo):
            continue
        if min_time > 0 and not _check_time_control(game_data, min_time):
            continue
        movetext = _get_movetext(game_data)
        san_moves = _extract_san_moves(movetext, 500)
        if not san_moves:
            continue
        matched += 1
        h = 0
        for san in san_moves:
            h = seq_hash(h, san)
        line_counts[h] += 1
        if h not in sequences:
            sequences[h] = tuple(san_moves)

    pack = _LC_ENTRY.pack
    data = b"".join(pack(h, c) for h, c in line_counts.items())
    return data, matched, scanned, sequences


def _replay_worker(items):
    """Replay unique opening sequences to extract position-move data."""
    move_counts = defaultdict(int)
    dest_hashes = {}
    for h, weight, san_moves in items:
        board = chess.Board()
        try:
            for san in san_moves:
                pos_hash = chess.polyglot.zobrist_hash(board)
                move = board.parse_san(san)
                promo = move.promotion if move.promotion is not None else 0
                key = (pos_hash, move.from_square, move.to_square, promo)
                move_counts[key] += weight
                board.push(move)
                if key not in dest_hashes:
                    dest_hashes[key] = chess.polyglot.zobrist_hash(board)
        except (
            chess.IllegalMoveError,
            chess.InvalidMoveError,
            chess.AmbiguousMoveError,
        ):
            pass
    return dict(move_counts), dest_hashes


def build_book(
    file_iter,
    min_elo: int,
    min_time: int,
    workers: int,
) -> tuple[dict[tuple[int, int, int, int], int], dict[tuple[int, int, int, int], int]]:
    """Count unique move sequences in workers, then replay each once."""
    print(f"Building book... ({workers} workers)")

    line_counts = Counter()
    all_sequences: dict[int, tuple[str, ...]] = {}
    total_matched = 0
    total_scanned = 0

    with mp.Pool(workers) as pool:
        pbar = tqdm(desc="  Scanning", unit=" games", unit_scale=True)
        for filepath in file_iter:
            boundaries = _find_boundaries(filepath, workers * 10)
            if not boundaries:
                continue
            args_list = [
                (filepath, s, e, min_elo, min_time) for s, e in boundaries
            ]
            for data, matched, scanned, sequences in pool.imap_unordered(
                _count_lines_worker, args_list
            ):
                total_matched += matched
                total_scanned += scanned
                pbar.update(scanned)
                for h, c in _LC_ENTRY.iter_unpack(data):
                    line_counts[h] += c
                for h, seq in sequences.items():
                    if h not in all_sequences:
                        all_sequences[h] = seq
        pbar.close()

    print(
        f"  {total_scanned:,} games scanned,"
        f" {total_matched:,} matched filters,"
        f" {len(line_counts):,} unique sequences"
    )

    # Replay unique sequences in parallel (board ops are the bottleneck)
    items = [(h, line_counts[h], all_sequences[h]) for h in line_counts]
    del all_sequences

    num_batches = workers * 4
    chunk_size = max(1, len(items) // num_batches)
    batches = [
        items[i : i + chunk_size] for i in range(0, len(items), chunk_size)
    ]
    del items

    move_counts: dict[tuple[int, int, int, int], int] = defaultdict(int)
    all_dest_hashes: dict[tuple[int, int, int, int], int] = {}

    with mp.Pool(workers) as pool:
        for mc, dh in tqdm(
            pool.imap_unordered(_replay_worker, batches),
            total=len(batches),
            desc="  Replaying",
        ):
            for k, v in mc.items():
                move_counts[k] += v
            for k, v in dh.items():
                if k not in all_dest_hashes:
                    all_dest_hashes[k] = v

    print(f"  {len(move_counts):,} position-moves before pruning")
    return dict(move_counts), all_dest_hashes


def write_book(
    move_counts: dict[tuple[int, int, int, int], int],
    dest_hashes: dict[tuple[int, int, int, int], int],
    output: Path,
    min_count: int = 1,
    min_weight_pct: float = 0.01,
) -> None:
    """Write the binary book file, pruning unreachable positions."""
    from collections import deque

    # Filter by min_count before grouping
    if min_count > 1:
        before = len(move_counts)
        move_counts = {k: v for k, v in move_counts.items() if v >= min_count}
        print(
            f"  Filtered {before:,} -> {len(move_counts):,} (min_count={min_count})"
        )

    # Group by position and compute weights
    positions: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for (h, from_sq, to_sq, promo), count in move_counts.items():
        positions[h].append((from_sq, to_sq, promo, count))

    # Compute weights, drop zero-weight and below-threshold moves
    weighted: dict[int, list[tuple[int, int, int, int, tuple]]] = {}
    for h, entries in positions.items():
        max_count = max(c for _, _, _, c in entries)
        moves = []
        for from_sq, to_sq, promo, count in entries:
            weight = round(count / max_count * 255)
            if weight > 0:
                key = (h, from_sq, to_sq, promo)
                dest = dest_hashes.get(key)
                moves.append((from_sq, to_sq, promo, weight, dest))
        if moves:
            total_weight = sum(w for _, _, _, w, _ in moves)
            threshold = min_weight_pct * total_weight
            moves = [m for m in moves if m[3] >= threshold]
        if moves:
            weighted[h] = moves

    # BFS from starting position to prune unreachable positions
    root = chess.polyglot.zobrist_hash(chess.Board())
    reachable: dict[int, int] = {}  # hash → depth
    queue = deque()
    if root in weighted:
        queue.append((root, 0))
        reachable[root] = 0
    while queue:
        h, d = queue.popleft()
        for _, _, _, _, dest in weighted[h]:
            if dest is not None and dest in weighted and dest not in reachable:
                reachable[dest] = d + 1
                queue.append((dest, d + 1))

    pruned = len(weighted) - len(reachable)
    if pruned > 0:
        print(f"  Pruned {pruned:,} unreachable positions")

    total_moves = sum(len(weighted[h]) for h in reachable)
    print(f"  {len(reachable):,} positions, {total_moves:,} moves after pruning")

    # Count leaf positions: positions where at least one move leads outside the book
    leaf_count = 0
    for h in reachable:
        for _, _, _, _, dest in weighted[h]:
            if dest is None or dest not in reachable:
                leaf_count += 1
                break
    print(f"  {leaf_count:,} leaf positions (at least one move exits book)")

    # Build output tables from reachable positions only
    sorted_hashes = sorted(reachable)
    pos_table = []
    moves_array = []

    for h in sorted_hashes:
        offset = len(moves_array)
        moves = weighted[h]
        moves.sort(key=lambda x: x[3], reverse=True)
        for from_sq, to_sq, promo, weight, _ in moves:
            moves_array.append((from_sq, to_sq, promo, weight))
        pos_table.append((h, offset, len(moves)))

    # Most common line: follow highest-weight move from starting position
    line_prob = 1.0
    cur = root
    line_depth = 0
    while cur in weighted and cur in reachable:
        moves = weighted[cur]
        total_weight = sum(w for _, _, _, w, _ in moves)
        best = max(moves, key=lambda x: x[3])
        line_prob *= best[3] / total_weight
        cur = best[4]
        line_depth += 1
        if cur is None:
            break
    print(f"  Most common line ({line_depth} plies): {line_prob:.2%} of book games")

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
        "--min-elo", type=int, default=1600, help="Min player Elo filter"
    )
    parser.add_argument(
        "--min-time",
        type=int,
        default=180,
        help="Min initial time in seconds (filters bullet; default: 180)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Minimum times a position-move must appear (default: 10)",
    )
    parser.add_argument(
        "--min-weight-pct",
        type=float,
        default=0.01,
        help="Drop moves below this fraction of total weight at their position (default: 0.01)",
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

    move_counts, dest_hashes = build_book(
        _iter_files(args.pgn),
        args.min_elo,
        args.min_time,
        args.workers,
    )

    write_book(
        move_counts, dest_hashes, args.output, args.min_count, args.min_weight_pct
    )


if __name__ == "__main__":
    main()
