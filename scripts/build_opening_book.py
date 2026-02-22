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
import numpy as np
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
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pgn", mode="wb")
        total = 0
        if leftover[0]:
            tmp.write(leftover[0])
            total += len(leftover[0])
            leftover[0] = b""
        eof = False
        while total < chunk_size:
            data = reader.read(1 << 22)
            if not data:
                eof = True
                break
            clean = data.replace(b"\r", b"")
            tmp.write(clean)
            total += len(clean)
        if total == 0:
            tmp.close()
            os.unlink(tmp.name)
            return None
        tmp.close()
        if not eof:
            # Read tail of file to find last game boundary
            tail_size = min(total, 1 << 20)  # 1 MB tail search
            with open(tmp.name, "rb") as f:
                f.seek(total - tail_size)
                tail = f.read()
            boundary = tail.rfind(_GAME_SEP)
            if boundary > 0:
                split_pos = (total - tail_size) + boundary + 1
                with open(tmp.name, "rb") as f:
                    f.seek(split_pos)
                    leftover[0] = f.read()
                # Truncate file at boundary
                with open(tmp.name, "r+b") as f:
                    f.truncate(split_pos)
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


_FH_ENTRY = struct.Struct("<IQ")  # seq_hash_u32, final_pos_hash_u64


def _count_lines_worker(args):
    """Count opening line frequencies, capture one example + final hash per seq."""
    filepath, start, end, min_elo, min_time = args
    chunk = _read_chunk(filepath, start, end).replace(b"\r", b"")
    games = _split_chunk(chunk)
    line_counts = Counter()
    sequences = {}
    final_hashes = {}
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
            sequences[h] = " ".join(san_moves)
            board = chess.Board()
            try:
                for san in san_moves:
                    board.push(board.parse_san(san))
                final_hashes[h] = chess.polyglot.zobrist_hash(board)
            except (
                chess.IllegalMoveError,
                chess.InvalidMoveError,
                chess.AmbiguousMoveError,
            ):
                pass

    lc_pack = _LC_ENTRY.pack
    data = b"".join(lc_pack(h, c) for h, c in line_counts.items())
    fh_pack = _FH_ENTRY.pack
    fh_data = b"".join(fh_pack(h, fh) for h, fh in final_hashes.items())
    return data, matched, scanned, sequences, fh_data


_REPLAY_ENTRY = struct.Struct("<QBBBxI")  # 16 bytes per entry

_SEQ_HEADER = struct.Struct("<IH")  # seq_hash_u32, san_len_u16

_REPLAY_DTYPE = np.dtype(
    [
        ("pos_hash", "<u8"),
        ("from_sq", "u1"),
        ("to_sq", "u1"),
        ("promo", "u1"),
        ("_pad", "u1"),
        ("count", "<u4"),
    ]
)


def _numpy_aggregate(arr):
    """Sort structured array by key fields and sum counts for duplicates."""
    if len(arr) == 0:
        return arr
    arr.sort(order=["pos_hash", "from_sq", "to_sq", "promo"])
    n = len(arr)
    diff_mask = np.empty(n, dtype=bool)
    diff_mask[0] = True
    diff_mask[1:] = (
        (arr["pos_hash"][1:] != arr["pos_hash"][:-1])
        | (arr["from_sq"][1:] != arr["from_sq"][:-1])
        | (arr["to_sq"][1:] != arr["to_sq"][:-1])
        | (arr["promo"][1:] != arr["promo"][:-1])
    )
    boundaries = np.nonzero(diff_mask)[0]
    summed = np.add.reduceat(arr["count"], boundaries)
    result = arr[boundaries].copy()
    result["count"] = summed
    return result


def _iter_replay_batches(seq_path, survivors, batch_size=5000):
    """Yield batches of (seq_hash, count, san_str) from sequence temp file."""
    header_size = _SEQ_HEADER.size
    unpack = _SEQ_HEADER.unpack
    batch = []
    with open(seq_path, "rb") as f:
        while True:
            hdr = f.read(header_size)
            if len(hdr) < header_size:
                break
            seq_hash, san_len = unpack(hdr)
            san_bytes = f.read(san_len)
            if seq_hash in survivors:
                batch.append((seq_hash, survivors[seq_hash], san_bytes.decode("utf-8")))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def _replay_worker(items):
    """Replay unique opening sequences, return packed binary data."""
    move_counts = defaultdict(int)
    for h, weight, san_str in items:
        board = chess.Board()
        try:
            for san in san_str.split():
                pos_hash = chess.polyglot.zobrist_hash(board)
                move = board.parse_san(san)
                promo = move.promotion if move.promotion is not None else 0
                key = (pos_hash, move.from_square, move.to_square, promo)
                move_counts[key] += weight
                board.push(move)
        except (
            chess.IllegalMoveError,
            chess.InvalidMoveError,
            chess.AmbiguousMoveError,
        ):
            pass
    pack = _REPLAY_ENTRY.pack
    return b"".join(
        pack(ph, fs, ts, pr, c) for (ph, fs, ts, pr), c in move_counts.items()
    )


def build_book(
    file_iter,
    min_elo: int,
    min_time: int,
    min_count: int,
    workers: int,
) -> dict[tuple[int, int, int, int], int]:
    """Count unique move sequences in workers, then replay each once."""
    print(f"Building book... ({workers} workers)")

    line_counts = Counter()
    all_final_hashes: dict[int, int] = {}
    seen_seqs: set[int] = set()
    total_matched = 0
    total_scanned = 0

    # Write unique sequences to temp file (avoids ~8GB dict in memory)
    seq_fd, seq_path = tempfile.mkstemp(suffix=".seq")
    seq_file = os.fdopen(seq_fd, "wb")
    seq_pack = _SEQ_HEADER.pack

    with mp.Pool(workers) as pool:
        pbar = tqdm(desc="  Scanning", unit=" games", unit_scale=True)
        for filepath in file_iter:
            boundaries = _find_boundaries(filepath, workers * 10)
            if not boundaries:
                continue
            args_list = [(filepath, s, e, min_elo, min_time) for s, e in boundaries]
            for data, matched, scanned, sequences, fh_data in pool.imap_unordered(
                _count_lines_worker, args_list
            ):
                total_matched += matched
                total_scanned += scanned
                pbar.update(scanned)
                for h, c in _LC_ENTRY.iter_unpack(data):
                    line_counts[h] += c
                for h, seq in sequences.items():
                    if h not in seen_seqs:
                        seen_seqs.add(h)
                        san_bytes = seq.encode("utf-8")
                        seq_file.write(seq_pack(h, len(san_bytes)))
                        seq_file.write(san_bytes)
                for h, fh in _FH_ENTRY.iter_unpack(fh_data):
                    if h not in all_final_hashes:
                        all_final_hashes[h] = fh
        pbar.close()
    seq_file.close()
    del seen_seqs

    print(
        f"  {total_scanned:,} games scanned,"
        f" {total_matched:,} matched filters,"
        f" {len(line_counts):,} unique sequences"
    )

    # Filter non-unique game endings
    final_pos_game_count: Counter = Counter()
    for seq_hash, final_hash in all_final_hashes.items():
        if seq_hash in line_counts:
            final_pos_game_count[final_hash] += line_counts[seq_hash]

    before_seqs = len(line_counts)
    survivors = {
        h: c
        for h, c in line_counts.items()
        if h in all_final_hashes and final_pos_game_count[all_final_hashes[h]] == 1
    }
    print(f"  Unique final positions: {before_seqs:,} -> {len(survivors):,} sequences")
    del line_counts, all_final_hashes, final_pos_game_count

    # Replay: stream batches from seq file, write results to replay temp file
    n_survivors = len(survivors)
    total_batches = (n_survivors + 4999) // 5000
    replay_fd, replay_path = tempfile.mkstemp(suffix=".bin")
    replay_file = os.fdopen(replay_fd, "wb")

    with mp.Pool(workers) as pool:
        for data in tqdm(
            pool.imap_unordered(
                _replay_worker,
                _iter_replay_batches(seq_path, survivors),
            ),
            total=total_batches,
            desc="  Replaying",
        ):
            replay_file.write(data)
    replay_file.close()
    os.unlink(seq_path)
    del survivors

    # Bucket-based aggregation: distribute by pos_hash prefix, sort each bucket
    file_size = os.path.getsize(replay_path)
    entry_size = _REPLAY_ENTRY.size
    n_entries = file_size // entry_size
    print(f"  {n_entries:,} raw position-move entries, aggregating...")

    N_BUCKETS = 64
    BUCKET_SHIFT = 64 - 6  # top 6 bits -> 64 buckets
    bucket_paths = []
    bucket_files = []
    for _ in range(N_BUCKETS):
        fd, path = tempfile.mkstemp(suffix=".bkt")
        bucket_files.append(os.fdopen(fd, "wb"))
        bucket_paths.append(path)

    # Distribute entries into buckets
    read_n = 10_000_000  # 10M entries per read = 160MB
    with open(replay_path, "rb") as f:
        for _ in tqdm(
            range(0, max(1, n_entries), read_n),
            desc="  Distributing",
            total=(n_entries + read_n - 1) // max(1, read_n),
        ):
            buf = f.read(read_n * entry_size)
            if not buf:
                break
            arr = np.frombuffer(buf, dtype=_REPLAY_DTYPE)
            bids = (arr["pos_hash"] >> BUCKET_SHIFT).astype(np.uint8)
            order = bids.argsort(kind="stable")
            sorted_arr = arr[order]
            sorted_bids = bids[order]
            bounds = np.searchsorted(sorted_bids, np.arange(N_BUCKETS + 1))
            for b in range(N_BUCKETS):
                s, e = int(bounds[b]), int(bounds[b + 1])
                if s < e:
                    bucket_files[b].write(sorted_arr[s:e].tobytes())

    for bf in bucket_files:
        bf.close()
    os.unlink(replay_path)

    # Sort + aggregate each bucket independently
    move_counts: dict[tuple[int, int, int, int], int] = {}
    for b in tqdm(range(N_BUCKETS), desc="  Aggregating"):
        bsize = os.path.getsize(bucket_paths[b])
        if bsize == 0:
            os.unlink(bucket_paths[b])
            continue
        arr = np.fromfile(bucket_paths[b], dtype=_REPLAY_DTYPE)
        os.unlink(bucket_paths[b])
        arr = _numpy_aggregate(arr)
        mask = arr["count"] >= min_count
        arr = arr[mask]
        if len(arr) > 0:
            ph = arr["pos_hash"].tolist()
            fs = arr["from_sq"].tolist()
            ts = arr["to_sq"].tolist()
            pr = arr["promo"].tolist()
            ct = arr["count"].tolist()
            for i in range(len(arr)):
                move_counts[(ph[i], fs[i], ts[i], pr[i])] = ct[i]
        del arr

    print(f"  {len(move_counts):,} position-moves after min_count={min_count}")
    return move_counts


def write_book(
    move_counts: dict[tuple[int, int, int, int], int],
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
        print(f"  Filtered {before:,} -> {len(move_counts):,} (min_count={min_count})")

    # Group by position and compute weights
    positions: dict[int, list] = defaultdict(list)
    for (h, from_sq, to_sq, promo), count in move_counts.items():
        positions[h].append((from_sq, to_sq, promo, count))

    # Compute weights, drop zero-weight and below-threshold moves
    weighted: dict[int, list] = {}
    for h, entries in positions.items():
        max_count = max(c for _, _, _, c in entries)
        moves = []
        for from_sq, to_sq, promo, count in entries:
            weight = round(count / max_count * 255)
            if weight > 0:
                moves.append((from_sq, to_sq, promo, weight))
        if moves:
            total_weight = sum(w for _, _, _, w in moves)
            threshold = min_weight_pct * total_weight
            moves = [m for m in moves if m[3] >= threshold]
        if moves:
            weighted[h] = moves

    # BFS from starting position; compute dest hashes on the fly
    root_board = chess.Board()
    root = chess.polyglot.zobrist_hash(root_board)
    reachable: dict[int, int] = {}
    boards: dict[int, chess.Board] = {}
    dest_map: dict[tuple[int, int, int, int], int] = {}
    queue = deque()
    if root in weighted:
        queue.append((root, 0))
        reachable[root] = 0
        boards[root] = root_board
    while queue:
        h, d = queue.popleft()
        board = boards[h]
        for from_sq, to_sq, promo, _ in weighted[h]:
            uci = chess.SQUARE_NAMES[from_sq] + chess.SQUARE_NAMES[to_sq]
            if promo:
                uci += {2: "n", 3: "b", 4: "r", 5: "q"}[promo]
            try:
                b = board.copy()
                b.push_uci(uci)
                dest = chess.polyglot.zobrist_hash(b)
                dest_map[(h, from_sq, to_sq, promo)] = dest
                if dest in weighted and dest not in reachable:
                    reachable[dest] = d + 1
                    boards[dest] = b
                    queue.append((dest, d + 1))
            except (ValueError, chess.IllegalMoveError):
                pass
        del boards[h]

    pruned = len(weighted) - len(reachable)
    if pruned > 0:
        print(f"  Pruned {pruned:,} unreachable positions")

    total_moves = sum(len(weighted[h]) for h in reachable)
    print(f"  {len(reachable):,} positions, {total_moves:,} moves after pruning")

    # Count leaf positions
    leaf_count = 0
    for h in reachable:
        for from_sq, to_sq, promo, _ in weighted[h]:
            dest = dest_map.get((h, from_sq, to_sq, promo))
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
        for from_sq, to_sq, promo, weight in moves:
            moves_array.append((from_sq, to_sq, promo, weight))
        pos_table.append((h, offset, len(moves)))

    # Most common line: follow highest-weight move from starting position
    line_prob = 1.0
    cur = root
    line_depth = 0
    while cur in weighted and cur in reachable:
        moves = weighted[cur]
        total_weight = sum(w for _, _, _, w in moves)
        best = max(moves, key=lambda x: x[3])
        line_prob *= best[3] / total_weight
        cur = dest_map.get((cur, best[0], best[1], best[2]))
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
        "--min-elo", type=int, default=1800, help="Min player Elo filter"
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

    move_counts = build_book(
        _iter_files(args.pgn),
        args.min_elo,
        args.min_time,
        args.min_count,
        args.workers,
    )

    write_book(move_counts, args.output, args.min_count, args.min_weight_pct)


if __name__ == "__main__":
    main()
