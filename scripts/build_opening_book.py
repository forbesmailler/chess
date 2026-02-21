"""Build a binary opening book from a PGN file.

Accepts .pgn or .pgn.zst files. For .zst, streams decompression in ~256 MB
chunks to avoid needing disk space for the full decompressed file.

Phase 1 — Depth analysis: finds maximum depth D where the top N opening
sequences cover >threshold of games.

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


_P1_ENTRY = struct.Struct("<BII")  # depth_u8, hash_u32, count_u32


def _phase1_worker(args):
    """Read a file range and process for depth analysis."""
    filepath, start, end, max_depth, min_elo = args
    chunk = _read_chunk(filepath, start, end).replace(b"\r", b"")
    games = _split_chunk(chunk)
    depth_counters = [Counter() for _ in range(max_depth + 1)]
    matched = 0
    scanned = 0
    seq_hash = _seq_hash

    for game_data in games:
        if len(game_data) < 20:
            continue
        scanned += 1
        if min_elo > 0 and not _check_elo(game_data, min_elo):
            continue
        movetext = _get_movetext(game_data)
        san_moves = _extract_san_moves(movetext, max_depth)
        if not san_moves:
            continue
        matched += 1
        h = 0
        for d, san in enumerate(san_moves, 1):
            h = seq_hash(h, san)
            depth_counters[d][h] += 1

    pack = _P1_ENTRY.pack
    parts = []
    for d in range(1, max_depth + 1):
        for h, c in depth_counters[d].items():
            parts.append(pack(d, h, c))
    return b"".join(parts), matched, scanned


def find_optimal_depth(
    file_iter,
    max_lines: int,
    coverage: float,
    max_depth: int,
    min_elo: int,
    workers: int,
) -> tuple[int, int]:
    """Find the max depth where top N lines cover >coverage of games."""
    print(f"Phase 1: Analyzing opening depth... ({workers} workers)")

    depth_counters = [Counter() for _ in range(max_depth + 1)]
    total_matched = 0
    total_scanned = 0

    with mp.Pool(workers) as pool:
        pbar = tqdm(desc="  Scanning", unit=" games", unit_scale=True)
        for filepath in file_iter:
            boundaries = _find_boundaries(filepath, workers * 10)
            if not boundaries:
                continue
            args_list = [(filepath, s, e, max_depth, min_elo) for s, e in boundaries]
            for data, matched, scanned in pool.imap_unordered(
                _phase1_worker, args_list
            ):
                total_matched += matched
                total_scanned += scanned
                pbar.update(scanned)
                for d, h, c in _P1_ENTRY.iter_unpack(data):
                    depth_counters[d][h] += c
        pbar.close()

    # Close generator to clean up any temp chunk file
    if hasattr(file_iter, "close"):
        file_iter.close()

    if total_matched == 0:
        print("No games found!")
        return 0, 0

    print(
        f"  {total_scanned:,} games scanned,"
        f" {total_matched:,} matched ELO filter"
    )

    optimal_depth = 1
    for depth in range(1, max_depth + 1):
        counter = depth_counters[depth]
        if not counter:
            break
        games_at_depth = sum(counter.values())

        if games_at_depth < total_matched * 0.5:
            break

        top_lines = counter.most_common(max_lines)
        top_coverage = sum(c for _, c in top_lines) / games_at_depth

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
    return optimal_depth, total_matched


_LC_ENTRY = struct.Struct("<II")  # hash_u32, count_u32


def _count_lines_worker(args):
    """Count opening line frequencies and capture one example per hash."""
    filepath, start, end, depth, min_elo = args
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
        movetext = _get_movetext(game_data)
        san_moves = _extract_san_moves(movetext, depth)
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


def build_book(
    file_iter,
    depth: int,
    min_elo: int,
    workers: int,
) -> dict[tuple[int, int, int, int], int]:
    """Count unique move sequences in workers, then replay each once."""
    print(f"Building book at depth {depth}... ({workers} workers)")

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
            args_list = [(filepath, s, e, depth, min_elo) for s, e in boundaries]
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
        f" {total_matched:,} matched ELO filter,"
        f" {len(line_counts):,} unique sequences"
    )

    # Replay each unique sequence once (board ops only here, not in workers)
    move_counts: dict[tuple[int, int, int, int], int] = defaultdict(int)
    depth_positions: set[int] = set()

    for h, weight in line_counts.items():
        san_moves = all_sequences[h]
        board = chess.Board()
        try:
            for san in san_moves:
                pos_hash = chess.polyglot.zobrist_hash(board)
                move = board.parse_san(san)
                promo = move.promotion if move.promotion is not None else 0
                move_counts[(pos_hash, move.from_square, move.to_square, promo)] += (
                    weight
                )
                board.push(move)
            if len(san_moves) == depth:
                depth_positions.add(chess.polyglot.zobrist_hash(board))
        except (
            chess.IllegalMoveError,
            chess.InvalidMoveError,
            chess.AmbiguousMoveError,
        ):
            pass

    print(
        f"  {len(depth_positions):,} unique positions at depth {depth},"
        f" {len(move_counts):,} position-moves in book"
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
        "--lines", type=int, default=250000, help="Max lines for depth analysis"
    )
    parser.add_argument(
        "--coverage", type=float, default=0.5, help="Coverage threshold for depth analysis"
    )
    parser.add_argument(
        "--max-depth", type=int, default=20, help="Max depth for depth analysis"
    )
    parser.add_argument(
        "--min-elo", type=int, default=2200, help="Min player Elo filter"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=0,
        help="Opening depth in plies (default: auto-detect)",
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
            _iter_files(args.pgn),
            args.lines,
            args.coverage,
            args.max_depth,
            args.min_elo,
            args.workers,
        )
        if depth == 0:
            print("No suitable depth found.", file=sys.stderr)
            sys.exit(1)

    move_counts = build_book(
        _iter_files(args.pgn),
        depth,
        args.min_elo,
        args.workers,
    )

    if args.min_count > 1:
        before = len(move_counts)
        move_counts = {k: v for k, v in move_counts.items() if v >= args.min_count}
        print(
            f"  Filtered {before:,} → {len(move_counts):,} (min_count={args.min_count})"
        )

    write_book(move_counts, args.output)


if __name__ == "__main__":
    main()
