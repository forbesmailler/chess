"""Play games against Stockfish via UCI, collect training data, and periodically retrain.

Uses python-chess to mediate between our engine (UCI) and Stockfish (UCI).
Runs multiple games in parallel using multiprocessing. Only our engine's
positions are recorded (with its search eval). After every N games, triggers
a train/compare/archive cycle identical to the normal training loop.
"""

import argparse
import multiprocessing
import struct
import sys
import time
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.load_config import deploy, load

_trn = load("training")
_dep = deploy()
_sp = _trn["self_play"]
_train_cfg = _trn["training"]
_cmp = _trn.get("compare", {})

BOT_EXE = str(Path(_dep["paths"]["bot_exe"]))
POSITION_BYTES = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_nnue_weights():
    pointer_file = PROJECT_ROOT / _dep["paths"]["current_best_file"]
    if pointer_file.exists():
        rel_path = pointer_file.read_text().strip()
        path = PROJECT_ROOT / rel_path
        if path.exists():
            return str(path)
    return ""


def build_engine_cmd(nnue_weights=""):
    cmd = [str(PROJECT_ROOT / BOT_EXE), "--uci"]
    if nnue_weights:
        cmd.append(f"--nnue-weights={nnue_weights}")
    book_path = PROJECT_ROOT / "book.bin"
    if book_path.exists():
        cmd.append(f"--book={book_path}")
    return cmd


def encode_piece(piece):
    if piece is None:
        return 0
    type_map = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }
    base = type_map[piece.piece_type]
    return base if piece.color == chess.WHITE else base + 6


def encode_position(board, search_eval, game_result, ply):
    piece_placement = bytearray(32)
    for sq in range(64):
        piece = board.piece_at(sq)
        nibble = encode_piece(piece)
        byte_idx = sq // 2
        if sq % 2 == 0:
            piece_placement[byte_idx] |= nibble
        else:
            piece_placement[byte_idx] |= nibble << 4

    side_to_move = 0 if board.turn == chess.WHITE else 1

    castling = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling |= 1
    if board.has_queenside_castling_rights(chess.WHITE):
        castling |= 2
    if board.has_kingside_castling_rights(chess.BLACK):
        castling |= 4
    if board.has_queenside_castling_rights(chess.BLACK):
        castling |= 8

    ep_file = 255
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)

    return struct.pack(
        "<32sBBBfBH",
        bytes(piece_placement),
        side_to_move,
        castling,
        ep_file,
        search_eval,
        game_result,
        ply,
    )


def play_game(our_engine, stockfish, our_color, move_time_ms):
    """Play one game, return (result_str, list of encoded position bytes)."""
    board = chess.Board()
    positions = []
    ply = 0
    limit = chess.engine.Limit(time=move_time_ms / 1000.0)

    while not board.is_game_over(claim_draw=True):
        is_our_turn = board.turn == our_color

        if is_our_turn:
            # Use analysis to capture score, then extract best move
            score = None
            move = None
            with our_engine.analysis(board, limit) as analysis:
                for info in analysis:
                    if "score" in info:
                        sc = info["score"].relative
                        if sc.is_mate():
                            score = 10000.0 if sc.mate() > 0 else -10000.0
                        else:
                            score = float(sc.score()) / 100.0
                    if "pv" in info and info["pv"]:
                        move = info["pv"][0]

            if move is None:
                # Fallback if analysis didn't yield a move
                result = our_engine.play(board, limit)
                move = result.move

            if score is not None:
                positions.append((board.copy(), score, ply))
        else:
            result = stockfish.play(board, limit)
            move = result.move

        board.push(move)
        ply += 1

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        result_str = "draw"
    elif outcome.winner == our_color:
        result_str = "win"
    else:
        result_str = "loss"

    encoded = []
    for pos_board, eval_score, pos_ply in positions:
        stm_is_our_color = pos_board.turn == our_color
        if result_str == "win":
            stm_result = 2 if stm_is_our_color else 0
        elif result_str == "loss":
            stm_result = 0 if stm_is_our_color else 2
        else:
            stm_result = 1
        encoded.append(encode_position(pos_board, eval_score, stm_result, pos_ply))

    return result_str, encoded


def worker(
    worker_id,
    num_games,
    total_games,
    nnue_weights,
    sf_path,
    sf_elo,
    move_time_ms,
    data_path,
    game_counter,
    position_counter,
    lock,
    start_time_val,
    log_interval,
):
    """Worker process: plays num_games sequentially, writing positions to data_path."""
    engine_cmd = build_engine_cmd(nnue_weights)
    our_engine = chess.engine.SimpleEngine.popen_uci(engine_cmd)
    stockfish = chess.engine.SimpleEngine.popen_uci(str(sf_path))
    stockfish.configure({"UCI_LimitStrength": True, "UCI_Elo": sf_elo})

    wins = losses = draws = positions_count = 0

    for i in range(num_games):
        our_color = chess.WHITE if i % 2 == 0 else chess.BLACK

        try:
            result_str, encoded = play_game(
                our_engine, stockfish, our_color, move_time_ms
            )
        except Exception as e:
            print(f"  [Worker {worker_id}] Game error: {e}", flush=True)
            continue

        if result_str == "win":
            wins += 1
        elif result_str == "loss":
            losses += 1
        else:
            draws += 1

        n_pos = len(encoded) if encoded else 0
        if encoded:
            with lock:
                with open(data_path, "ab") as f:
                    for pos_bytes in encoded:
                        f.write(pos_bytes)
        positions_count += n_pos

        with lock:
            game_counter.value += 1
            position_counter.value += n_pos
            completed = game_counter.value
            total_pos = position_counter.value

        if completed % log_interval == 0 or completed == total_games:
            elapsed = int(time.time() - start_time_val)
            print(
                f"Stockfish progress: {completed}/{total_games} games, "
                f"{total_pos} positions, {elapsed}s elapsed",
                flush=True,
            )

    our_engine.quit()
    stockfish.quit()

    return wins, losses, draws, positions_count


def run_training_cycle(data_path, args):
    """Run train -> export -> compare -> archive, same as train_loop.py."""
    import shutil
    import tempfile
    from datetime import datetime

    from scripts.train_loop import (
        cap_training_data,
        read_current_best,
        run,
        run_compare,
        write_current_best,
        write_report,
    )

    pointer_file = Path(_dep["paths"]["current_best_file"])
    models_dir = Path("models")
    accepted_dir = Path(_dep["paths"]["accepted_models_dir"])
    rejected_dir = Path(_dep["paths"]["rejected_models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    cap_training_data(data_path)
    total_positions = data_path.stat().st_size // POSITION_BYTES
    print(f"Total accumulated positions: {total_positions}")

    print(f"\n--- Train NNUE ({args.epochs} epochs, {total_positions} positions) ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_name = f"nnue_{timestamp}_{total_positions}pos.bin"
    log_path = Path(tempfile.gettempdir()) / candidate_name.replace(".bin", "_train.md")
    run(
        f"python -u engine/train/train_nnue.py --data {data_path}"
        f" --output engine/train/nnue_weights.pt"
        f" --epochs {args.epochs} --batch-size {args.batch_size}"
        f" --eval-weight {args.eval_weight}"
        f" --log {log_path}"
    )

    candidate_path = models_dir / candidate_name
    print(f"\n--- Export candidate ({candidate_name}) ---")
    run(
        f"python engine/train/export_nnue.py"
        f" --model engine/train/nnue_weights.pt"
        f" --output {candidate_path}"
    )

    if not candidate_path.exists() or candidate_path.stat().st_size == 0:
        print(f"Error: candidate missing or empty at {candidate_path}")
        candidate_path.unlink(missing_ok=True)
        return None

    current_best = read_current_best(pointer_file)
    best_path = Path(current_best) if current_best else None
    old_name = Path(current_best).stem if current_best else "handcrafted"
    old_arg = str(best_path) if best_path else "handcrafted"

    print(
        f"\n--- Compare: {candidate_name} vs {old_name} ({args.compare_games} games) ---"
    )
    wld = run_compare(
        f"{BOT_EXE} --compare {old_arg} {candidate_path}"
        f" {args.compare_games} {data_path} {args.threads}"
    )

    if wld["improved"]:
        archive_path = accepted_dir / candidate_name
        shutil.move(str(candidate_path), str(archive_path))
        write_current_best(pointer_file, str(archive_path))
        status = "ACCEPTED"
    else:
        archive_path = rejected_dir / candidate_name
        shutil.move(str(candidate_path), str(archive_path))
        status = "REJECTED"

    if log_path.exists():
        shutil.move(str(log_path), str(archive_path.parent / log_path.name))

    write_report(archive_path, status, candidate_path.stem, old_name, wld)

    current_best = read_current_best(pointer_file)
    best_label = current_best if current_best else "handcrafted"
    print(f"\n--- Training cycle: {status} ---")
    print(f"Archived to: {archive_path}")
    print(f"Current best: {best_label}")

    return resolve_nnue_weights() if status == "ACCEPTED" else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("elo", type=int, help="Stockfish Elo limit (e.g. 2600)")
    p.add_argument(
        "--stockfish",
        default="stockfish",
        help="Path to Stockfish executable (default: stockfish)",
    )
    p.add_argument(
        "--games",
        type=int,
        default=_sp["num_games"],
        help=f"Games per training cycle (default: {_sp['num_games']})",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=_sp["num_threads"],
        help=f"Number of parallel workers (default: {_sp['num_threads']})",
    )
    p.add_argument(
        "--move-time",
        type=int,
        default=_sp["search_time_ms"],
        help=f"Think time per move in ms (default: {_sp['search_time_ms']})",
    )
    p.add_argument(
        "--data",
        default=_sp["output_file"],
        help=f"Training data path (default: {_sp['output_file']})",
    )
    p.add_argument("--epochs", type=int, default=_train_cfg["epochs"])
    p.add_argument("--batch-size", type=int, default=_train_cfg["batch_size"])
    p.add_argument("--eval-weight", type=float, default=_train_cfg["eval_weight"])
    p.add_argument("--compare-games", type=int, default=_cmp.get("num_games", 100))
    p.add_argument(
        "--no-retrain",
        action="store_true",
        help="Just play games and collect data, don't retrain",
    )
    args = p.parse_args()

    # Verify Stockfish exists
    sf_path = Path(args.stockfish)
    if not sf_path.exists() and args.stockfish == "stockfish":
        for candidate in [
            Path("stockfish/stockfish.exe"),
            Path("stockfish/stockfish"),
            Path("C:/stockfish/stockfish.exe"),
            Path(
                "C:/Users/forbe/AppData/Local/Microsoft/WinGet/Packages/"
                "Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe/"
                "stockfish/stockfish-windows-x86-64-avx2.exe"
            ),
        ]:
            if candidate.exists():
                sf_path = candidate
                break
    sf_path = sf_path.resolve()
    if not sf_path.exists() and args.stockfish == "stockfish":
        sys.exit(
            "Stockfish not found. Install it or pass --stockfish=/path/to/stockfish"
        )

    data_path = Path(args.data)
    nnue_weights = resolve_nnue_weights()
    eval_label = f"NNUE ({Path(nnue_weights).name})" if nnue_weights else "handcrafted"

    print(f"=== Play vs Stockfish {args.elo} ===")
    print(f"Our eval: {eval_label}")
    print(f"Move time: {args.move_time}ms")
    print(f"Workers: {args.threads}")
    print(f"Games per cycle: {args.games}")
    print(f"Training data: {args.data}")
    if not args.no_retrain:
        print(f"Retrain every {args.games} games")
    print()

    iteration = 0
    total_games = 0
    total_wins = 0
    total_losses = 0
    total_draws = 0

    while True:
        iteration += 1
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {iteration} (total games: {total_games}) ===")
        print(f"{'=' * 60}")

        # Distribute games across workers
        num_workers = min(args.threads, args.games)
        base = args.games // num_workers
        remainder = args.games % num_workers
        games_per_worker = [
            base + (1 if i < remainder else 0) for i in range(num_workers)
        ]

        log_interval = _sp.get("progress_log_interval", 10)

        manager = multiprocessing.Manager()
        game_counter = manager.Value("i", 0)
        position_counter = manager.Value("i", 0)
        lock = manager.Lock()

        start_time = time.time()
        eval_label = (
            f"NNUE ({Path(nnue_weights).name})" if nnue_weights else "handcrafted"
        )
        print(f"=== Stockfish {args.elo} Data Generation ===")
        print(f"Games: {args.games}")
        print(f"Output: {args.data}")
        print(f"Threads: {num_workers}")
        print(f"Eval: {eval_label}")

        with multiprocessing.Pool(num_workers) as pool:
            results = pool.starmap(
                worker,
                [
                    (
                        i,
                        games_per_worker[i],
                        args.games,
                        nnue_weights,
                        sf_path,
                        args.elo,
                        args.move_time,
                        str(data_path),
                        game_counter,
                        position_counter,
                        lock,
                        start_time,
                        log_interval,
                    )
                    for i in range(num_workers)
                ],
            )

        elapsed = time.time() - start_time
        cycle_wins = sum(r[0] for r in results)
        cycle_losses = sum(r[1] for r in results)
        cycle_draws = sum(r[2] for r in results)
        cycle_positions = sum(r[3] for r in results)
        cycle_games = cycle_wins + cycle_losses + cycle_draws

        total_games += cycle_games
        total_wins += cycle_wins
        total_losses += cycle_losses
        total_draws += cycle_draws

        print(
            f"Stockfish complete: {cycle_games} games, "
            f"{cycle_positions} positions in {elapsed:.0f}s"
        )
        print(f"W/L/D: {cycle_wins}/{cycle_losses}/{cycle_draws}")
        print(
            f"Overall: W/L/D = {total_wins}/{total_losses}/{total_draws} "
            f"({total_games} games)"
        )

        if not args.no_retrain:
            print("\n--- Retraining ---")
            new_weights = run_training_cycle(data_path, args)
            if new_weights:
                nnue_weights = new_weights
                print(f"Updated to new weights: {nnue_weights}")
            else:
                nnue_weights = resolve_nnue_weights()


if __name__ == "__main__":
    main()
