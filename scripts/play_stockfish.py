"""Play games against Stockfish via UCI, collect training data, and periodically retrain.

Uses python-chess to mediate between our engine (UCI) and Stockfish (UCI).
Only our engine's positions are recorded (with its search eval). Game results
are written once each game finishes. After every N games, triggers a
train/compare/archive cycle identical to the normal training loop.
"""

import argparse
import struct
import sys
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
    cmd = [BOT_EXE, "--uci"]
    if nnue_weights:
        cmd.append(f"--nnue-weights={nnue_weights}")
    book_path = PROJECT_ROOT / "book.bin"
    if book_path.exists():
        cmd.append(f"--book={book_path}")
    return cmd


def encode_piece(piece):
    """Encode a chess.Piece to our 4-bit format."""
    if piece is None:
        return 0
    # 1=wP,2=wN,3=wB,4=wR,5=wQ,6=wK, 7=bP,8=bN,9=bB,10=bR,11=bQ,12=bK
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
    """Encode a position into 42-byte TrainingPosition format.

    search_eval: from side-to-move's perspective
    game_result: from side-to-move's perspective (0=loss, 1=draw, 2=win)
    """
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


def play_game(our_engine, stockfish, our_color, move_time_ms, game_num):
    """Play one game, return (result_str, list of encoded position bytes)."""
    board = chess.Board()
    positions = []  # (board_copy, eval_from_stm, ply)
    ply = 0

    while not board.is_game_over(claim_draw=True):
        is_our_turn = board.turn == our_color

        if is_our_turn:
            result = our_engine.play(
                board, chess.engine.Limit(time=move_time_ms / 1000.0)
            )
            move = result.move
            # Get eval from info if available
            score = None
            if result.info and "score" in result.info:
                sc = result.info["score"].relative
                if sc.is_mate():
                    score = 10000.0 if sc.mate() > 0 else -10000.0
                else:
                    score = float(sc.score()) / 100.0

            if score is not None:
                positions.append((board.copy(), score, ply))
        else:
            result = stockfish.play(
                board, chess.engine.Limit(time=move_time_ms / 1000.0)
            )
            move = result.move

        board.push(move)
        ply += 1

    # Determine result
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result_str = "draw"
    elif outcome.winner == our_color:
        result_str = "win"
    elif outcome.winner is None:
        result_str = "draw"
    else:
        result_str = "loss"

    # Encode positions with game result
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
    p.add_argument("--threads", type=int, default=_sp["num_threads"])
    p.add_argument(
        "--no-retrain",
        action="store_true",
        help="Just play games and collect data, don't retrain",
    )
    args = p.parse_args()

    # Verify Stockfish exists
    sf_path = Path(args.stockfish)
    if not sf_path.exists() and args.stockfish == "stockfish":
        # Try common locations
        for p in [
            Path("stockfish/stockfish.exe"),
            Path("stockfish/stockfish"),
            Path("C:/stockfish/stockfish.exe"),
        ]:
            if p.exists():
                sf_path = p
                break
    if not sf_path.exists() and args.stockfish == "stockfish":
        sys.exit(
            "Stockfish not found. Install it or pass --stockfish=/path/to/stockfish"
        )

    data_path = Path(args.data)
    nnue_weights = resolve_nnue_weights()
    eval_label = f"NNUE ({nnue_weights})" if nnue_weights else "handcrafted"

    print(f"=== Play vs Stockfish {args.elo} ===")
    print(f"Our eval: {eval_label}")
    print(f"Move time: {args.move_time}ms")
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

        # Start engines
        engine_cmd = build_engine_cmd(nnue_weights)
        print(f"Starting our engine: {' '.join(engine_cmd)}")
        our_engine = chess.engine.SimpleEngine.popen_uci(engine_cmd)

        print(f"Starting Stockfish: {sf_path} (Elo {args.elo})")
        stockfish = chess.engine.SimpleEngine.popen_uci(str(sf_path))
        stockfish.configure({"UCI_LimitStrength": True, "UCI_Elo": args.elo})

        wins = losses = draws = 0
        cycle_positions = 0

        for game_num in range(1, args.games + 1):
            # Alternate colors
            our_color = chess.WHITE if game_num % 2 == 1 else chess.BLACK
            color_str = "white" if our_color == chess.WHITE else "black"

            try:
                result_str, encoded = play_game(
                    our_engine, stockfish, our_color, args.move_time, game_num
                )
            except Exception as e:
                print(f"  Game {game_num} error: {e}")
                break

            if result_str == "win":
                wins += 1
            elif result_str == "loss":
                losses += 1
            else:
                draws += 1

            # Write positions to training data
            if encoded:
                with open(data_path, "ab") as f:
                    for pos_bytes in encoded:
                        f.write(pos_bytes)
                cycle_positions += len(encoded)

            total_games += 1
            if game_num % 10 == 0 or game_num == args.games:
                print(
                    f"  Game {game_num}/{args.games} as {color_str}: {result_str} "
                    f"| Cycle W/L/D: {wins}/{losses}/{draws} "
                    f"| Positions: {cycle_positions}"
                )

        # Close engines
        our_engine.quit()
        stockfish.quit()

        total_wins += wins
        total_losses += losses
        total_draws += draws

        print(f"\nCycle {iteration} complete: W/L/D = {wins}/{losses}/{draws}")
        print(
            f"Overall: W/L/D = {total_wins}/{total_losses}/{total_draws} ({total_games} games)"
        )
        print(f"Positions this cycle: {cycle_positions}")

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
