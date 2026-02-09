"""Continuous RL training loop: self-play -> train -> compare -> repeat."""

import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


def _load(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


_trn = _load("training")
_dep = _load("deploy")

BOT_EXE = str(Path(_dep["paths"]["bot_exe"]))
_sp = _trn["self_play"]
_train_cfg = _trn["training"]
_inv = _trn["invoke"]
_cmp = _trn.get("compare", {})

POSITION_BYTES = 42


def run(cmd: str) -> subprocess.CompletedProcess:
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=True)


def run_check(cmd: str) -> bool:
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=_inv["train_games"])
    p.add_argument("--depth", type=int, default=_sp["search_depth"])
    p.add_argument("--threads", type=int, default=_inv["train_threads"])
    p.add_argument("--data", default=_sp["output_file"])
    p.add_argument("--weights", default=_inv["weights"])
    p.add_argument("--epochs", type=int, default=_train_cfg["epochs"])
    p.add_argument("--batch-size", type=int, default=_train_cfg["batch_size"])
    p.add_argument("--eval-weight", type=float, default=_train_cfg["eval_weight"])
    p.add_argument("--compare-games", type=int, default=_cmp.get("num_games", 100))
    p.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip self-play and training; just compare candidate vs current best and archive.",
    )
    p.add_argument(
        "--train-only",
        action="store_true",
        help="Skip self-play; just train, export, compare, and archive.",
    )
    args = p.parse_args()

    weights_path = Path(args.weights)
    data_path = Path(args.data)
    candidate_path = weights_path.with_name("nnue_candidate.bin")
    accepted_dir = Path(_dep["paths"]["accepted_models_dir"])
    rejected_dir = Path(_dep["paths"]["rejected_models_dir"])
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {iteration} ===")
        print(f"{'=' * 60}")

        if not args.compare_only and not args.train_only:
            # 1. Self-play using current best model (or handcrafted)
            weights_arg = str(weights_path) if weights_path.exists() else ""
            eval_label = (
                f"NNUE ({args.weights})" if weights_path.exists() else "handcrafted"
            )
            print(f"\n--- Self-play ({args.games} games, eval: {eval_label}) ---")
            selfplay_cmd = (
                f"{BOT_EXE} --selfplay {args.games} {args.depth}"
                f" {args.data} {args.threads}"
            )
            if weights_arg:
                selfplay_cmd += f" {weights_arg}"
            run(selfplay_cmd)

        if not args.compare_only:
            total_positions = data_path.stat().st_size // POSITION_BYTES
            print(f"Total accumulated positions: {total_positions}")

            # 2. Train NNUE from full accumulated data
            print(
                f"\n--- Train NNUE"
                f" ({args.epochs} epochs, {total_positions} positions) ---"
            )
            run(
                f"python engine/train/train_nnue.py --data {args.data}"
                f" --output engine/train/nnue_weights.pt"
                f" --epochs {args.epochs} --batch-size {args.batch_size}"
                f" --eval-weight {args.eval_weight}"
            )

            # 3. Export to candidate
            print("\n--- Export candidate ---")
            run(
                f"python engine/train/export_nnue.py"
                f" --model engine/train/nnue_weights.pt"
                f" --output {candidate_path}"
            )

        if not candidate_path.exists():
            print(f"Error: candidate not found at {candidate_path}")
            break

        # 4. Compare candidate vs current best
        old_arg = str(weights_path) if weights_path.exists() else "handcrafted"
        print(f"\n--- Compare: candidate vs {old_arg} ({args.compare_games} games) ---")
        improved = run_check(
            f"{BOT_EXE} --compare {old_arg} {candidate_path}"
            f' {args.compare_games} "" {args.threads}'
        )

        # 5. Archive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        total_positions = (
            data_path.stat().st_size // POSITION_BYTES if data_path.exists() else 0
        )
        archive_name = f"nnue_{timestamp}_{total_positions}pos.bin"

        if improved:
            shutil.copy2(candidate_path, weights_path)
            archive_path = accepted_dir / archive_name
            shutil.copy2(candidate_path, archive_path)
            status = "ACCEPTED"
        else:
            archive_path = rejected_dir / archive_name
            shutil.copy2(candidate_path, archive_path)
            status = "REJECTED"

        candidate_path.unlink(missing_ok=True)

        # 6. Summary
        print(f"\n--- Iteration {iteration}: {status} ---")
        print(f"Archived to: {archive_path}")
        print(f"Total positions: {total_positions}")
        best = args.weights if weights_path.exists() else "handcrafted"
        print(f"Current best: {best}")


if __name__ == "__main__":
    main()
