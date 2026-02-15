"""Continuous RL training loop: self-play -> train -> compare -> repeat."""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.load_config import load

_eng = load("engine")
_trn = load("training")
_dep = load("deploy")

BOT_EXE = str(Path(_dep["paths"]["bot_exe"]))
_sp = _trn["self_play"]
_train_cfg = _trn["training"]
_inv = _trn["invoke"]
_cmp = _trn.get("compare", {})

POSITION_BYTES = 42

_WLD_RE = re.compile(r"New wins:\s*(\d+),\s*Old wins:\s*(\d+),\s*Draws:\s*(\d+)")
DATA_CAP_MULTIPLIER = 100


def compute_param_count() -> int:
    nnue = _eng["nnue"]
    inp, h1, h2, out = (
        nnue["input_size"],
        nnue["hidden1_size"],
        nnue["hidden2_size"],
        nnue["output_size"],
    )
    return inp * h1 + h1 + h1 * h2 + h2 + h2 * out + out


def cap_training_data(data_path: Path) -> None:
    if not data_path.exists():
        return
    file_size = data_path.stat().st_size
    total_positions = file_size // POSITION_BYTES
    max_positions = DATA_CAP_MULTIPLIER * compute_param_count()
    if total_positions <= max_positions:
        return
    trim_count = total_positions - max_positions
    keep_offset = trim_count * POSITION_BYTES
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = data_path.with_name(f"{data_path.stem}_{timestamp}{data_path.suffix}")
    shutil.copy2(str(data_path), str(backup_path))
    print(f"Backed up {data_path.name} -> {backup_path.name}")
    tmp_path = data_path.with_suffix(".tmp")
    with open(data_path, "rb") as src, open(tmp_path, "wb") as dst:
        src.seek(keep_offset)
        while chunk := src.read(1 << 20):
            dst.write(chunk)
    tmp_path.replace(data_path)
    print(
        f"Data capped: removed {trim_count} oldest positions"
        f" ({total_positions} -> {max_positions}, max={max_positions})"
    )


def run(cmd: str) -> subprocess.CompletedProcess:
    print(f"$ {cmd}")
    return subprocess.run(cmd, shell=True, check=True)


def run_compare(cmd: str) -> dict:
    print(f"$ {cmd}")
    proc = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    stdout = "".join(lines)

    if m := _WLD_RE.search(stdout):
        new_wins, old_wins, draws = (int(g) for g in m.groups())
    else:
        new_wins, old_wins, draws = 0, 0, 0

    return {
        "improved": proc.returncode == 0,
        "new_wins": new_wins,
        "old_wins": old_wins,
        "draws": draws,
    }


def read_current_best(pointer_file: Path) -> str | None:
    if not pointer_file.exists():
        return None
    return pointer_file.read_text().strip() or None


def write_current_best(pointer_file: Path, path: str) -> None:
    pointer_file.parent.mkdir(parents=True, exist_ok=True)
    pointer_file.write_text(Path(path).as_posix() + "\n")


def write_report(
    archive_path: Path,
    status: str,
    new_name: str,
    old_name: str,
    wld: dict,
) -> None:
    stem = archive_path.stem
    report_path = archive_path.with_suffix(".md")
    lines = [
        f"# {stem}",
        "",
        f"## {status}",
        "",
        "| Model | W | L | D |",
        "|-------|---|---|---|",
        f"| {new_name} (new) | {wld['new_wins']} | {wld['old_wins']} | {wld['draws']} |",
        f"| {old_name} (old) | {wld['old_wins']} | {wld['new_wins']} | {wld['draws']} |",
        "",
    ]
    report_path.write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=_inv["train_games"])
    p.add_argument("--depth", type=int, default=_sp["search_depth"])
    p.add_argument("--threads", type=int, default=_sp["num_threads"])
    p.add_argument("--data", default=_sp["output_file"])
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
    p.add_argument(
        "--candidate",
        help="Candidate weights path (required with --compare-only).",
    )
    p.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Max iterations (0 = infinite, default: 0).",
    )
    args = p.parse_args()

    pointer_file = Path(_dep["paths"]["current_best_file"])
    data_path = Path(args.data)
    models_dir = Path("models")
    accepted_dir = Path(_dep["paths"]["accepted_models_dir"])
    rejected_dir = Path(_dep["paths"]["rejected_models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    log_path = None

    iteration = 0
    while args.iterations == 0 or iteration < args.iterations:
        iteration += 1
        print(f"\n{'=' * 60}")
        print(f"=== Iteration {iteration} ===")
        print(f"{'=' * 60}")

        current_best = read_current_best(pointer_file)
        best_path = Path(current_best) if current_best else None

        if not args.compare_only and not args.train_only:
            # 1. Self-play using current best model (or handcrafted)
            eval_label = f"NNUE ({current_best})" if best_path else "handcrafted"
            print(f"\n--- Self-play ({args.games} games, eval: {eval_label}) ---")
            selfplay_cmd = (
                f"{BOT_EXE} --selfplay {args.games} {args.depth}"
                f" {args.data} {args.threads}"
            )
            if best_path:
                selfplay_cmd += f" {best_path}"
            run(selfplay_cmd)

        if not args.compare_only:
            cap_training_data(data_path)
            total_positions = data_path.stat().st_size // POSITION_BYTES
            print(f"Total accumulated positions: {total_positions}")

            # 2. Train NNUE from full accumulated data
            print(
                f"\n--- Train NNUE"
                f" ({args.epochs} epochs, {total_positions} positions) ---"
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            candidate_name = f"nnue_{timestamp}_{total_positions}pos.bin"
            log_path = Path(tempfile.gettempdir()) / candidate_name.replace(
                ".bin", "_train.md"
            )
            run(
                f"python -u engine/train/train_nnue.py --data {args.data}"
                f" --output engine/train/nnue_weights.pt"
                f" --epochs {args.epochs} --batch-size {args.batch_size}"
                f" --eval-weight {args.eval_weight}"
                f" --log {log_path}"
            )

            # 3. Export candidate with full descriptive name
            candidate_path = models_dir / candidate_name
            print(f"\n--- Export candidate ({candidate_name}) ---")
            run(
                f"python engine/train/export_nnue.py"
                f" --model engine/train/nnue_weights.pt"
                f" --output {candidate_path}"
            )
        else:
            # --compare-only: use the provided --candidate path
            if args.candidate:
                candidate_path = Path(args.candidate)
            else:
                print("Error: --compare-only requires --candidate")
                break
            candidate_name = candidate_path.name

        if not candidate_path.exists() or candidate_path.stat().st_size == 0:
            print(f"Error: candidate missing or empty at {candidate_path}")
            candidate_path.unlink(missing_ok=True)
            break

        # 4. Compare candidate vs current best
        old_name = Path(current_best).stem if current_best else "handcrafted"
        old_arg = str(best_path) if best_path else "handcrafted"
        print(
            f"\n--- Compare: {candidate_name} vs {old_name} ({args.compare_games} games) ---"
        )
        wld = run_compare(
            f"{BOT_EXE} --compare {old_arg} {candidate_path}"
            f" {args.compare_games} {data_path} {args.threads}"
        )
        improved = wld["improved"]

        # 5. Archive
        archive_name = candidate_name

        if improved:
            archive_path = accepted_dir / archive_name
            shutil.move(str(candidate_path), str(archive_path))
            write_current_best(pointer_file, str(archive_path))
            status = "ACCEPTED"
        else:
            archive_path = rejected_dir / archive_name
            shutil.move(str(candidate_path), str(archive_path))
            status = "REJECTED"

        if log_path and log_path.exists():
            shutil.move(str(log_path), str(archive_path.parent / log_path.name))

        write_report(archive_path, status, candidate_path.stem, old_name, wld)

        # 6. Summary
        current_best = read_current_best(pointer_file)
        best_label = current_best if current_best else "handcrafted"
        total_positions = (
            data_path.stat().st_size // POSITION_BYTES if data_path.exists() else 0
        )
        print(f"\n--- Iteration {iteration}: {status} ---")
        print(f"Archived to: {archive_path}")
        print(f"Total positions: {total_positions}")
        print(f"Current best: {best_label}")


if __name__ == "__main__":
    main()
