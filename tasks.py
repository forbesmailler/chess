import os
from pathlib import Path

from invoke import task

from config.load_config import deploy, engine, training

_eng = engine()
_trn = training()
_dep = deploy()

CPP_FILES = "engine/*.cpp engine/*.h tests/engine/*.cpp"
BOT_EXE = str(Path(_dep["paths"]["bot_exe"]))

_sp = _trn["self_play"]
_train_cfg = _trn["training"]
_cmp = _trn.get("compare", {})


@task
def gen_config(c):
    """Regenerate engine/generated_config.h from YAML config files."""
    c.run("python scripts/gen_config_header.py")


@task
def format(c):
    """Format all code (Python + C++)."""
    c.run("ruff format .")
    c.run("ruff check --fix --unsafe-fixes .")
    c.run(f"clang-format -i {CPP_FILES}")


@task
def test_cpp(c):
    """Run C++ tests via ctest."""
    with c.cd("engine/build"):
        c.run("ctest -C Release --output-on-failure")


@task
def test(c):
    """Run all tests (Python + C++)."""
    c.run("pytest")
    test_cpp(c)


@task
def build(c):
    """Build the C++ engine."""
    with c.cd("engine/build"):
        c.run("cmake .. -DCMAKE_BUILD_TYPE=Release")
        c.run("cmake --build . --config Release")


@task
def prepare(c):
    """Regenerate config, format, build, and test."""
    print("=== Step 1/4: Gen config ===")
    gen_config(c)

    print("=== Step 2/4: Format ===")
    format(c)

    print("=== Step 3/4: Build ===")
    build(c)

    print("=== Step 4/4: Test ===")
    test(c)


@task(
    help={
        "games": f"Number of self-play games per iteration (default: {_sp['num_games']})",
        "threads": f"Number of threads (default: {_sp['num_threads']})",
        "data": f"Training data path (default: {_sp['output_file']})",
        "epochs": f"Training epochs (default: {_train_cfg['epochs']})",
        "batch_size": f"Training batch size (default: {_train_cfg['batch_size']})",
        "eval_weight": f"Search eval vs game result blend (default: {_train_cfg['eval_weight']})",
        "compare_games": f"Comparison games (default: {_cmp.get('num_games', 100)})",
        "train_only": "Skip self-play; just train, export, compare, and archive",
        "compare_only": "Skip self-play and training; just compare candidate vs current best",
        "candidate": "Candidate weights path (required with --compare-only)",
        "freeze_baseline": "Don't update current best on accept",
    }
)
def train(
    c,
    games=_sp["num_games"],
    threads=_sp["num_threads"],
    data=_sp["output_file"],
    epochs=_train_cfg["epochs"],
    batch_size=_train_cfg["batch_size"],
    eval_weight=_train_cfg["eval_weight"],
    compare_games=_cmp.get("num_games", 100),
    train_only=False,
    compare_only=False,
    candidate=None,
    freeze_baseline=False,
):
    """Prepare then run continuous RL loop (Ctrl+C to stop)."""
    prepare(c)
    cmd = (
        f"python -u scripts/train_loop.py"
        f" --games {games} --threads {threads}"
        f" --data {data}"
        f" --epochs {epochs} --batch-size {batch_size}"
        f" --eval-weight {eval_weight} --compare-games {compare_games}"
    )
    if train_only:
        cmd += " --train-only"
    if compare_only:
        cmd += " --compare-only"
    if candidate:
        cmd += f" --candidate {candidate}"
    if freeze_baseline:
        cmd += " --freeze-baseline"
    c.run(cmd)


@task(
    help={
        "eval_weights": "Comma-separated eval weights to sweep (e.g., 0.5,0.6,0.75,0.9)",
        "data": f"Training data path (default: {_sp['output_file']})",
        "epochs": f"Training epochs (default: {_train_cfg['epochs']})",
        "batch_size": f"Training batch size (default: {_train_cfg['batch_size']})",
        "compare_games": f"Comparison games (default: {_cmp.get('num_games', 100)})",
        "freeze_baseline": "Don't update current best on accept (default: true)",
    }
)
def sweep(
    c,
    eval_weights,
    data=_sp["output_file"],
    epochs=_train_cfg["epochs"],
    batch_size=_train_cfg["batch_size"],
    compare_games=_cmp.get("num_games", 100),
    freeze_baseline=True,
):
    """Train and compare one model per eval weight."""
    import tempfile

    prepare(c)
    tmp = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    tmp.close()
    tmp_path = tmp.name
    try:
        weights = [float(w.strip()) for w in eval_weights.split(",")]
        for i, w in enumerate(weights, 1):
            print(f"\n{'=' * 60}")
            print(f"=== Sweep {i}/{len(weights)}: eval_weight={w} ===")
            print(f"{'=' * 60}")
            cmd = (
                f"python -u scripts/train_loop.py --train-only"
                f" --data {data}"
                f" --epochs {epochs} --batch-size {batch_size}"
                f" --eval-weight {w} --compare-games {compare_games}"
                f" --compare-data {tmp_path}"
            )
            if freeze_baseline:
                cmd += " --freeze-baseline"
            c.run(cmd)

        # Append comparison positions to training data
        tmp_file = Path(tmp_path)
        if tmp_file.exists() and tmp_file.stat().st_size > 0:
            with open(data, "ab") as dst, open(tmp_path, "rb") as src:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
            positions = tmp_file.stat().st_size // 42
            print(f"\nAppended {positions} comparison positions to {data}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@task(
    help={
        "pgn": "Path to PGN file (.pgn or .pgn.zst)",
        "output": "Output book file (default: book.bin)",
        "min_elo": "Min player Elo filter (default: 2200)",
        "min_time": "Min initial time in seconds (default: 180)",
        "min_count": "Min games per position-move (default: 5)",
        "min_weight_pct": "Min weight fraction per position (default: 0.01)",
    }
)
def build_book(
    c,
    pgn,
    output="book.bin",
    min_elo=1800,
    min_time=180,
    min_count=10,
    min_weight_pct=0.01,
    workers=0,
):
    """Build opening book from a PGN file."""
    cmd = (
        f"python scripts/build_opening_book.py {pgn}"
        f" --output {output}"
        f" --min-elo {min_elo} --min-time {min_time}"
        f" --min-count {min_count} --min-weight-pct {min_weight_pct}"
    )
    if workers > 0:
        cmd += f" --workers {workers}"
    c.run(cmd)


@task(
    help={
        "username": "Lichess bot username to challenge",
        "time": "Clock time in seconds (default: 300)",
        "increment": "Clock increment in seconds (default: 0)",
        "casual": "Make it a casual (unrated) game",
    }
)
def challenge(c, username, time=300, increment=0, casual=False):
    """Challenge a Lichess bot (the deployed service handles the game)."""
    # Load .env file if present
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

    rated = "casual" if casual else "rated"
    c.run(f"{BOT_EXE} --challenge {username} {time} {increment} {rated}")


@task(
    help={
        "training_data": "Training data output file (default: training_data.bin)",
    }
)
def play_bots(c, training_data="training_data.bin"):
    """Prepare, then launch bot and challenge online Lichess bots (Ctrl+C to stop)."""
    prepare(c)
    c.run(f"python -u scripts/play_bots.py --training-data {training_data}")


@task(
    help={
        "elo": "Stockfish Elo limit (e.g. 2600)",
        "stockfish": "Path to Stockfish executable (default: stockfish)",
        "games": f"Games per training cycle (default: {_sp['num_games']})",
        "move_time": f"Think time per move in ms (default: {_sp['search_time_ms']})",
        "data": f"Training data path (default: {_sp['output_file']})",
        "no_retrain": "Just play and collect data, don't retrain",
    }
)
def play_stockfish(
    c,
    elo,
    stockfish="stockfish",
    games=_sp["num_games"],
    move_time=_sp["search_time_ms"],
    data=_sp["output_file"],
    no_retrain=False,
):
    """Prepare, then play against Stockfish at given Elo, collect data, and retrain."""
    prepare(c)
    cmd = (
        f"python -u scripts/play_stockfish.py {elo}"
        f" --stockfish {stockfish}"
        f" --games {games} --move-time {move_time}"
        f" --data {data}"
    )
    if no_retrain:
        cmd += " --no-retrain"
    c.run(cmd)


@task(
    help={
        "time": "Clock time in seconds (default: 60)",
        "increment": "Clock increment in seconds (default: 0)",
        "rating_range": "Max rating difference from bot (default: 2000)",
        "training_data": "Training data output file (default: training_data.bin)",
    }
)
def play_humans(
    c, time=60, increment=0, rating_range=2000, training_data="training_data.bin"
):
    """Prepare, then launch bot and challenge active humans within rating range (Ctrl+C to stop)."""
    prepare(c)
    c.run(
        f"python -u scripts/play_humans.py --time {time} --increment {increment} --rating-range {rating_range} --training-data {training_data}"
    )


_vps = _dep["vps"]
_pointer_file = _dep["paths"]["current_best_file"]


@task(
    help={
        "weights": "NNUE weights file path (default: read from pointer file)",
    }
)
def deploy(c, weights=None):
    """Deploy the bot on a Linux VPS: pull, build, test, install, restart service."""
    repo_dir = _vps["repo_dir"]
    install_dir = _vps["install_dir"]
    service = _vps["service_name"]
    pointer = f"{repo_dir}/{_pointer_file}"

    print("=== Step 1/5: Pull latest code ===")
    with c.cd(repo_dir):
        c.run("git pull")

    print("=== Step 2/5: Build ===")
    gen_config(c)
    format(c)
    build(c)

    print("=== Step 3/5: Test ===")
    test_cpp(c)

    print("=== Step 4/5: Install ===")
    c.run("sudo -v", pty=True)
    c.run(f"sudo systemctl stop {service}", warn=True)
    c.run(f"sudo mkdir -p {install_dir}")
    c.run(f"sudo cp {repo_dir}/engine/build/lichess_bot {install_dir}/")
    if weights:
        c.run(f"sudo cp {weights} {install_dir}/nnue.bin")
    else:
        c.run(
            f"test -f {pointer}"
            f' && sudo cp "{repo_dir}/$(cat {pointer})" {install_dir}/nnue.bin',
            warn=True,
        )
    book_src = f"{repo_dir}/book.bin"
    c.run(f"test -f {book_src} && sudo cp {book_src} {install_dir}/", warn=True)
    c.run(f"sudo cp {repo_dir}/{_vps['service_file']} {_vps['systemd_path']}")

    print("=== Step 5/5: Restart service ===")
    c.run("sudo systemctl daemon-reload")
    c.run(f"sudo systemctl enable {service}")
    c.run(f"sudo systemctl restart {service}")
    c.run(f"systemctl status {service}")
