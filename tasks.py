import os

from invoke import task

CPP_FILES = "engine/*.cpp engine/*.h"
BOT_EXE = os.path.join("engine", "build", "Release", "lichess_bot.exe")


@task
def format(c):
    """Format all code (Python + C++)."""
    format_py(c)
    format_cpp(c)


@task
def format_py(c):
    """Format Python code with ruff."""
    c.run("ruff format .")
    c.run("ruff check --fix --unsafe-fixes .")


@task
def format_cpp(c):
    """Format C++ code with clang-format."""
    c.run(f"clang-format -i {CPP_FILES}")


@task
def test(c):
    """Run all tests (Python + C++)."""
    c.run("pytest", warn=True)
    with c.cd("engine/build"):
        c.run("ctest -C Release --output-on-failure")


@task
def test_py(c):
    """Run pytest."""
    c.run("pytest")


@task
def test_cpp(c):
    """Run C++ unit tests with ctest."""
    with c.cd("engine/build"):
        c.run("ctest -C Release --output-on-failure")


@task
def build_cpp(c):
    """Build C++ engine."""
    with c.cd("engine/build"):
        c.run("cmake --build . --config Release")


@task(
    help={
        "games": "Number of self-play games (default: 1000)",
        "depth": "Search time budget (default: 6)",
        "threads": "Number of threads (default: 16)",
        "data": "Training data output path (default: training_data.bin)",
        "weights": "NNUE weights output path (default: nnue.bin)",
        "epochs": "Training epochs (default: 100)",
        "batch_size": "Training batch size (default: 4096)",
        "eval_weight": "Search eval vs game result blend (default: 0.75)",
    }
)
def train(
    c,
    games=1000,
    depth=6,
    threads=16,
    data="training_data.bin",
    weights="nnue.bin",
    epochs=100,
    batch_size=4096,
    eval_weight=0.75,
):
    """Generate self-play data, train NNUE, and export weights."""
    bot_exe = BOT_EXE

    print(f"=== Step 1/3: Self-play ({games} games, {threads} threads) ===")
    c.run(f"{bot_exe} --selfplay {games} {depth} {data} {threads}")

    print(f"=== Step 2/3: Train NNUE ({epochs} epochs) ===")
    with c.cd("engine/train"):
        c.run(
            f"python train_nnue.py --data ../../{data} --output nnue_weights.pt "
            f"--epochs {epochs} --batch-size {batch_size} --eval-weight {eval_weight}"
        )

    print(f"=== Step 3/3: Export NNUE to {weights} ===")
    with c.cd("engine/train"):
        c.run(f"python export_nnue.py --model nnue_weights.pt --output ../../{weights}")

    print(f"Done. NNUE weights saved to {weights}")


@task(
    help={
        "token": "Lichess API token",
        "time": "Max search time in ms (default: 1000)",
        "engine": "Search algorithm: negamax or mcts (default: negamax)",
        "weights": "NNUE weights path (default: nnue.bin)",
    }
)
def run(c, token, time=1000, engine="negamax", weights="nnue.bin"):
    """Run the Lichess bot with NNUE evaluation."""
    bot_exe = BOT_EXE
    c.run(
        f"{bot_exe} {token} {time} --engine={engine} "
        f"--eval=nnue --nnue-weights={weights}"
    )


@task(
    help={
        "token": "Lichess API token",
        "time": "Max search time in ms (default: 1000)",
        "engine": "Search algorithm: negamax or mcts (default: negamax)",
        "weights": "NNUE weights path (default: nnue.bin)",
    }
)
def deploy(c, token, time=1000, engine="negamax", weights="nnue.bin"):
    """Format, test, build, and run the bot."""
    print("=== Step 1/4: Format ===")
    format(c)

    print("=== Step 2/4: Test ===")
    test(c)

    print("=== Step 3/4: Build ===")
    build_cpp(c)

    print("=== Step 4/4: Run ===")
    run(c, token=token, time=time, engine=engine, weights=weights)
