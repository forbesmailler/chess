from invoke import task

from config.load_config import deploy, training

_trn = training()
_dep = deploy()

CPP_FILES = "engine/*.cpp engine/*.h engine/tests/*.cpp"
BOT_EXE = _dep["paths"]["bot_exe"]

_train_defaults = _trn["invoke_defaults"]["train"]
_run_defaults = _trn["invoke_defaults"]["run"]


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
    from glob import glob

    files = []
    for pattern in CPP_FILES.split():
        files.extend(glob(pattern))
    print(f"Formatting {len(files)} C++ files...")
    c.run(f"clang-format -i {CPP_FILES}")
    print("Done.")


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


@task
def gen_config(c):
    """Regenerate engine/generated_config.h from YAML config files."""
    c.run("python scripts/gen_config_header.py")


@task(
    help={
        "games": f"Number of self-play games (default: {_train_defaults['games']})",
        "depth": f"Search time budget (default: {_train_defaults['depth']})",
        "threads": f"Number of threads (default: {_train_defaults['threads']})",
        "data": f"Training data output path (default: {_train_defaults['data']})",
        "weights": f"NNUE weights output path (default: {_train_defaults['weights']})",
        "epochs": f"Training epochs (default: {_train_defaults['epochs']})",
        "batch_size": f"Training batch size (default: {_train_defaults['batch_size']})",
        "eval_weight": f"Search eval vs game result blend (default: {_train_defaults['eval_weight']})",
    }
)
def train(
    c,
    games=_train_defaults["games"],
    depth=_train_defaults["depth"],
    threads=_train_defaults["threads"],
    data=_train_defaults["data"],
    weights=_train_defaults["weights"],
    epochs=_train_defaults["epochs"],
    batch_size=_train_defaults["batch_size"],
    eval_weight=_train_defaults["eval_weight"],
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
        "time": f"Max search time in ms (default: {_run_defaults['time']})",
        "engine": f"Search algorithm: negamax or mcts (default: {_run_defaults['engine']})",
        "weights": f"NNUE weights path (default: {_run_defaults['weights']})",
    }
)
def run(
    c,
    time=_run_defaults["time"],
    engine=_run_defaults["engine"],
    weights=_run_defaults["weights"],
):
    """Run the Lichess bot with NNUE evaluation. Reads LICHESS_TOKEN env var."""
    bot_exe = BOT_EXE
    c.run(f"{bot_exe} {time} --engine={engine} --eval=nnue --nnue-weights={weights}")


@task(
    help={
        "time": f"Max search time in ms (default: {_run_defaults['time']})",
        "engine": f"Search algorithm: negamax or mcts (default: {_run_defaults['engine']})",
        "weights": f"NNUE weights path (default: {_run_defaults['weights']})",
    }
)
def deploy_local(
    c,
    time=_run_defaults["time"],
    engine=_run_defaults["engine"],
    weights=_run_defaults["weights"],
):
    """Format, test, build, and run the bot locally. Reads LICHESS_TOKEN env var."""
    print("=== Step 1/4: Format ===")
    format(c)

    print("=== Step 2/4: Test ===")
    test(c)

    print("=== Step 3/4: Build ===")
    build_cpp(c)

    print("=== Step 4/4: Run ===")
    run(c, time=time, engine=engine, weights=weights)


_vps = _dep["vps"]


@task(
    help={
        "time": f"Max search time in ms (default: {_run_defaults['time']})",
        "engine": f"Search algorithm: negamax or mcts (default: {_run_defaults['engine']})",
        "weights": f"NNUE weights file name (default: {_run_defaults['weights']})",
    }
)
def deploy(
    c,
    time=_run_defaults["time"],
    engine=_run_defaults["engine"],
    weights=_run_defaults["weights"],
):
    """Deploy the bot on a Linux VPS: pull, build, install, restart service."""
    repo_dir = _vps["repo_dir"]
    install_dir = _vps["install_dir"]
    service = _vps["service_name"]

    print("=== Step 1/5: Pull latest code ===")
    with c.cd(repo_dir):
        c.run("git pull")

    print("=== Step 2/5: Build ===")
    with c.cd(f"{repo_dir}/engine/build"):
        c.run("cmake .. -DCMAKE_BUILD_TYPE=Release")
        c.run("cmake --build . --config Release")

    print("=== Step 3/5: Run tests ===")
    with c.cd(f"{repo_dir}/engine/build"):
        c.run("ctest --output-on-failure")

    print("=== Step 4/5: Install ===")
    c.run(f"mkdir -p {install_dir}")
    c.run(f"cp {repo_dir}/engine/build/lichess_bot {install_dir}/")
    if weights:
        c.run(f"cp {repo_dir}/{weights} {install_dir}/", warn=True)
    c.run(f"cp {repo_dir}/{_vps['service_file']} {_vps['systemd_path']}")

    print("=== Step 5/5: Restart service ===")
    c.run("systemctl daemon-reload")
    c.run(f"systemctl enable {service}")
    c.run(f"systemctl restart {service}")
    c.run(f"systemctl status {service}")
