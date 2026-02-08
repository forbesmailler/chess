from glob import glob
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
_inv = _trn["invoke"]
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
    files = []
    for pattern in CPP_FILES.split():
        files.extend(glob(pattern))
    print(f"Formatting {len(files)} C++ files...")
    c.run(f"clang-format -i {CPP_FILES}")


@task
def test_cpp(c):
    """Run C++ tests via ctest."""
    with c.cd("engine/build"):
        c.run("ctest -C Release --output-on-failure")


@task
def test(c):
    """Run all tests (Python + C++)."""
    c.run("pytest", warn=True)
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
        "games": f"Number of self-play games per iteration (default: {_inv['train_games']})",
        "depth": f"Search depth (default: {_sp['search_depth']})",
        "threads": f"Number of threads (default: {_inv['train_threads']})",
        "data": f"Training data path (default: {_sp['output_file']})",
        "weights": f"Current best NNUE weights path (default: {_inv['weights']})",
        "epochs": f"Training epochs (default: {_train_cfg['epochs']})",
        "batch_size": f"Training batch size (default: {_train_cfg['batch_size']})",
        "eval_weight": f"Search eval vs game result blend (default: {_train_cfg['eval_weight']})",
        "compare_games": f"Comparison games (default: {_cmp.get('num_games', 1000)})",
    }
)
def train(
    c,
    games=_inv["train_games"],
    depth=_sp["search_depth"],
    threads=_inv["train_threads"],
    data=_sp["output_file"],
    weights=_inv["weights"],
    epochs=_train_cfg["epochs"],
    batch_size=_train_cfg["batch_size"],
    eval_weight=_train_cfg["eval_weight"],
    compare_games=_cmp.get("num_games", 1000),
):
    """Prepare then run continuous RL loop (Ctrl+C to stop)."""
    prepare(c)
    c.run(
        f"python scripts/train_loop.py"
        f" --games {games} --depth {depth} --threads {threads}"
        f" --data {data} --weights {weights}"
        f" --epochs {epochs} --batch-size {batch_size}"
        f" --eval-weight {eval_weight} --compare-games {compare_games}"
    )


_vps = _dep["vps"]


@task(
    help={
        "weights": f"NNUE weights file name (default: {_inv['weights']})",
    }
)
def deploy(c, weights=_inv["weights"]):
    """Deploy the bot on a Linux VPS: pull, build, test, install, restart service."""
    repo_dir = _vps["repo_dir"]
    install_dir = _vps["install_dir"]
    service = _vps["service_name"]

    print("=== Step 1/5: Pull latest code ===")
    with c.cd(repo_dir):
        c.run("git pull")

    print("=== Step 2/5: Build ===")
    build(c)

    print("=== Step 3/5: Test ===")
    test_cpp(c)

    print("=== Step 4/5: Install ===")
    c.run(f"systemctl stop {service}", warn=True)
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
