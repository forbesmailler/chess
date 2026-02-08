from glob import glob

from invoke import task

from config.load_config import deploy, engine, training

_eng = engine()
_trn = training()
_dep = deploy()

CPP_FILES = "engine/*.cpp engine/*.h tests/engine/*.cpp"
BOT_EXE = _dep["paths"]["bot_exe"]

_sp = _trn["self_play"]
_train_cfg = _trn["training"]
_inv = _trn["invoke"]


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
def test(c):
    """Run all tests (Python + C++)."""
    c.run("pytest", warn=True)
    with c.cd("engine/build"):
        c.run("ctest -C Release --output-on-failure")


@task
def prepare(c):
    """Regenerate config, format, and test."""
    print("=== Step 1/3: Gen config ===")
    gen_config(c)

    print("=== Step 2/3: Format ===")
    format(c)

    print("=== Step 3/3: Test ===")
    test(c)


@task(
    help={
        "games": f"Number of self-play games (default: {_inv['train_games']})",
        "depth": f"Search depth (default: {_sp['search_depth']})",
        "threads": f"Number of threads (default: {_inv['train_threads']})",
        "data": f"Training data output path (default: {_sp['output_file']})",
        "weights": f"NNUE weights output path (default: {_inv['weights']})",
        "epochs": f"Training epochs (default: {_train_cfg['epochs']})",
        "batch_size": f"Training batch size (default: {_train_cfg['batch_size']})",
        "eval_weight": f"Search eval vs game result blend (default: {_train_cfg['eval_weight']})",
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
):
    """Prepare, generate self-play data, train NNUE, and export weights."""
    print("=== Step 1/4: Prepare ===")
    prepare(c)

    print(f"=== Step 2/4: Self-play ({games} games, {threads} threads) ===")
    c.run(f"{BOT_EXE} --selfplay {games} {depth} {data} {threads}")

    print(f"=== Step 3/4: Train NNUE ({epochs} epochs) ===")
    with c.cd("engine/train"):
        c.run(
            f"python train_nnue.py --data ../../{data} --output nnue_weights.pt "
            f"--epochs {epochs} --batch-size {batch_size} --eval-weight {eval_weight}"
        )

    print(f"=== Step 4/4: Export NNUE to {weights} ===")
    with c.cd("engine/train"):
        c.run(f"python export_nnue.py --model nnue_weights.pt --output ../../{weights}")

    print(f"Done. NNUE weights saved to {weights}")


_vps = _dep["vps"]


@task(
    help={
        "weights": f"NNUE weights file name (default: {_inv['weights']})",
    }
)
def deploy(c, weights=_inv["weights"]):
    """Deploy the bot on a Linux VPS: pull, test, build, install, restart service."""
    repo_dir = _vps["repo_dir"]
    install_dir = _vps["install_dir"]
    service = _vps["service_name"]

    print("=== Step 1/5: Pull latest code ===")
    with c.cd(repo_dir):
        c.run("git pull")

    print("=== Step 2/5: Test ===")
    test(c)

    print("=== Step 3/5: Build ===")
    with c.cd(f"{repo_dir}/engine/build"):
        c.run("cmake .. -DCMAKE_BUILD_TYPE=Release")
        c.run("cmake --build . --config Release")

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
