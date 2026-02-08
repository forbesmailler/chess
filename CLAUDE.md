# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Lichess chess bot: C++ engine with Python training pipeline. Two eval modes (handcrafted, NNUE), two search algorithms (negamax alpha-beta, MCTS), self-play data generation, and NNUE training.

## Build & Test

```powershell
# Windows (requires vcpkg with curl, nlohmann-json)
cd engine/build
cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build . --config Release
ctest -C Release --output-on-failure
```

```bash
# Linux/macOS
cd engine && mkdir -p build && cd build
cmake .. && cmake --build . --config Release
ctest --output-on-failure
```

### Invoke Tasks

```bash
invoke gen-config         # regenerate engine/generated_config.h from YAML
invoke format             # format all (ruff + clang-format)
invoke test               # all tests (Python + C++)
invoke prepare            # gen-config → format → build → test
invoke train              # prepare → continuous RL loop (self-play → train → export → compare)
invoke deploy             # pull → build → test → install → restart service on VPS
```

## Architecture

### C++ Engine (`engine/`)

| File | Purpose |
|------|---------|
| `base_engine.h` | Abstract `BaseEngine` interface: `evaluate()`, `get_best_move()`, `EvalMode` enum |
| `chess_engine.h/cpp` | Negamax with alpha-beta, transposition tables, quiescence search, iterative deepening |
| `mcts_engine.h/cpp` | Monte Carlo Tree Search with UCT selection |
| `handcrafted_eval.h/cpp` | Tapered eval: material, PSTs, pawn structure, mobility, king safety |
| `nnue_model.h/cpp` | NNUE inference: binary weights, 773→256→32→1 with ClippedReLU and tanh output |
| `self_play.h/cpp` | Multi-threaded self-play data generator and model comparator (binary output) |
| `chess_board.h/cpp` | Board wrapper utilities |
| `utils.h/cpp` | Shared helper functions |
| `lichess_client.h/cpp` | HTTP streaming to Lichess API via libcurl |
| `main.cpp` | `LichessBot` game loop; `--selfplay` and `--compare` modes |
| `generated_config.h` | Auto-generated constants from YAML (run `invoke gen-config`) |

### Config (`config/`)

| File | Purpose |
|------|---------|
| `engine.yaml` | Search, MCTS, NNUE architecture, bot/curl |
| `eval.yaml` | Material, PSTs, pawn/rook/bishop/king bonuses |
| `training.yaml` | Self-play defaults, training hyperparams, invoke task defaults, compare |
| `deploy.yaml` | File paths, VPS paths, service config |
| `load_config.py` | Python YAML loader with caching |

**Config change rule**: When adding or modifying values in `config/*.yaml` that the C++ engine needs at compile time, update `scripts/gen_config_header.py` to emit them into `engine/generated_config.h`. Values only used by Python (`tasks.py`, training scripts) do not need header generation.

### Python Components

| File | Purpose |
|------|---------|
| `engine/train/train_nnue.py` | PyTorch NNUE training from self-play binary data |
| `engine/train/export_nnue.py` | Export PyTorch model → binary for C++ loading |
| `scripts/gen_config_header.py` | Generates `engine/generated_config.h` from YAML |
| `scripts/train_loop.py` | Continuous RL loop: self-play → train → export → compare → archive |

### Tests

- C++: `tests/engine/test_*.cpp` (GTest, run via ctest)
- Python: `tests/{config,scripts,train}/test_*.py` (pytest)

## Evaluation Modes

- **Handcrafted**: Tapered middlegame/endgame blend. Material, PSTs, pawn structure (passed/isolated/doubled), rook on open files, bishop pair, mobility, king pawn shield.
- **NNUE**: 773 features (STM perspective) → 256 → 32 → 1 with ClippedReLU and tanh output. eval = tanh(logit) × MATE_VALUE.

## NNUE Feature Encoding

773 features from side-to-move perspective:
- 0–383: own pieces (6 types × 64 squares, one-hot)
- 384–767: opponent pieces (6 types × 64 squares, one-hot)
- 768–771: castling rights (own KS, own QS, opponent KS, opponent QS)
- 772: en passant available
- When black to move: board vertically flipped, colors swapped, castling reoriented

## Self-Play Binary Format

42 bytes per position:

| Offset | Size | Field |
|--------|------|-------|
| 0 | 32 | Piece placement (64 nibbles, 4 bits/square) |
| 32 | 1 | Side to move (0=white, 1=black) |
| 33 | 1 | Castling rights (4 bits) |
| 34 | 1 | En passant file (0–7 or 255=none) |
| 35 | 4 | Search eval (float32, STM perspective) |
| 39 | 1 | Game result (0=loss, 1=draw, 2=win, STM perspective) |
| 40 | 2 | Ply number (uint16) |

## Dependencies

- **C++**: CMake 3.16+, C++17, libcurl, nlohmann-json, [chess-library](https://github.com/Disservin/chess-library) (FetchContent)
- **Python**: see `pyproject.toml` — chess, numpy, tqdm; optional: torch (NNUE training), invoke/pytest/ruff (dev)
