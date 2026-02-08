# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Lichess chess bot: C++ engine with Python training pipeline. Three eval modes (handcrafted, logistic, NNUE), two search algorithms (negamax alpha-beta, MCTS), self-play data generation, and NNUE training.

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
invoke build-cpp          # build engine
invoke test               # all tests (Python + C++)
invoke test-cpp           # C++ tests only
invoke test-py            # Python tests only
invoke format             # format all (ruff + clang-format)
invoke train              # self-play → train NNUE → export weights
invoke run              # run bot with NNUE (reads LICHESS_TOKEN env var)
invoke deploy             # deploy to Linux VPS (reads LICHESS_TOKEN env var)
```

## Architecture

### C++ Engine (`engine/`)

| File | Purpose |
|------|---------|
| `base_engine.h` | Abstract `BaseEngine` interface: `evaluate()`, `get_best_move()`, `EvalMode` enum |
| `chess_engine.h/cpp` | Negamax with alpha-beta, transposition tables, quiescence search, iterative deepening |
| `mcts_engine.h/cpp` | Monte Carlo Tree Search with UCT selection |
| `handcrafted_eval.h/cpp` | Tapered eval: material, PSTs, pawn structure, mobility, king safety |
| `nnue_model.h/cpp` | NNUE inference: binary weights, 768→256→32→3 with ClippedReLU |
| `feature_extractor.h/cpp` | 1542-dim feature vector for logistic model |
| `logistic_model.h/cpp` | Loads exported coefficients from `model_coefficients.txt` |
| `self_play.h/cpp` | Multi-threaded self-play data generator (binary output) |
| `chess_board.h/cpp` | Board wrapper utilities |
| `utils.h/cpp` | Shared helper functions |
| `lichess_client.h/cpp` | HTTP streaming to Lichess API via libcurl |
| `main.cpp` | `LichessBot` game loop; `--selfplay` mode |

### Python Components

| File | Purpose |
|------|---------|
| `engine/train/train_nnue.py` | PyTorch NNUE training from self-play binary data |
| `engine/train/export_nnue.py` | Export PyTorch model → binary for C++ loading |
| `engine/train/process_pgn.py` | PGN → feature CSVs for logistic regression |
| `engine/train/train_logistic_model.py` | Logistic regression training |
| `engine/train/export_model.py` | Export logistic coefficients for C++ loading |
| `experimental/lichess.py` | Legacy Python bot (berserk library) |
| `experimental/self_play/` | PyTorch MCTS self-play training (`ChessNet`, `MCTS`) |

### Tests

- C++: `engine/tests/test_*.cpp` (GTest, run via ctest)
- Python: `tests/` (pytest)

## Evaluation Modes

- **Handcrafted**: Tapered middlegame/endgame blend. Material, PSTs, pawn structure (passed/isolated/doubled), rook on open files, bishop pair, mobility, king pawn shield.
- **Logistic**: 1542-dim features → P(win)/P(draw)/P(loss).
- **NNUE**: 768 piece-square features (STM perspective) → 256 → 32 → 3 with ClippedReLU. eval = (P(win) - P(loss)) × MATE_VALUE.

## NNUE Feature Encoding

768 features from side-to-move perspective:
- 0–383: own pieces (6 types × 64 squares, one-hot)
- 384–767: opponent pieces (6 types × 64 squares, one-hot)
- When black to move: board vertically flipped, colors swapped

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
- **Python**: see `pyproject.toml` — chess, numpy, pandas, scikit-learn, joblib, tqdm, berserk; optional: torch (NNUE training), invoke/pytest/ruff (dev)
