# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Lichess chess bot with a C++ engine and Python training/experimental components. The bot supports three evaluation modes: handcrafted (tapered eval), logistic regression, and NNUE (neural network). Search is via negamax (alpha-beta) or MCTS. Includes a self-play data generator and NNUE training pipeline.

## Build & Run

### C++ Engine (primary)

```powershell
# Windows (requires vcpkg with curl, nlohmann-json installed)
cd engine
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build . --config Release

# Run unit tests
ctest -C Release --output-on-failure

# Run bot
.\Release\lichess_bot.exe <lichess_token> [max_time_ms] [--engine=negamax|mcts] [--eval=handcrafted|logistic|nnue] [--nnue-weights=path]

# Self-play data generation
.\Release\lichess_bot.exe --selfplay [num_games] [search_depth] [output_file] [num_threads]
```

```bash
# Linux/macOS
cd engine && mkdir build && cd build
cmake .. && cmake --build . --config Release
ctest --output-on-failure                        # run unit tests
./lichess_bot <token> 1000 --engine=mcts --eval=handcrafted  # run bot
./lichess_bot --selfplay 1000 6 training_data.bin 8          # generate training data
```

### NNUE Training Pipeline

```bash
# 1. Generate self-play data (C++)
./lichess_bot --selfplay 1000 6 training_data.bin 8

# 2. Train NNUE (Python/PyTorch)
cd engine/train
python train_nnue.py --data ../../training_data.bin --output nnue_weights.pt --epochs 100

# 3. Export to binary for C++ inference
python export_nnue.py --model nnue_weights.pt --output nnue.bin

# 4. Run bot with NNUE eval
./lichess_bot <token> 1000 --eval=nnue --nnue-weights=nnue.bin
```

### Python Training (logistic regression)

```bash
cd engine/train
python process_pgn.py      # generates train.csv, val.csv
python train_logistic_model.py  # produces chess_lr.joblib
python export_model.py     # creates model_coefficients.txt
```

### Python Bot (legacy)

```bash
cd experimental
python lichess.py --token <lichess_token>
```

### Development Tasks

```bash
pip install -e ".[dev]"   # install with dev dependencies

invoke format             # format all (Python + C++)
invoke format-py          # format Python with ruff
invoke format-cpp         # format C++ with clang-format

invoke test               # run all tests
invoke test-py            # run pytest
invoke test-cpp           # run C++ unit tests (gtest/ctest)

invoke build-cpp          # build C++ engine
```

## Architecture

### C++ Engine (`engine/`)

- `base_engine.h` – Abstract `BaseEngine` interface with `evaluate()`, `get_best_move()`, and `EvalMode` enum (HANDCRAFTED, LOGISTIC, NNUE)
- `chess_engine.h/cpp` – Negamax with alpha-beta, transposition tables, quiescence search, iterative deepening
- `mcts_engine.h/cpp` – Monte Carlo Tree Search with UCT selection
- `handcrafted_eval.h/cpp` – Tapered eval with material, PSTs, pawn structure, mobility, king safety
- `nnue_model.h/cpp` – NNUE inference: loads binary weights, 768->256->32->3 architecture with ClippedReLU
- `self_play.h/cpp` – Multi-threaded self-play data generator with binary output format
- `feature_extractor.h/cpp` – 1542-dim features for logistic model
- `logistic_model.h/cpp` – Loads exported coefficients from `model_coefficients.txt`
- `lichess_client.h/cpp` – HTTP streaming to Lichess API via libcurl
- `main.cpp` – `LichessBot` orchestrating games; `--selfplay` mode for data generation

### Python Components

- `engine/train/train_nnue.py` – PyTorch NNUE training from self-play binary data
- `engine/train/export_nnue.py` – Export PyTorch model to binary format for C++ loading
- `engine/train/` – Logistic regression training (process_pgn.py, train_logistic_model.py)
- `experimental/self_play/` – PyTorch MCTS self-play training (`ChessNet`, `MCTS`)
- `experimental/lichess.py` – Python bot using berserk library

### Evaluation Modes

- **Handcrafted**: Tapered eval blending middlegame/endgame scores. Material, piece-square tables, pawn structure (passed/isolated/doubled), rook on open files, bishop pair, mobility, king pawn shield.
- **Logistic**: 1542-dim features → sklearn LogisticRegression → P(win)/P(draw)/P(loss).
- **NNUE**: 768 piece-square features (side-to-move perspective) → 256 → 32 → 3 with ClippedReLU. Outputs P(win)/P(draw)/P(loss), eval = (P(win) - P(loss)) × MATE_VALUE.

### NNUE Feature Extraction

768 features from side-to-move perspective:
- 0-383: own pieces (6 types × 64 squares, one-hot)
- 384-767: opponent pieces (6 types × 64 squares, one-hot)
- When black to move, board is vertically flipped and colors swapped

### Self-Play Binary Format

42 bytes per position:
- 32 bytes: piece placement (64 nibbles, 4 bits per square)
- 1 byte: side to move (0=white, 1=black)
- 1 byte: castling rights (4 bits)
- 1 byte: en passant file (0-7 or 255)
- 4 bytes: search eval (float32, from STM perspective)
- 1 byte: game result (0=loss, 1=draw, 2=win from STM perspective)
- 2 bytes: ply number (uint16)

### Deployment

- `deploy/chess-bot.service` – systemd unit file for Linux VPS deployment
- `deploy/README.md` – deployment instructions

## Dependencies

- **C++**: CMake 3.16+, C++17, libcurl, nlohmann-json, chess-library (fetched via CMake)
- **Python**: chess, numpy, pandas, scikit-learn, joblib, torch (for NNUE training), berserk (for Python bot)
