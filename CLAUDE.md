# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Lichess chess bot with a C++ engine and Python training/experimental components. The bot uses a logistic regression model trained on historical game data, with search via either negamax (alpha-beta) or MCTS.

## Build & Run

### C++ Engine (primary)

```powershell
# Windows (requires vcpkg with curl, nlohmann-json installed)
cd engine
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build . --config Release

# Run tests
.\Release\lichess_bot.exe

# Run bot
.\Release\lichess_bot.exe <lichess_token> [max_time_ms] [--engine=negamax|mcts]
```

```bash
# Linux/macOS
cd engine && mkdir build && cd build
cmake .. && cmake --build . --config Release
./lichess_bot                                    # test mode
./lichess_bot <token> 1000 --engine=mcts         # bot mode
```

### Python Training

```bash
# Train logistic regression from PGN data
cd engine/train
python process_pgn.py      # generates train.csv, val.csv
python train_logistic_model.py  # produces chess_lr.joblib

# Export model for C++ engine
python export_model.py     # creates model_coefficients.txt
```

### Python Bot (legacy)

```bash
cd experimental
python lichess.py --token <lichess_token>
```

### Python Development

```bash
pip install -e ".[dev]"   # install with dev dependencies
invoke format             # format code with ruff
invoke lint               # check without modifying
invoke test               # run pytest
```

## Architecture

### C++ Engine (`engine/`)

- `base_engine.h` – Abstract `BaseEngine` interface with `evaluate()` and `get_best_move()`
- `chess_engine.h/cpp` – Negamax with alpha-beta, transposition tables, quiescence search, iterative deepening
- `mcts_engine.h/cpp` – Monte Carlo Tree Search with UCT selection and neural/logistic evaluation
- `feature_extractor.h/cpp` – 1544-dim features matching Python: piece-square tables × piece-count scaling + mobility
- `logistic_model.h/cpp` – Loads exported coefficients from `model_coefficients.txt`
- `lichess_client.h/cpp` – HTTP streaming to Lichess API via libcurl
- `main.cpp` – `LichessBot` orchestrating games with retry/heartbeat logic

### Python Components

- `engine/train/` – Data processing and sklearn LogisticRegression training
- `experimental/self_play/` – PyTorch MCTS self-play training (`ChessNet`, `MCTS`)
- `experimental/lichess.py` – Python bot using berserk library

### Feature Extraction

Both C++ and Python use identical 1544-dimensional features:
1. 768 piece-square values (12 piece types × 64 squares)
2. 2 check indicators (white/black in check)
3. All 770 features scaled by piece count factor, then duplicated with (1-factor)
4. 2 mobility features (scaled legal move counts in endgame)

## Dependencies

- **C++**: CMake 3.16+, C++17, libcurl, nlohmann-json, chess-library (fetched via CMake)
- **Python**: chess, numpy, pandas, scikit-learn, joblib, torch (for self-play), berserk (for Python bot)
