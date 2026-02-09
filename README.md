# Chess Bot

A Lichess chess bot written in C++ with a Python training pipeline.

## Features

- **Search**: Negamax (alpha-beta pruning, transposition tables, quiescence search, iterative deepening) and MCTS (UCT selection)
- **Evaluation**: Handcrafted tapered eval or NNUE (773→256→32→1)
- **Training**: Self-play data generation → PyTorch NNUE training → binary weight export
- **Deployment**: Connects to Lichess API; runs as a systemd service on Linux

## Quick Start

### Build

```powershell
# Windows (requires vcpkg with curl, nlohmann-json)
cd engine
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="C:/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build . --config Release
```

```bash
# Linux/macOS
sudo apt install build-essential cmake libcurl4-openssl-dev nlohmann-json3-dev  # Debian/Ubuntu
cd engine && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
```

### Run

Set the `LICHESS_TOKEN` environment variable, then:

```bash
# Play on Lichess
./lichess_bot [max_time_ms] [--engine=negamax|mcts] [--eval=handcrafted|nnue] [--nnue-weights=path]

# Generate self-play training data
./lichess_bot --selfplay [num_games] [search_depth] [output_file] [num_threads]

# Compare two models
./lichess_bot --compare <old_weights|handcrafted> <new_weights> [num_games] [output_file] [threads]
```

### Test

```bash
ctest -C Release --output-on-failure   # C++ tests
pytest                                  # Python tests
```

## NNUE Training Pipeline

```bash
# 1. Generate self-play data
./lichess_bot --selfplay 1000 6 training_data.bin 8

# 2. Train NNUE model
cd engine/train
python train_nnue.py --data ../../training_data.bin --output nnue_weights.pt --epochs 100

# 3. Export to binary for C++ inference
python export_nnue.py --model nnue_weights.pt --output nnue.bin

# 4. Run with NNUE
./lichess_bot 1000 --eval=nnue --nnue-weights=nnue.bin
```

Or use the all-in-one invoke task:

```bash
invoke train --games=1000 --depth=6 --threads=8 --epochs=100
```

## Development

```bash
pip install -e ".[dev]"       # install dev dependencies

invoke gen-config             # regenerate engine/generated_config.h from YAML
invoke format                 # format Python (ruff) + C++ (clang-format)
invoke test                   # run all tests (Python + C++)
invoke prepare                # gen-config → format → build → test
invoke train                  # prepare → continuous RL loop
invoke deploy                 # deploy to Linux VPS
```

## Project Structure

```
engine/                         # C++ engine and Lichess bot
├── base_engine.h/cpp           # BaseEngine interface, EvalMode enum
├── chess_engine.h/cpp          # Negamax search
├── mcts_engine.h/cpp           # MCTS search
├── handcrafted_eval.h/cpp      # Tapered evaluation
├── nnue_model.h/cpp            # NNUE inference
├── self_play.h/cpp             # Self-play data generator
├── chess_board.h/cpp           # Board wrapper utilities
├── utils.h/cpp                 # Shared helper functions
├── lichess_client.h/cpp        # Lichess API client (libcurl)
├── main.cpp                    # Entry point
├── generated_config.h          # Auto-generated from YAML
└── train/
    ├── train_nnue.py           # PyTorch NNUE training
    └── export_nnue.py          # Export to binary weights
config/                         # YAML configuration files
scripts/                        # Code generation & training loop
tests/                          # All tests (Python + C++)
├── engine/                     # C++ GTest unit tests
├── config/                     # Config loader tests
├── scripts/                    # Gen-config & train loop tests
└── train/                      # NNUE training & export tests
deploy/                         # systemd service & deployment guide
```

## Dependencies

- **C++**: CMake 3.16+, C++17, libcurl, nlohmann-json, [chess-library](https://github.com/Disservin/chess-library) (fetched automatically)
- **Python**: chess, numpy, tqdm; torch (NNUE training)
