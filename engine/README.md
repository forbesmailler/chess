# C++ Engine

The chess engine and Lichess bot. See the [top-level README](../README.md) for build instructions and project overview.

## Running

Set `LICHESS_TOKEN` environment variable, then:

```bash
# Bot mode (defaults: 1000ms search time, negamax, handcrafted eval)
./lichess_bot [max_time_ms] [--engine=negamax|mcts] [--eval=handcrafted|nnue] [--nnue-weights=path]

# Self-play data generation
./lichess_bot --selfplay [num_games] [search_depth] [output_file] [num_threads]

# Compare two models
./lichess_bot --compare <old_weights|handcrafted> <new_weights> [num_games] [output_file] [threads]
```

## Dependencies

Managed via CMake. [chess-library](https://github.com/Disservin/chess-library) and GoogleTest are fetched automatically. libcurl and nlohmann-json must be installed on the system (or via vcpkg on Windows).

## Tests

```bash
cd build
ctest -C Release --output-on-failure
```

Test files in `../tests/engine/`:
- `test_chess_board.cpp` — board operations, FEN parsing, move generation
- `test_chess_board_edge.cpp` — edge cases for board operations
- `test_chess_engine.cpp` — negamax search, time management
- `test_engine_edge.cpp` — edge cases for search
- `test_handcrafted_eval.cpp` — evaluation terms
- `test_nnue_model.cpp` — NNUE forward pass
- `test_self_play.cpp` — self-play data generation
- `test_utils.cpp` — utility functions
