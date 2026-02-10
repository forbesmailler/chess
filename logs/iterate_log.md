# Iteration Log


============================================================
  Task: Bug fixes
============================================================

  --- Bug fixes - iteration 1 ---

NO_CHANGES

  Converged after 1 iteration(s) (3.1m)

============================================================
  Task: Test coverage
============================================================

  --- Test coverage - iteration 1 ---

98% branch coverage. Every module is well above 90%. The only uncovered lines/branches are the `if __name__ == "__main__"` guards, which are standard boilerplate and don't contain testable logic.

NO_CHANGES

  Converged after 1 iteration(s) (5.2m)

============================================================
  Task: Conciseness
============================================================

  --- Conciseness - iteration 1 ---

Here's a summary of the changes made:

**C++ changes:**

1. **Removed `LichessClient::offer_draw()`** — Dead code, never called anywhere. Identical implementation to `accept_draw()` (both POST to `/draw/yes`). Removed from both `lichess_client.h` and `lichess_client.cpp`.

2. **Removed `ChessBoard::piece_at()`** — Trivial alias that just forwarded to `piece_type_at()`. Removed declaration from `chess_board.h`, definition from `chess_board.cpp`, and the test `PieceAtIsAlias` that only existed to verify the alias.

**Python changes:**

3. **Removed dead code in `test_train_loop.py`** — `fake_run` was defined but `fake_run_with_interrupt` was used instead. `fake_run_with_interrupt` didn't append to `calls`, making the assertion on `calls` vacuously true. Fixed by using `fake_run` (which tracks calls) and removing the unused `call_count` / `fake_run_with_interrupt`.

4. **Removed unused `files` list in `tasks.py`** — The `glob` results were collected into a list and counted but never used; `clang-format` was called with the pattern string directly. Also removed the now-unused `from glob import glob`.

**Results:** All 183 C++ tests pass, all 142 Python tests pass. The `lichess_bot.exe` link error is a pre-existing file lock (the exe is running).


  --- Conciseness - iteration 2 ---

All changes are complete. Summary of additional changes in this pass:

- **Removed unused `#include <algorithm>`** from `engine/mcts_engine.cpp` — no algorithm functions used (only `<cmath>` and `<limits>` needed)
- **Removed unused `#include <cstring>`** from `engine/mcts_engine.h` — no `memcpy`/`memset`/etc. used
- **Removed unused `#include <fstream>`** from `engine/main.cpp` — no file stream operations in main

All 183 C++ tests and 142 Python tests pass. The `lichess_bot.exe` link error is a pre-existing file lock unrelated to our changes (the library compiles fine).


  --- Conciseness - iteration 3 ---

All 183 C++ tests and 142 Python tests pass. Summary of changes in this pass:

1. **Removed unused `#include <algorithm>`** from `engine/mcts_engine.cpp` — no algorithm functions used
2. **Removed unused `#include <cstring>`** from `engine/mcts_engine.h` — no cstring functions used
3. **Removed unused `#include <numeric>`** from `engine/self_play.cpp` — no numeric functions used
4. **Removed unused `#include "handcrafted_eval.h"`** from `engine/self_play.cpp` — function called indirectly through engine virtual method, not directly
5. **Simplified LMR in negamax** (`chess_engine.cpp:249-266`) — three branches with identical `negamax()` calls collapsed into one call with a computed `search_depth`, removing 6 lines
6. **Flattened nesting in `quiescence_search`** (`chess_engine.cpp:348-383`) — replaced `if (!tactical_moves.empty()) { ... }` wrapper with early return, reducing nesting by one level


  --- Conciseness - iteration 4 ---

All 183 C++ tests and 142 Python tests pass.

**Changes in this iteration:**

- Removed unused `#include <memory>` from `tests/engine/test_chess_engine.cpp` (no smart pointers used)
- Verified the `#include <cstring>` removal from `tests/engine/test_engine_edge.cpp` (done just before context compaction)

I thoroughly scanned all remaining C++ source files, headers, and test files. No further unused includes, dead code, unreachable branches, or simplification opportunities remain that I'm confident about.


  --- Conciseness - iteration 5 ---

No commented-out Python code either. I've exhaustively scanned the codebase. The only change in this iteration was removing `#include <memory>` from test_chess_engine.cpp, which has already been verified by a successful build and all tests passing.

NO_CHANGES

  Converged after 5 iteration(s) (32.1m)

============================================================
  Task: Optimization
============================================================

  --- Optimization - iteration 1 ---

All 183 C++ tests and 142 Python tests pass. Formatter ran clean. Here's a summary of the changes:

### Changes Made

**1. `engine/mcts_engine.cpp` — `expand()`: Eliminated redundant board copy + evaluation**
- **Before**: Called `get_move_prior()` which copied the board a second time and re-evaluated the child position — even though `child->board` was already a copy with the move applied.
- **After**: Evaluates `child->board` directly (already exists), computing the prior inline. Removes one `ChessBoard` copy per legal move during expansion.

**2. `engine/mcts_engine.cpp` — `expand()`: Replaced heap-allocated vector with `isGameOver()` for terminal check**
- **Before**: Called `child->board.get_legal_moves()` which heap-allocates a `std::vector<ChessBoard::Move>` just to check `.empty()` for terminal detection.
- **After**: Uses `child->board.board.isGameOver()` — no allocation needed.

**3. `engine/mcts_engine.cpp` — `simulate()`: Replaced heap-allocated vector with stack-allocated `chess::Movelist`**
- **Before**: Called `sim_board.get_legal_moves()` every iteration of the simulation loop, heap-allocating a `std::vector<ChessBoard::Move>` each time (up to `max_simulation_depth` iterations).
- **After**: Uses `chess::Movelist` (stack-allocated, 256 entries) and `chess::movegen::legalmoves()` directly. Also uses `board.inCheck()` instead of `is_checkmate()` (which internally regenerates legal moves) since we already know moves are empty.

**4. `engine/mcts_engine.h` — Removed unused `get_move_prior()` declaration**

**5. `engine/lichess_client.cpp` — `stream_callback()`: Fixed O(n^2) buffer erosion**
- **Before**: Called `buffer.erase(0, pos + 1)` after each line, shifting the entire remaining buffer each time — O(n^2) for chunks with multiple lines.
- **After**: Tracks `start` position, processes all lines in one scan, erases consumed prefix once at the end — O(n). Also avoids creating an intermediate `std::string chunk` by using `.append()` directly.


  --- Optimization - iteration 2 ---

All tests pass (183 C++ + 142 Python), formatter ran clean. The additional change was:

**6. `engine/chess_engine.cpp` — `clear_caches()`: Replaced `assign()` with `memset()` for TT clearing**
- **Before**: `transposition_table.assign(TT_SIZE, TranspositionEntry{})` — constructs 1M `TranspositionEntry` objects individually.
- **After**: `std::memset(transposition_table.data(), 0, TT_SIZE * sizeof(TranspositionEntry))` — bulk zero-fill. Safe because all default field values are zero (`key=0`, `score=0`, `depth=0`, `type=EXACT=0`, `best_move=NO_MOVE=0`).

I don't see any remaining performance issues that would be confident, correct fixes. The remaining patterns (MCTS board copies per node, get_legal_moves wrapper allocation) are either inherent to the architecture or not in hot paths.


  --- Optimization - iteration 3 ---

All done. Here's a summary of all performance fixes made across the three rounds:

1. **`engine/mcts_engine.cpp` — `expand()`**: Eliminated redundant board copy by inlining `get_move_prior()` logic. Replaced `get_legal_moves().empty()` (heap-allocating vector) with `board.isGameOver()` for terminal detection.

2. **`engine/mcts_engine.cpp` — `simulate()`**: Replaced heap-allocated `std::vector<ChessBoard::Move>` with stack-allocated `chess::Movelist`. Replaced `is_checkmate()` with `board.inCheck()`.

3. **`engine/mcts_engine.cpp` — `select()` + `get_uct_value()`**: Precomputed `std::log(parent_visits)` and `std::sqrt(parent_visits)` once per parent, passing them to all children instead of recomputing per child.

4. **`engine/lichess_client.cpp` — `stream_callback()`**: Fixed O(n^2) buffer erosion — tracked consumed offset and erased once at the end instead of per-line.

5. **`engine/chess_engine.cpp` — `clear_caches()`**: Replaced `vector::assign()` with `std::memset()` for zeroing the TT (1M entries) and eval cache (256K entries).

All 183 C++ tests pass, all 142 Python tests pass, formatter reports no changes needed.


  --- Optimization - iteration 4 ---

I've thoroughly re-read every source file in the codebase. After this comprehensive review across all four rounds, I'm confident there are no remaining performance issues that meet the criteria (O(n^2)+ algorithms, repeated lookups that should be cached, unnecessary copies, allocations in tight loops, redundant recomputation) without sacrificing readability.

NO_CHANGES

  Converged after 4 iteration(s) (17.1m)

============================================================
  Task: Config
============================================================

  --- Config - iteration 1 ---

All done. Here's a summary of what was accomplished:

**Constants moved to YAML config files:**

- **engine.yaml**: TT/eval cache sizes, aspiration window params, mate threshold margin, move scoring bonuses (TT move, captures, promotions, killers, countermove, history max), pruning margins (reverse futility, futility depth 1/2, LMP base), quiescence values (piece values, delta margin, en passant value), min/max time bounds, bot retry constants (stream retries, reconnect delay, move/draw retry delays)

- **training.yaml**: Self-play time control, progress log intervals, training random seed, max data workers, comparator settings (threads, max ply, search time, progress interval)

- **deploy.yaml**: Model archive directories (accepted/rejected)

**Files updated:**
- `scripts/gen_config_header.py` — emits all new constants into `generated_config.h`
- `engine/chess_engine.h/cpp` — all search/scoring/pruning constants from config
- `engine/mcts_engine.h` — eval cache size from config
- `engine/self_play.h/cpp` — comparator defaults and time control from config
- `engine/main.cpp` — bot retry constants and time bounds from config
- `engine/train/train_nnue.py` — random seed and max workers from config
- `scripts/train_loop.py` — archive dirs from config
- `tasks.py` — compare games default from config

**Tests:** 183 C++ tests passed, 142 Python tests passed. Formatting applied (ruff + clang-format).


  --- Config - iteration 2 ---

All done. Summary of this iteration:

**Constants moved to YAML config:**

- **`engine.yaml`**: Added `move_scoring.victim_value_multiplier: 10` (MVV-LVA scaling), `tt_replacement.depth_weight: 4` and `tt_replacement.exact_bonus: 2` (TT replacement policy weights), `mcts.node_count_interval: 1000` (MCTS node counter frequency)

**Source files updated:**
- `chess_engine.cpp`: `victim * 10` → `victim * config::search::move_scoring::VICTIM_VALUE_MULTIPLIER`; TT replacement `depth * 4 + 2` → uses `config::search::tt_replacement::DEPTH_WEIGHT` and `EXACT_BONUS`
- `mcts_engine.cpp`: `iterations % 1000` → `iterations % config::mcts::NODE_COUNT_INTERVAL`
- `scripts/gen_config_header.py`: Emits `VICTIM_VALUE_MULTIPLIER`, `tt_replacement::DEPTH_WEIGHT`/`EXACT_BONUS`, `NODE_COUNT_INTERVAL`

**Bug fix:**
- `tasks.py`: Fixed stale fallback `1000` → `100` to match config value

**Tests:** 183 C++ + 142 Python = 325 tests all passing. Formatting applied.


  --- Config - iteration 3 ---

All 183 C++ + 142 Python = 325 tests passing. 

The only change in this iteration was replacing the hardcoded `"https://lichess.org/api"` URL in `lichess_client.cpp:102` with `base_url` (which is already initialized from config). This was a bug — the connectivity test was bypassing the configurable URL.


  --- Config - iteration 4 ---



NO_CHANGES

  Converged after 4 iteration(s) (24.5m)

============================================================
  Task: Markdown
============================================================

  --- Markdown - iteration 1 ---

Here's a summary of what I verified and changed:

**Verified (no issues found):**
- `HANDCRAFTED_EVAL.md` — all 10 claims (phase weights, material values, pawn structure bonuses/penalties, rook file bonuses, bishop pair, mobility, king shield, terminal values) match the code exactly
- `CLAUDE.md` — NNUE architecture (773→256→32→1, ClippedReLU, tanh output, feature encoding, black-to-move transformation) all correct
- `CLAUDE.md` — self-play binary format (42 bytes, all offsets/sizes) matches C++ struct and Python reader
- `CLAUDE.md` — config files, test directories all exist as documented
- `deploy/README.md` — correct
- `scripts/README.md` — correct

**Fixed:**

| File | Change | Reason |
|------|--------|--------|
| `CLAUDE.md` | `base_engine.h` → `base_engine.h/cpp`, updated description | `base_engine.cpp` exists with `raw_evaluate()` and `calculate_search_time()` |
| `README.md` | `base_engine.h` → `base_engine.h/cpp` in project structure | Same as above |
| `README.md` | Added `chess_board.h/cpp` and `utils.h/cpp` to project structure | Files exist and are listed in CLAUDE.md but were missing from the tree |
| `README.md` | Added `--compare` mode to usage examples | Mode exists in `main.cpp` but was undocumented |
| `engine/README.md` | Added `--compare` mode to usage, removed stale "test mode" entry | `--compare` was missing; test mode description replaced |
| `engine/README.md` | Added missing test files to list | `test_chess_board_edge.cpp`, `test_engine_edge.cpp`, `test_utils.cpp` were unlisted |
| `MEMORY.md` | Updated test count from "40 tests total" to "183 C++ + 142 Python = 325 total" | Count was very outdated |


  --- Markdown - iteration 2 ---

All remaining claims verified as correct. No further changes needed.

NO_CHANGES

  Converged after 2 iteration(s) (8.2m)

============================================================
  Summary (90.2m)
============================================================
  converged       Bug fixes (1 iter, 3.1m)
  converged       Test coverage (1 iter, 5.2m)
  converged       Conciseness (5 iter, 32.2m)
  converged       Optimization (4 iter, 17.1m)
  converged       Config (4 iter, 24.5m)
  converged       Markdown (2 iter, 8.2m)

