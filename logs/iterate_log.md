# Iteration Log

  Running formatter...

Formatting completed successfully — both ruff (Python) and clang-format (C++) ran with no issues.


============================================================
  Task: Config
============================================================

  --- Config - iteration 1 ---

Now I have a complete picture. Let me analyze what's actually hardcoded vs. already in config.

**C++ source files - Analysis of potential hardcoded constants:**

1. **NNUE feature indices (384, 768-772)**: These are derived from the NNUE architecture (6 types × 64 squares = 384, etc.) and are part of the documented NNUE encoding spec. They're fundamentally tied to the architecture constants already in config (input_size=773). Extracting these to YAML wouldn't make sense — they're computed from `6*64=384`, `6*64*2=768`, etc. These are structural constants, not tunable parameters.

2. **NNUE quantization scales (Q1_SCALE=512, Q2_SCALE=512)**: These are NNUE implementation details tied to the int16 quantization strategy. They must match between training and inference. Not really tunable — changing them would require retraining and re-exporting weights. More of a build-time architectural choice.

3. **Tanh approximation constants (4.0, 27.0, 9.0)**: These are mathematical constants for the Padé approximant of tanh, not tuning parameters. The formula `x*(27+x²)/(27+9x²)` is a well-known approximation. Configuring these doesn't make sense.

4. **Array size 256**: This is `chess::Movelist`'s max capacity — a chess library constant, not our tuning parameter.

5. **ACC_STACK_SIZE=128**: Maximum search depth for accumulator stack. Already have MAX_DEPTH=50 in config; this is a safe upper bound for stack allocation.

6. **MAX_ACTIVE_FEATURES=37**: Derived from chess rules (32 max pieces + 4 castling + 1 EP). Not configurable.

7. **SEARCH_INTERRUPTED = -99999.0f**: A sentinel value, not a tuning parameter.

8. **50ms in single-move response** (chess_engine.cpp:131, mcts_engine.cpp:22): This is a trivial constant for when there's only one legal move — the returned time_used is meaningless. Not worth configuring.

9. **1000 default max_time_ms** (base_engine.h:30, chess_engine.h:21): These are constructor default parameter values, not runtime tuning. The actual time comes from TimeControl.

10. **main.cpp line 789 `max_time_ms = 1000`**: Default CLI argument, overridden by user.

**Python source files:**
- All training hyperparameters, NNUE architecture sizes, and self-play parameters are already loaded from config YAML.
- `POSITION_SIZE=42` and `POSITION_BYTES=42` reflect the binary struct layout — not configurable.
- `VERSION=1` in export_nnue.py — file format version, not a tuning parameter.
- `MAX_ACTIVE=37` — chess-derived constant.

**Conclusion:** Every genuine tuning parameter, timeout, URL, and configurable constant is already in the YAML config files and flows through `generated_config.h` to C++ or through `load_config.py` to Python. The remaining hardcoded values are either:
- Chess fundamentals (6 piece types, 64 squares)
- Mathematical constants (tanh approximation)
- Derived from config values (feature indices from architecture)
- Sentinel/format values (SEARCH_INTERRUPTED, POSITION_SIZE)
- Constructor defaults overridden at runtime

NO_CHANGES

  Converged after 1 iteration(s) (2.1m)

============================================================
  Task: Markdown
============================================================

  --- Markdown - iteration 1 ---

Here's a summary of what I found and fixed:

**3 fixes made:**

1. **`engine/HANDCRAFTED_EVAL.md`**: PST section falsely claimed "Separate MG and EG tables exist for pawns and kings; other pieces share the same table across both phases." All 6 piece types have separate MG and EG tables (verified in `eval.yaml` and `generated_config.h`). Fixed to: "All piece types have separate MG and EG tables."

2. **`CLAUDE.md`**: Dependencies listed `chess, numpy, tqdm` but `pyyaml` is a core dependency in `pyproject.toml`. Added `pyyaml`.

3. **`README.md`**: Same missing `pyyaml` dependency. Added it.

**Additionally updated `MEMORY.md`**: Test counts were stale (183 C++ → 177, 142 Python → 157).

**Verified as accurate** (no changes needed):
- NNUE architecture (773→256→32→1)
- All handcrafted eval values (material, PST descriptions, pawn structure, rook bonuses, bishop pair, mobility, king shield)
- Self-play binary format (42 bytes, all fields/offsets)
- CLI arguments and modes
- All file paths in architecture tables
- Invoke task names and descriptions
- Search feature documentation
- deploy/README.md, scripts/README.md, engine/README.md — all accurate
- No broken links found

**Tests**: 177/177 C++ passed, 109/109 Python passed (torch tests skipped — no torch installed). Formatter ran cleanly.


  --- Markdown - iteration 2 ---

All tests pass, formatter ran cleanly. Two additional fixes made:

1. **`engine/HANDCRAFTED_EVAL.md`**: Corrected "scores a position in centipawns" — the function sigmoid-scales the centipawn score to (-MATE_VALUE, +MATE_VALUE), so the output is not in centipawns.

2. **`scripts/README.md`**: Fixed "dead code" → "conciseness" in the task sequence description to match the actual task name in the table below.


  --- Markdown - iteration 3 ---

Confirmed — ClippedReLU is used. Everything checks out.

Let me also do a final check on the `scripts/README.md` `Markdown` reference in the `scripts/iterate_log.md` — not relevant since the README only documents defaults.

I've now exhaustively reviewed every claim in every markdown file. All remaining content is accurate.

NO_CHANGES

  Converged after 3 iteration(s) (9.8m)

============================================================
  Summary (12.2m)
============================================================
  converged       Config (1 iter, 2.1m)
  converged       Markdown (3 iter, 9.8m)

