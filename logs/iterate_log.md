# Iteration Log

  Running formatter...

<<<<<<< HEAD
Formatting completed successfully — no output indicates no issues were found by ruff or clang-format.
=======
All files already formatted � no changes needed.
>>>>>>> 42405512f103aa92fddf7436f9e4e24782c87cf7


============================================================
  Task: Task 1
============================================================

  --- Task 1 - iteration 1 ---
<<<<<<< HEAD




  --- Task 1 - iteration 2 ---

All done. Here's a summary of what was implemented:

## NNUE Incremental Accumulation

**Problem**: NNUE eval was 2.3x slower than handcrafted (0.27 us vs 0.13 us per eval). The bottleneck was Layer 1 sparse accumulation: ~33 features × 16 AVX2 ops per position, recomputed from scratch every time.

**Solution**: Dual-perspective incremental accumulators (same approach used by Stockfish). Instead of recomputing all ~33 feature activations from scratch, maintain two Layer 1 accumulators (one per color's perspective) on a stack. When a move is made, update both incrementally with only ~2-6 feature changes.

**Files modified**:
- `engine/nnue_model.h` — Added `Accumulator` struct, stack, and incremental API
- `engine/nnue_model.cpp` — Implemented `init_accumulator`, `push/pop_accumulator`, `update_accumulator` (handles normal moves, castling, en passant, promotion, captures), `update_accumulator_null_move`, `predict_from_accumulator`, `forward_from_accumulator`
- `engine/base_engine.cpp` — Routes eval to accumulator path when available
- `engine/chess_engine.cpp` — Hooks push/update/pop around every `makeMove`/`unmakeMove` in IDS root, negamax, null move, and quiescence search
- `tests/engine/test_nnue_model.cpp` — Added 8 accumulator correctness tests (start pos, move sequences, castling, en passant, promotion, null move) and incremental benchmark

**Results**:
- **Before**: NNUE 0.27 us/eval, Handcrafted 0.13 us/eval (2.1x slower)
- **After**: NNUE incremental 0.12 us/eval, Handcrafted 0.13 us/eval (**0.94x ratio — NNUE is now faster**)
- **193 C++ tests + 157 Python tests = 350 total, all passing**


  --- Task 1 - iteration 3 ---

NO_CHANGES

  Converged after 3 iteration(s) (25.2m)

============================================================
  Summary (25.5m)
============================================================
  converged       Task 1 (3 iter, 25.2m)

=======
>>>>>>> 42405512f103aa92fddf7436f9e4e24782c87cf7
