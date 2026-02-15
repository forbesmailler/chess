# Iteration Log

  Running formatter...

All files already formatted — no changes needed.


============================================================
  Task: Optimization
============================================================

  --- Optimization - iteration 1 ---

One performance fix made in `chess_engine.cpp:order_moves`:

**`order_moves` indirect index sort replaced with direct ScoredMove sort** — The old code built an index array, sorted it with indirect score comparisons (`scores[a] > scores[b]`), then copied moves through a temp buffer (two indirection levels + two copy passes). The new code pairs each move with its score in a `ScoredMove` struct, sorts directly (cache-friendly field access), and copies once. Eliminates one array and one copy pass, with better cache locality during the sort.

The rest of the codebase is already well-optimized:
- Negamax/qsearch use incremental `pick_move` (not full sort) — correct for alpha-beta
- TT and eval cache use flat arrays with power-of-2 hash masking — O(1) lookup
- NNUE uses SIMD accumulation, quantized weights, stack-allocated accumulators
- Handcrafted eval does single-pass bitboard iteration with no allocations
- No O(n^2)+ algorithms found in hot paths


  --- Optimization - iteration 2 ---

I've now thoroughly reviewed every source file in the codebase. Let me do a final systematic check for the specific categories mentioned in the task:

**O(n^2)+ algorithms:** 
- `pick_move` in negamax/qsearch is O(n) per call × n moves = O(n^2) total, but this is intentional (incremental selection for alpha-beta, usually exits early on cutoffs)
- No other O(n^2)+ patterns found

**Repeated lookups that should be cached:**
- `nnue_model && eval_mode == EvalMode::NNUE` is evaluated per-move in negamax (line 259) and qsearch (line 398) loops, but these are cheap boolean checks on member fields that the compiler will optimize (likely kept in registers). Not worth hoisting for readability cost.
- All eval lookups use the eval cache already. TT lookups are cached.

**Unnecessary copies:**
- The `order_moves` indirect sort was the main one — already fixed.
- `ChessBoard::get_legal_moves()` returns `vector<Move>` (heap allocation) — but it's only used in MCTS and self-play, not in the hot negamax path (which uses `chess::Movelist` directly).

**Allocations in tight loops:**
- No heap allocations in negamax or qsearch (stack arrays throughout).
- Self-play softmax phase allocates `vector<float>` per ply, but only for ~16 plies per game — not a tight loop.

**Redundant recomputation:**
- `MCTS::select` computes both `log` and `sqrt` of parent visits — these are passed to `get_uct_value`, so no redundancy.
- No redundant recomputation patterns found.

The codebase is genuinely well-optimized. The `order_moves` fix from the previous round was the only meaningful improvement.

NO_CHANGES

  Converged after 2 iteration(s) (5.7m)

============================================================
  Summary (5.9m)
============================================================
  converged       Optimization (2 iter, 5.7m)

