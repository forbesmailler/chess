# Iteration Log

  Running formatter...

Formatting complete — no issues reported.


============================================================
  Task: Task 1
============================================================

  --- Task 1 - iteration 1 ---

Here's a summary of the changes made:

## Changes

### 1. CMakeLists.txt — Fix AVX2 flags never being applied on Windows

**Root cause of the 2.6x slowdown**: The `CMAKE_BUILD_TYPE` check never matched on Windows because Visual Studio is a multi-config generator where `CMAKE_BUILD_TYPE` is empty at configure time. This meant `/arch:AVX2` was **never applied**, so the NNUE was running on the SSE2 fallback path (processing 8 int16s at a time instead of 16, no FMA).

**Fix**: Use generator expressions `$<$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>:...>` for `/O2 /GL` etc., and apply `/arch:AVX2` unconditionally (it's always wanted).

### 2. nnue_model.cpp — Fuse ClippedReLU into Layer 2 dot product (`forward_from_accumulator`)

Eliminated the copy + separate clamp pass. Previously: `memcpy(h1_q, h1_acc)` → clamp h1_q → read h1_q for each of 32 outputs. Now: read h1_acc directly, clamp in-register, then madd — no intermediate buffer, no extra memory traffic.

### 3. nnue_model.cpp — Merge feature extraction into `predict()`

Instead of extracting all features into an array then iterating the array to accumulate, features are now accumulated directly as they're discovered from bitboards. This eliminates the intermediate feature array and the second loop.

## Results

| Metric | Before | After |
|--------|--------|-------|
| NNUE predict (from-scratch) | 2.6x slower than handcrafted | 2.0x slower (informational, no assertion) |
| NNUE incremental | slower than handcrafted (failed) | **0.94x** — faster than handcrafted (passes) |

The incremental benchmark (`EvalSpeedBenchmarkIncremental`) now passes — NNUE incremental eval is faster than handcrafted eval, which is the path used during actual search.


  --- Task 1 - iteration 2 ---
