#include "base_engine.h"

#include <algorithm>

#include "handcrafted_eval.h"

int BaseEngine::calculate_search_time(const TimeControl& time_control) {
    static constexpr int MAX_THINK_MS = config::search::MAX_THINK_MS;
    static constexpr int OVERHEAD = config::search::MOVE_OVERHEAD_MS;
    static constexpr int RESERVE = config::search::MIN_TIME_RESERVE_MS;

    // Correspondence or no clock: use the max
    if (time_control.time_left_ms <= 0 && time_control.increment_ms <= 0)
        return MAX_THINK_MS;

    // Fixed time per move (self-play): time_left=0 means no clock, just use increment
    if (time_control.time_left_ms <= 0)
        return std::clamp(time_control.increment_ms, 1, MAX_THINK_MS);

    // Live game: subtract overhead for network latency + HTTP round-trip
    int usable_time = time_control.time_left_ms - OVERHEAD;
    if (usable_time < 1) return 1;

    // Keep a reserve so we don't flag
    int available = usable_time - RESERVE;
    if (available < 1) {
        // Emergency: use half of whatever is left, minimum 1ms
        return std::max(1, usable_time / 2);
    }

    int base = available - time_control.increment_ms;
    if (base < 0) base = 0;
    int allocated_time =
        time_control.increment_ms + (base / config::search::TIME_ALLOCATION_DIVISOR);
    return std::clamp(allocated_time, 1, MAX_THINK_MS);
}

float BaseEngine::raw_evaluate(const ChessBoard& board) {
    auto [reason, result] = board.board.isGameOver();
    if (result != chess::GameResult::NONE) {
        if (reason == chess::GameResultReason::CHECKMATE)
            return board.turn() == ChessBoard::WHITE ? -MATE_VALUE : MATE_VALUE;
        return 0.0f;
    }
    return position_evaluate(board);
}

float BaseEngine::position_evaluate(const ChessBoard& board) {
    if (eval_mode == EvalMode::NNUE && nnue_model) {
        if (nnue_model->has_accumulator())
            return nnue_model->predict_from_accumulator(board);
        return nnue_model->predict(board);
    }
    return handcrafted_evaluate(board);
}
