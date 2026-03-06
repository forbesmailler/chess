#include "base_engine.h"

#include <algorithm>

#include "handcrafted_eval.h"

int BaseEngine::calculate_search_time(const TimeControl& time_control) {
    static constexpr int MAX_THINK_MS = config::search::MAX_THINK_MS;

    // Correspondence or no clock: use the max
    if (time_control.time_left_ms <= 0 && time_control.increment_ms <= 0)
        return MAX_THINK_MS;

    int allocated_time =
        time_control.increment_ms +
        (time_control.time_left_ms / config::search::TIME_ALLOCATION_DIVISOR);
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
