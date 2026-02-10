#include "base_engine.h"

#include <algorithm>

#include "handcrafted_eval.h"

int BaseEngine::calculate_search_time(const TimeControl& time_control) {
    if (time_control.time_left_ms <= 0) return max_search_time_ms;

    int allocated_time =
        time_control.increment_ms +
        (time_control.time_left_ms / config::search::TIME_ALLOCATION_DIVISOR);
    return std::max(1, std::min(allocated_time, max_search_time_ms));
}

float BaseEngine::raw_evaluate(const ChessBoard& board) {
    auto [reason, result] = board.board.isGameOver();
    if (result != chess::GameResult::NONE) {
        if (reason == chess::GameResultReason::CHECKMATE)
            return board.turn() == ChessBoard::WHITE ? -MATE_VALUE : MATE_VALUE;
        return 0.0f;
    }

    if (eval_mode == EvalMode::NNUE && nnue_model) {
        if (nnue_model->has_accumulator())
            return nnue_model->predict_from_accumulator(board);
        return nnue_model->predict(board);
    }
    return handcrafted_evaluate(board);
}
