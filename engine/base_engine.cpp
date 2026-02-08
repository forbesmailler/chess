#include "base_engine.h"

#include <algorithm>

#include "feature_extractor.h"
#include "handcrafted_eval.h"

std::string BaseEngine::get_position_key(const ChessBoard& board) const {
    std::string fen = board.to_fen();
    size_t pos = fen.find(' ');
    for (int i = 0; i < 3 && pos != std::string::npos; ++i) {
        pos = fen.find(' ', pos + 1);
    }
    return pos != std::string::npos ? fen.substr(0, pos) : fen;
}

int BaseEngine::calculate_search_time(const TimeControl& time_control) {
    if (time_control.time_left_ms <= 0) return max_search_time_ms;

    int allocated_time = time_control.increment_ms + (time_control.time_left_ms / 40);
    return std::min(allocated_time, max_search_time_ms);
}

float BaseEngine::raw_evaluate(const ChessBoard& board) {
    if (board.is_checkmate()) return board.turn() == ChessBoard::WHITE ? -MATE_VALUE : MATE_VALUE;
    if (board.is_stalemate() || board.is_draw()) return 0.0f;

    switch (eval_mode) {
        case EvalMode::HANDCRAFTED:
            return handcrafted_evaluate(board);
        case EvalMode::NNUE:
            if (nnue_model) return nnue_model->predict(board);
            return handcrafted_evaluate(board);
        case EvalMode::LOGISTIC:
        default: {
            auto features = FeatureExtractor::extract_features(board);
            auto proba = model->predict_proba(features);
            return (proba[2] - proba[0]) * MATE_VALUE;
        }
    }
}
