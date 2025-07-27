#pragma once
#include "chess_board.h"
#include "logistic_model.h"
#include <memory>
#include <unordered_map>

class ChessEngine {
public:
    explicit ChessEngine(std::shared_ptr<LogisticModel> model);
    
    float evaluate(const ChessBoard& board);
    ChessBoard::Move get_best_move(const ChessBoard& board);
    
private:
    static constexpr float WIN_VALUE = 1.0f;
    static constexpr int DEFAULT_DEPTH = 4;
    static constexpr size_t CACHE_SIZE = 100000;
    
    std::shared_ptr<LogisticModel> model;
    mutable std::unordered_map<std::string, float> eval_cache;
    
    float negamax(const ChessBoard& board, int depth, float alpha, float beta);
    void clear_cache_if_needed();
};
