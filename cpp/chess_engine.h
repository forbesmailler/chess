#pragma once
#include "chess_board.h"
#include "logistic_model.h"
#include <memory>
#include <unordered_map>

class ChessEngine {
public:
    explicit ChessEngine(std::shared_ptr<LogisticModel> model, int search_depth = 4);
    
    float evaluate(const ChessBoard& board);
    ChessBoard::Move get_best_move(const ChessBoard& board);
    void set_depth(int depth) { search_depth = depth; }
    int get_depth() const { return search_depth; }
    
private:
    static constexpr float WIN_VALUE = 1.0f;
    static constexpr size_t CACHE_SIZE = 500000; // Increased cache size
    
    std::shared_ptr<LogisticModel> model;
    int search_depth;
    mutable std::unordered_map<std::string, float> eval_cache;
    
    float negamax(const ChessBoard& board, int depth, float alpha, float beta);
    void clear_cache_if_needed();
};
