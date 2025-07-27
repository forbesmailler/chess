#pragma once
#include "chess_board.h"
#include "logistic_model.h"
#include <memory>
#include <unordered_map>
#include <vector>

struct TranspositionEntry {
    float score;
    int depth;
    enum NodeType { EXACT, LOWER_BOUND, UPPER_BOUND } type;
    ChessBoard::Move best_move;
};

class ChessEngine {
public:
    explicit ChessEngine(std::shared_ptr<LogisticModel> model, int search_depth = 4);
    
    float evaluate(const ChessBoard& board);
    ChessBoard::Move get_best_move(const ChessBoard& board);
    void set_depth(int depth) { search_depth = depth; }
    int get_depth() const { return search_depth; }
    
private:
    static constexpr float MATE_VALUE = 1.0f;
    static constexpr size_t CACHE_SIZE = 1000000; // Larger transposition table
    
    std::shared_ptr<LogisticModel> model;
    int search_depth;
    mutable std::unordered_map<std::string, TranspositionEntry> transposition_table;
    mutable std::unordered_map<std::string, float> eval_cache;
    
    // Search functions
    float negamax(const ChessBoard& board, int depth, float alpha, float beta, bool is_pv = false);
    float quiescence_search(const ChessBoard& board, float alpha, float beta, int depth = 0);
    
    // Move ordering
    std::vector<ChessBoard::Move> order_moves(const ChessBoard& board, const std::vector<ChessBoard::Move>& moves, const ChessBoard::Move& tt_move = ChessBoard::Move{});
    int score_move(const ChessBoard& board, const ChessBoard::Move& move);
    
    // Evaluation helpers
    float get_piece_value(ChessBoard::PieceType piece) const;
    float evaluate_position(const ChessBoard& board);
    
    // Utility
    void clear_cache_if_needed();
    std::string get_position_key(const ChessBoard& board) const;
};
