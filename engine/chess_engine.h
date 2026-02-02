#pragma once
#include "base_engine.h"
#include "chess_board.h"
#include "logistic_model.h"
#include <memory>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <atomic>

struct TranspositionEntry {
    float score;
    int depth;
    enum NodeType { EXACT, LOWER_BOUND, UPPER_BOUND } type;
    ChessBoard::Move best_move;
};

class ChessEngine : public BaseEngine {
public:
    explicit ChessEngine(std::shared_ptr<LogisticModel> model, int max_time_ms = 1000);
    
    float evaluate(const ChessBoard& board) override;
    SearchResult get_best_move(const ChessBoard& board, const TimeControl& time_control) override;
    
    void set_max_time(int max_time_ms) override { max_search_time_ms = max_time_ms; }
    int get_max_time() const override { return max_search_time_ms; }
    void stop_search() override { should_stop.store(true); }
    
private:
    static constexpr float SEARCH_INTERRUPTED = -99999.0f;  // Special value for interrupted search
    static constexpr size_t CACHE_SIZE = 1000000;
    
    mutable std::unordered_map<std::string, TranspositionEntry> transposition_table;
    mutable std::unordered_map<std::string, float> eval_cache;
    
    // Time management
    int calculate_search_time(const TimeControl& time_control);
    
    // Search functions
    SearchResult iterative_deepening_search(const ChessBoard& board, int max_time_ms);
    float negamax(const ChessBoard& board, int depth, float alpha, float beta, bool is_pv = false);
    float quiescence_search(const ChessBoard& board, float alpha, float beta, int depth = 0);
    std::vector<ChessBoard::Move> order_moves(const ChessBoard& board, const std::vector<ChessBoard::Move>& moves, const ChessBoard::Move& tt_move = ChessBoard::Move{});
    int score_move(const ChessBoard& board, const ChessBoard::Move& move);
    void clear_cache_if_needed();
    std::string get_position_key(const ChessBoard& board) const;
};
