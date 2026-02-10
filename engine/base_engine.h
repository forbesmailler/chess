#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <string>

#include "chess_board.h"
#include "generated_config.h"
#include "nnue_model.h"

enum class EvalMode { HANDCRAFTED, NNUE };

struct TimeControl {
    int time_left_ms;
    int increment_ms;
    int moves_to_go;
};

struct SearchResult {
    ChessBoard::Move best_move;
    float score;
    int depth;
    std::chrono::milliseconds time_used;
    int nodes_searched;
};

class BaseEngine {
   public:
    explicit BaseEngine(int max_time_ms = 1000,
                        EvalMode eval_mode = EvalMode::HANDCRAFTED,
                        std::shared_ptr<NNUEModel> nnue_model = nullptr)
        : nnue_model(std::move(nnue_model)),
          max_search_time_ms(max_time_ms),
          eval_mode(eval_mode) {}

    virtual ~BaseEngine() = default;

    virtual float evaluate(const ChessBoard& board) = 0;
    virtual SearchResult get_best_move(const ChessBoard& board,
                                       const TimeControl& time_control) = 0;

    virtual void set_max_time(int max_time_ms) { max_search_time_ms = max_time_ms; }
    virtual int get_max_time() const { return max_search_time_ms; }
    virtual void stop_search() { should_stop.store(true); }
    EvalMode get_eval_mode() const { return eval_mode; }

   protected:
    static constexpr float MATE_VALUE = config::MATE_VALUE;

    std::shared_ptr<NNUEModel> nnue_model;
    int max_search_time_ms;
    EvalMode eval_mode;
    mutable std::atomic<bool> should_stop{false};
    mutable std::atomic<int> nodes_searched{0};

    int calculate_search_time(const TimeControl& time_control);
    float raw_evaluate(const ChessBoard& board);
};
