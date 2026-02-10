#include "mcts_engine.h"

#include <cmath>
#include <limits>

MCTSEngine::MCTSEngine(int max_time_ms, EvalMode eval_mode,
                       std::shared_ptr<NNUEModel> nnue_model)
    : BaseEngine(max_time_ms, eval_mode, std::move(nnue_model)),
      rng(std::random_device{}()) {
    eval_cache.resize(EVAL_CACHE_SIZE);
}

SearchResult MCTSEngine::get_best_move(const ChessBoard& board,
                                       const TimeControl& time_control) {
    auto legal_moves = board.get_legal_moves();

    if (legal_moves.empty()) {
        float score = board.is_in_check(board.turn()) ? -MATE_VALUE : 0.0f;
        return {ChessBoard::Move{}, score, 0, std::chrono::milliseconds(0), 0};
    }
    if (legal_moves.size() == 1) {
        return {legal_moves[0], 0.0f, 1, std::chrono::milliseconds(50), 1};
    }

    int search_time = calculate_search_time(time_control);
    auto start_time = std::chrono::steady_clock::now();
    should_stop.store(false);
    nodes_searched.store(0);

    auto root = std::make_unique<MCTSNode>();
    root->board = board;
    root->visits = 1;

    int iterations = 0;
    while (!should_stop.load()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
        if (elapsed.count() >= search_time) break;

        MCTSNode* leaf = select(root.get());

        if (leaf && !leaf->is_terminal) {
            expand(leaf);

            if (!leaf->children.empty()) {
                leaf = leaf->children[0].get();
            }
        }

        if (leaf) {
            float score = simulate(leaf->board);
            backpropagate(leaf, score);
        }

        iterations++;
        if (iterations % config::mcts::NODE_COUNT_INTERVAL == 0) {
            nodes_searched.store(iterations);
        }
    }

    ChessBoard::Move best_move;
    int max_visits = 0;
    float best_score = -std::numeric_limits<float>::infinity();

    for (const auto& child : root->children) {
        if (child->visits > max_visits) {
            max_visits = child->visits;
            best_move = child->move;
            best_score = -child->get_average_score();
        }
    }

    auto time_used = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start_time);

    return {best_move, best_score, 0, time_used, iterations};
}

float MCTSEngine::evaluate(const ChessBoard& board) {
    uint64_t pos_key = board.hash();
    auto& entry = eval_cache[pos_key & EVAL_CACHE_MASK];
    if (entry.key == pos_key) return entry.score;

    float eval = raw_evaluate(board);
    entry = {pos_key, eval};
    return eval;
}

MCTSEngine::MCTSNode* MCTSEngine::select(MCTSNode* root) {
    MCTSNode* current = root;

    while (!current->is_leaf() && !current->is_terminal) {
        float log_parent = std::log(static_cast<float>(current->visits));
        float sqrt_parent = std::sqrt(static_cast<float>(current->visits));

        MCTSNode* best_child = nullptr;
        float best_uct = -std::numeric_limits<float>::infinity();

        for (const auto& child : current->children) {
            float uct =
                child->get_uct_value(exploration_constant, log_parent, sqrt_parent);
            if (uct > best_uct) {
                best_uct = uct;
                best_child = child.get();
            }
        }

        if (best_child) {
            current = best_child;
        } else {
            break;
        }
    }

    return current;
}

void MCTSEngine::expand(MCTSNode* node) {
    if (node->is_expanded || node->is_terminal) return;

    const auto& legal_moves = node->get_legal_moves();
    if (legal_moves.empty()) {
        node->is_terminal = true;
        return;
    }

    float parent_eval = evaluate(node->board);

    node->children.reserve(legal_moves.size());
    for (const auto& move : legal_moves) {
        auto child = std::make_unique<MCTSNode>();
        child->board = node->board;
        child->move = move;
        child->parent = node;

        if (child->board.make_move(move)) {
            float child_eval = evaluate(child->board);
            float improvement = node->board.turn() == ChessBoard::WHITE
                                    ? child_eval - parent_eval
                                    : parent_eval - child_eval;
            child->prior_probability =
                1.0f /
                (1.0f + std::exp(-improvement / config::mcts::PRIOR_SIGMOID_SCALE));

            auto [reason, result] = child->board.board.isGameOver();
            if (result != chess::GameResult::NONE) {
                child->is_terminal = true;
            }

            node->children.push_back(std::move(child));
        }
    }

    node->is_expanded = true;
}

float MCTSEngine::simulate(const ChessBoard& board) {
    ChessBoard sim_board = board;
    int depth = 0;

    while (depth < max_simulation_depth) {
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, sim_board.board);
        if (moves.empty()) {
            if (sim_board.board.inCheck()) {
                return sim_board.turn() == board.turn() ? -MATE_VALUE : MATE_VALUE;
            }
            return 0.0f;
        }

        if (depth % config::mcts::EVAL_FREQUENCY == 0) {
            float eval = evaluate(sim_board);
            if (std::abs(eval) > MATE_VALUE * config::mcts::EARLY_TERMINATION_FACTOR) {
                return board.turn() == ChessBoard::WHITE ? eval : -eval;
            }
        }

        std::uniform_int_distribution<int> dist(0, moves.size() - 1);
        chess::Move selected = moves[dist(rng)];
        sim_board.board.makeMove(selected);

        depth++;
    }

    float eval = evaluate(sim_board);
    return board.turn() == ChessBoard::WHITE ? eval : -eval;
}

void MCTSEngine::backpropagate(MCTSNode* node, float score) {
    MCTSNode* current = node;
    bool flip_score = false;

    while (current != nullptr) {
        current->visits++;
        current->total_score += flip_score ? -score : score;

        current = current->parent;
        flip_score = !flip_score;
    }
}

float MCTSEngine::MCTSNode::get_uct_value(float exploration_constant, float log_parent,
                                          float sqrt_parent) const {
    if (visits == 0) return std::numeric_limits<float>::infinity();

    float exploitation = -get_average_score();
    float exploration = exploration_constant * std::sqrt(log_parent / visits);

    float prior_bonus =
        prior_probability * exploration_constant * sqrt_parent / (1 + visits);

    return exploitation + exploration + prior_bonus;
}

const std::vector<ChessBoard::Move>& MCTSEngine::MCTSNode::get_legal_moves() const {
    if (!legal_moves_cached) {
        legal_moves_cache = board.get_legal_moves();
        legal_moves_cached = true;
    }
    return legal_moves_cache;
}
