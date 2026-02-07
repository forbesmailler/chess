#include "nnue_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

bool NNUEModel::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open NNUE weight file: " << path << std::endl;
        return false;
    }

    // Read header
    char magic[4];
    file.read(magic, 4);
    if (std::memcmp(magic, "NNUE", 4) != 0) {
        std::cerr << "Invalid NNUE file magic number" << std::endl;
        return false;
    }

    uint32_t version, input_size, hidden1_size, hidden2_size, output_size;
    file.read(reinterpret_cast<char*>(&version), 4);
    file.read(reinterpret_cast<char*>(&input_size), 4);
    file.read(reinterpret_cast<char*>(&hidden1_size), 4);
    file.read(reinterpret_cast<char*>(&hidden2_size), 4);
    file.read(reinterpret_cast<char*>(&output_size), 4);

    if (input_size != INPUT_SIZE || hidden1_size != HIDDEN1_SIZE || hidden2_size != HIDDEN2_SIZE ||
        output_size != OUTPUT_SIZE) {
        std::cerr << "NNUE architecture mismatch: expected " << INPUT_SIZE << "/" << HIDDEN1_SIZE
                  << "/" << HIDDEN2_SIZE << "/" << OUTPUT_SIZE << ", got " << input_size << "/"
                  << hidden1_size << "/" << hidden2_size << "/" << output_size << std::endl;
        return false;
    }

    // Read weights and biases
    auto read_vec = [&file](std::vector<float>& vec, size_t count) {
        vec.resize(count);
        file.read(reinterpret_cast<char*>(vec.data()), count * sizeof(float));
    };

    read_vec(w1, INPUT_SIZE * HIDDEN1_SIZE);
    read_vec(b1, HIDDEN1_SIZE);
    read_vec(w2, HIDDEN1_SIZE * HIDDEN2_SIZE);
    read_vec(b2, HIDDEN2_SIZE);
    read_vec(w3, HIDDEN2_SIZE * OUTPUT_SIZE);
    read_vec(b3, OUTPUT_SIZE);

    if (!file) {
        std::cerr << "Error reading NNUE weight file (truncated?)" << std::endl;
        return false;
    }

    loaded = true;
    std::cout << "Loaded NNUE model (v" << version << "): " << INPUT_SIZE << " -> " << HIDDEN1_SIZE
              << " -> " << HIDDEN2_SIZE << " -> " << OUTPUT_SIZE << std::endl;
    return true;
}

std::vector<float> NNUEModel::extract_features(const ChessBoard& board) {
    std::vector<float> features(INPUT_SIZE, 0.0f);
    const auto& b = board.board;
    bool white_to_move = b.sideToMove() == chess::Color::WHITE;

    // Feature layout: 0-383 = own pieces (P/N/B/R/Q/K x 64), 384-767 = opponent pieces
    // When black to move, flip board vertically and swap colors
    static constexpr chess::PieceType PIECE_TYPES[] = {
        chess::PieceType::PAWN, chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
        chess::PieceType::ROOK, chess::PieceType::QUEEN,  chess::PieceType::KING};

    auto own_color = white_to_move ? chess::Color::WHITE : chess::Color::BLACK;
    auto opp_color = white_to_move ? chess::Color::BLACK : chess::Color::WHITE;

    for (int pt = 0; pt < 6; ++pt) {
        // Own pieces -> indices 0-383
        auto own_pieces = b.pieces(PIECE_TYPES[pt], own_color);
        while (own_pieces) {
            int sq = static_cast<int>(own_pieces.pop());
            if (!white_to_move) sq ^= 56;  // Vertical flip for black
            features[pt * 64 + sq] = 1.0f;
        }

        // Opponent pieces -> indices 384-767
        auto opp_pieces = b.pieces(PIECE_TYPES[pt], opp_color);
        while (opp_pieces) {
            int sq = static_cast<int>(opp_pieces.pop());
            if (!white_to_move) sq ^= 56;  // Vertical flip for black
            features[384 + pt * 64 + sq] = 1.0f;
        }
    }

    return features;
}

float NNUEModel::clipped_relu(float x) { return std::max(0.0f, std::min(1.0f, x)); }

std::vector<float> NNUEModel::softmax(const std::vector<float>& logits) {
    float max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<float> result(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp(logits[i] - max_val);
        sum += result[i];
    }
    for (auto& v : result) v /= sum;
    return result;
}

float NNUEModel::predict(const ChessBoard& board) const {
    if (!loaded) return 0.0f;

    if (board.is_checkmate()) return board.turn() == ChessBoard::WHITE ? -MATE_VALUE : MATE_VALUE;
    if (board.is_stalemate() || board.is_draw()) return 0.0f;

    auto input = extract_features(board);

    // Layer 1: input -> hidden1 (ClippedReLU)
    std::vector<float> h1(HIDDEN1_SIZE);
    for (int j = 0; j < HIDDEN1_SIZE; ++j) {
        float sum = b1[j];
        for (int i = 0; i < INPUT_SIZE; ++i) sum += input[i] * w1[i * HIDDEN1_SIZE + j];
        h1[j] = clipped_relu(sum);
    }

    // Layer 2: hidden1 -> hidden2 (ClippedReLU)
    std::vector<float> h2(HIDDEN2_SIZE);
    for (int j = 0; j < HIDDEN2_SIZE; ++j) {
        float sum = b2[j];
        for (int i = 0; i < HIDDEN1_SIZE; ++i) sum += h1[i] * w2[i * HIDDEN2_SIZE + j];
        h2[j] = clipped_relu(sum);
    }

    // Layer 3: hidden2 -> output (softmax)
    std::vector<float> logits(OUTPUT_SIZE);
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        float sum = b3[j];
        for (int i = 0; i < HIDDEN2_SIZE; ++i) sum += h2[i] * w3[i * OUTPUT_SIZE + j];
        logits[j] = sum;
    }

    auto proba = softmax(logits);
    // proba[0] = P(win), proba[1] = P(draw), proba[2] = P(loss)
    // from side-to-move's perspective
    float stm_eval = (proba[0] - proba[2]) * MATE_VALUE;

    // Convert to white's perspective
    bool white_to_move = board.turn() == ChessBoard::WHITE;
    return white_to_move ? stm_eval : -stm_eval;
}
