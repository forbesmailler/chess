#include "nnue_model.h"

#include <xmmintrin.h>

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

    if (input_size != INPUT_SIZE || hidden1_size != HIDDEN1_SIZE ||
        hidden2_size != HIDDEN2_SIZE || output_size != OUTPUT_SIZE) {
        std::cerr << "NNUE architecture mismatch: expected " << INPUT_SIZE << "/"
                  << HIDDEN1_SIZE << "/" << HIDDEN2_SIZE << "/" << OUTPUT_SIZE
                  << ", got " << input_size << "/" << hidden1_size << "/"
                  << hidden2_size << "/" << output_size << std::endl;
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

    // Transpose w2 from (HIDDEN1_SIZE x HIDDEN2_SIZE) to (HIDDEN2_SIZE x HIDDEN1_SIZE)
    // for cache-friendly sequential access in the dense layer 2 inner loop.
    {
        std::vector<float> w2_t(HIDDEN1_SIZE * HIDDEN2_SIZE);
        for (int i = 0; i < HIDDEN1_SIZE; ++i)
            for (int j = 0; j < HIDDEN2_SIZE; ++j)
                w2_t[j * HIDDEN1_SIZE + i] = w2[i * HIDDEN2_SIZE + j];
        w2 = std::move(w2_t);
    }

    loaded = true;
    return true;
}

void NNUEModel::extract_features(const ChessBoard& board, std::vector<int>& active) {
    active.clear();
    const auto& b = board.board;
    bool white_to_move = b.sideToMove() == chess::Color::WHITE;

    static constexpr chess::PieceType PIECE_TYPES[] = {
        chess::PieceType::PAWN, chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
        chess::PieceType::ROOK, chess::PieceType::QUEEN,  chess::PieceType::KING};

    auto own_color = white_to_move ? chess::Color::WHITE : chess::Color::BLACK;
    auto opp_color = white_to_move ? chess::Color::BLACK : chess::Color::WHITE;

    for (int pt = 0; pt < 6; ++pt) {
        auto own_pieces = b.pieces(PIECE_TYPES[pt], own_color);
        while (own_pieces) {
            int sq = static_cast<int>(own_pieces.pop());
            if (!white_to_move) sq ^= 56;
            active.push_back(pt * 64 + sq);
        }

        auto opp_pieces = b.pieces(PIECE_TYPES[pt], opp_color);
        while (opp_pieces) {
            int sq = static_cast<int>(opp_pieces.pop());
            if (!white_to_move) sq ^= 56;
            active.push_back(384 + pt * 64 + sq);
        }
    }

    auto rights = board.get_castling_rights();
    if (white_to_move) {
        if (rights.white_kingside) active.push_back(768);
        if (rights.white_queenside) active.push_back(769);
        if (rights.black_kingside) active.push_back(770);
        if (rights.black_queenside) active.push_back(771);
    } else {
        if (rights.black_kingside) active.push_back(768);
        if (rights.black_queenside) active.push_back(769);
        if (rights.white_kingside) active.push_back(770);
        if (rights.white_queenside) active.push_back(771);
    }

    if (b.enpassantSq() != chess::Square::NO_SQ) active.push_back(772);
}

float NNUEModel::predict(const ChessBoard& board) const {
    if (!loaded) return 0.0f;

    thread_local std::vector<int> active;
    float h1[HIDDEN1_SIZE];
    float h2[HIDDEN2_SIZE];

    extract_features(board, active);

    // Layer 1: sparse accumulation (input features are binary) — SSE vectorized
    std::memcpy(h1, b1.data(), HIDDEN1_SIZE * sizeof(float));
    for (size_t k = 0; k < active.size(); ++k) {
        const float* row = w1.data() + active[k] * HIDDEN1_SIZE;
        if (k + 1 < active.size()) {
            _mm_prefetch(
                reinterpret_cast<const char*>(w1.data() + active[k + 1] * HIDDEN1_SIZE),
                _MM_HINT_T0);
        }
        for (int j = 0; j < HIDDEN1_SIZE; j += 4) {
            __m128 h = _mm_loadu_ps(&h1[j]);
            __m128 r = _mm_loadu_ps(&row[j]);
            _mm_storeu_ps(&h1[j], _mm_add_ps(h, r));
        }
    }
    {
        __m128 zero = _mm_setzero_ps();
        __m128 one = _mm_set1_ps(1.0f);
        for (int j = 0; j < HIDDEN1_SIZE; j += 4) {
            __m128 h = _mm_loadu_ps(&h1[j]);
            _mm_storeu_ps(&h1[j], _mm_max_ps(zero, _mm_min_ps(one, h)));
        }
    }

    // Layer 2: dense, w2 transposed to (HIDDEN2_SIZE x HIDDEN1_SIZE) — SSE vectorized
    for (int j = 0; j < HIDDEN2_SIZE; ++j) {
        __m128 sum_vec = _mm_setzero_ps();
        const float* row = w2.data() + j * HIDDEN1_SIZE;
        for (int i = 0; i < HIDDEN1_SIZE; i += 4) {
            __m128 h = _mm_loadu_ps(&h1[i]);
            __m128 r = _mm_loadu_ps(&row[i]);
            sum_vec = _mm_add_ps(sum_vec, _mm_mul_ps(h, r));
        }
        __m128 hi = _mm_movehl_ps(sum_vec, sum_vec);
        sum_vec = _mm_add_ps(sum_vec, hi);
        __m128 shuf = _mm_shuffle_ps(sum_vec, sum_vec, 1);
        sum_vec = _mm_add_ss(sum_vec, shuf);
        float sum = _mm_cvtss_f32(sum_vec) + b2[j];
        h2[j] = std::max(0.0f, std::min(1.0f, sum));
    }

    // Layer 3: single output (tanh)
    float logit = b3[0];
    for (int i = 0; i < HIDDEN2_SIZE; ++i) logit += h2[i] * w3[i];
    float stm_eval = std::tanh(logit) * MATE_VALUE;

    // Convert to white's perspective
    bool white_to_move = board.turn() == ChessBoard::WHITE;
    return white_to_move ? stm_eval : -stm_eval;
}
