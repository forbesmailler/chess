#pragma once

#include <cstdint>
#include <istream>
#include <memory>
#include <string>
#include <vector>

#include "chess_board.h"
#include "generated_config.h"

class NNUEModel {
   public:
    NNUEModel() = default;

    // Load binary weight file exported by export_nnue.py
    bool load_weights(const std::string& path);

    // Load weights from an already-open stream (for in-memory loading)
    bool load_weights(std::istream& stream);

    // Evaluate position, returns score from white's perspective in centipawns
    float predict(const ChessBoard& board) const;

    bool is_loaded() const { return loaded; }

    // Public accessor for testing: returns active feature indices for a position
    std::vector<int> get_active_features(const ChessBoard& board) const;

   private:
    static constexpr int INPUT_SIZE = config::nnue::INPUT_SIZE;
    static constexpr int HIDDEN1_SIZE = config::nnue::HIDDEN1_SIZE;
    static constexpr int HIDDEN2_SIZE = config::nnue::HIDDEN2_SIZE;
    static constexpr int OUTPUT_SIZE = config::nnue::OUTPUT_SIZE;
    static constexpr float MATE_VALUE = config::MATE_VALUE;
    static constexpr int MAX_ACTIVE_FEATURES = 37;  // 32 pieces + 4 castling + 1 ep

    // Quantization scales
    static constexpr int Q1_SCALE = 512;  // Layer 1 weights/accumulators
    static constexpr int Q2_SCALE = 512;  // Layer 2 weights
    // Padded HIDDEN1_SIZE rounded up to multiple of 16 for AVX2 int16 ops
    static constexpr int H1_PADDED = (HIDDEN1_SIZE + 15) & ~15;

    // Aligned memory deleter for SIMD-aligned allocations
    struct AlignedDeleter {
        void operator()(void* p) const;
    };
    template <typename T>
    using AlignedPtr = std::unique_ptr<T[], AlignedDeleter>;

    template <typename T>
    static AlignedPtr<T> alloc_aligned(size_t count);

    // Quantized layer 1 weights (int16, scaled by Q1_SCALE)
    AlignedPtr<int16_t> w1_q;  // INPUT_SIZE x H1_PADDED
    AlignedPtr<int16_t> b1_q;  // H1_PADDED

    // Quantized layer 2 weights (int16, scaled by Q2_SCALE, transposed)
    AlignedPtr<int16_t> w2_q;  // HIDDEN2_SIZE x H1_PADDED
    AlignedPtr<float> b2;      // HIDDEN2_SIZE

    // Layer 3 weights as float
    AlignedPtr<float> w3;  // HIDDEN2_SIZE x OUTPUT_SIZE
    AlignedPtr<float> b3;  // OUTPUT_SIZE

    // Dequantization constant: 1 / (Q1_SCALE * Q2_SCALE)
    static constexpr float DEQUANT_SCALE = 1.0f / (Q1_SCALE * Q2_SCALE);

    bool loaded = false;

    // Extract features into a stack array, returns count
    static int extract_features(const ChessBoard& board, int* active);
};
