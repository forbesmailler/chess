#include "nnue_model.h"

#ifdef __AVX2__
#include <immintrin.h>
#else
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

void NNUEModel::AlignedDeleter::operator()(void* p) const {
#ifdef _MSC_VER
    _aligned_free(p);
#else
    std::free(p);
#endif
}

template <typename T>
NNUEModel::AlignedPtr<T> NNUEModel::alloc_aligned(size_t count) {
    constexpr size_t ALIGNMENT = 32;
    size_t bytes = count * sizeof(T);
    bytes = (bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
#ifdef _MSC_VER
    T* p = static_cast<T*>(_aligned_malloc(bytes, ALIGNMENT));
#else
    T* p = static_cast<T*>(std::aligned_alloc(ALIGNMENT, bytes));
#endif
    return AlignedPtr<T>(p);
}

static int16_t quantize_clamp(float v, float scale) {
    float scaled = v * scale;
    return static_cast<int16_t>(
        std::max(-32767.0f, std::min(32767.0f, std::round(scaled))));
}

bool NNUEModel::load_weights(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open NNUE weight file: " << path << std::endl;
        return false;
    }
    return load_weights(file);
}

bool NNUEModel::load_weights(std::istream& file) {
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

    auto read_floats = [&file](size_t count) -> std::vector<float> {
        std::vector<float> v(count);
        file.read(reinterpret_cast<char*>(v.data()), count * sizeof(float));
        return v;
    };

    auto w1_f = read_floats(INPUT_SIZE * HIDDEN1_SIZE);
    auto b1_f = read_floats(HIDDEN1_SIZE);
    auto w2_f = read_floats(HIDDEN1_SIZE * HIDDEN2_SIZE);
    auto b2_f = read_floats(HIDDEN2_SIZE);
    auto w3_f = read_floats(HIDDEN2_SIZE * OUTPUT_SIZE);
    auto b3_f = read_floats(OUTPUT_SIZE);

    if (!file) {
        std::cerr << "Error reading NNUE weight file (truncated?)" << std::endl;
        return false;
    }

    // Quantize w1 to int16 with padding to H1_PADDED
    w1_q = alloc_aligned<int16_t>(INPUT_SIZE * H1_PADDED);
    for (int i = 0; i < INPUT_SIZE; ++i) {
        for (int j = 0; j < HIDDEN1_SIZE; ++j)
            w1_q[i * H1_PADDED + j] =
                quantize_clamp(w1_f[i * HIDDEN1_SIZE + j], Q1_SCALE);
        for (int j = HIDDEN1_SIZE; j < H1_PADDED; ++j) w1_q[i * H1_PADDED + j] = 0;
    }

    // Quantize b1
    b1_q = alloc_aligned<int16_t>(H1_PADDED);
    for (int j = 0; j < HIDDEN1_SIZE; ++j) b1_q[j] = quantize_clamp(b1_f[j], Q1_SCALE);
    for (int j = HIDDEN1_SIZE; j < H1_PADDED; ++j) b1_q[j] = 0;

    // Quantize w2 and transpose: (HIDDEN1_SIZE x HIDDEN2_SIZE) → (HIDDEN2_SIZE x
    // H1_PADDED)
    w2_q = alloc_aligned<int16_t>(HIDDEN2_SIZE * H1_PADDED);
    for (int j = 0; j < HIDDEN2_SIZE; ++j) {
        for (int i = 0; i < HIDDEN1_SIZE; ++i)
            w2_q[j * H1_PADDED + i] =
                quantize_clamp(w2_f[i * HIDDEN2_SIZE + j], Q2_SCALE);
        for (int i = HIDDEN1_SIZE; i < H1_PADDED; ++i) w2_q[j * H1_PADDED + i] = 0;
    }

    // Copy b2, w3, b3 to aligned buffers
    size_t b2_pad = (HIDDEN2_SIZE + 7) & ~size_t(7);
    b2 = alloc_aligned<float>(b2_pad);
    std::memcpy(b2.get(), b2_f.data(), HIDDEN2_SIZE * sizeof(float));

    size_t w3_pad = (static_cast<size_t>(HIDDEN2_SIZE) * OUTPUT_SIZE + 7) & ~size_t(7);
    w3 = alloc_aligned<float>(w3_pad);
    std::memcpy(w3.get(), w3_f.data(), HIDDEN2_SIZE * OUTPUT_SIZE * sizeof(float));

    size_t b3_pad = (OUTPUT_SIZE + 7) & ~size_t(7);
    b3 = alloc_aligned<float>(b3_pad);
    std::memcpy(b3.get(), b3_f.data(), OUTPUT_SIZE * sizeof(float));

    loaded = true;
    return true;
}

int NNUEModel::extract_features(const ChessBoard& board, int* active) {
    int count = 0;
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
            active[count++] = pt * 64 + sq;
        }

        auto opp_pieces = b.pieces(PIECE_TYPES[pt], opp_color);
        while (opp_pieces) {
            int sq = static_cast<int>(opp_pieces.pop());
            if (!white_to_move) sq ^= 56;
            active[count++] = 384 + pt * 64 + sq;
        }
    }

    int cr = b.castlingRights().hashIndex();
    if (white_to_move) {
        if (cr & 1) active[count++] = 768;
        if (cr & 2) active[count++] = 769;
        if (cr & 4) active[count++] = 770;
        if (cr & 8) active[count++] = 771;
    } else {
        if (cr & 4) active[count++] = 768;
        if (cr & 8) active[count++] = 769;
        if (cr & 1) active[count++] = 770;
        if (cr & 2) active[count++] = 771;
    }

    if (b.enpassantSq() != chess::Square::NO_SQ) active[count++] = 772;
    return count;
}

std::vector<int> NNUEModel::get_active_features(const ChessBoard& board) const {
    int active[MAX_ACTIVE_FEATURES];
    int count = extract_features(board, active);
    return std::vector<int>(active, active + count);
}

float NNUEModel::predict(const ChessBoard& board) const {
    if (!loaded) return 0.0f;

    int active[MAX_ACTIVE_FEATURES];
    int num_active = extract_features(board, active);

    alignas(32) float h2[HIDDEN2_SIZE];

#ifdef __AVX2__
    // Layer 1: quantized int16 sparse accumulation (16 values per AVX2 op)
    alignas(32) int16_t h1_q[H1_PADDED];
    std::memcpy(h1_q, b1_q.get(), H1_PADDED * sizeof(int16_t));
    for (int k = 0; k < num_active; ++k) {
        const int16_t* row = w1_q.get() + active[k] * H1_PADDED;
        if (k + 1 < num_active) {
            _mm_prefetch(
                reinterpret_cast<const char*>(w1_q.get() + active[k + 1] * H1_PADDED),
                _MM_HINT_T0);
        }
        for (int j = 0; j < H1_PADDED; j += 16) {
            __m256i h = _mm256_load_si256(reinterpret_cast<const __m256i*>(&h1_q[j]));
            __m256i r = _mm256_load_si256(reinterpret_cast<const __m256i*>(&row[j]));
            _mm256_store_si256(reinterpret_cast<__m256i*>(&h1_q[j]),
                               _mm256_adds_epi16(h, r));
        }
    }

    // ClippedReLU in int16: clamp to [0, Q1_SCALE]
    {
        __m256i zero = _mm256_setzero_si256();
        __m256i qmax = _mm256_set1_epi16(static_cast<int16_t>(Q1_SCALE));
        for (int j = 0; j < H1_PADDED; j += 16) {
            __m256i h = _mm256_load_si256(reinterpret_cast<const __m256i*>(&h1_q[j]));
            h = _mm256_max_epi16(zero, _mm256_min_epi16(qmax, h));
            _mm256_store_si256(reinterpret_cast<__m256i*>(&h1_q[j]), h);
        }
    }

    // Layer 2: tiled int16 dot product — process 4 outputs per pass
    // Reads h1_q once per tile, accumulates 4 sums simultaneously
    for (int j = 0; j < HIDDEN2_SIZE; j += 4) {
        __m256i s0 = _mm256_setzero_si256();
        __m256i s1 = _mm256_setzero_si256();
        __m256i s2 = _mm256_setzero_si256();
        __m256i s3 = _mm256_setzero_si256();
        const int16_t* r0 = w2_q.get() + (j + 0) * H1_PADDED;
        const int16_t* r1 = w2_q.get() + (j + 1) * H1_PADDED;
        const int16_t* r2 = w2_q.get() + (j + 2) * H1_PADDED;
        const int16_t* r3 = w2_q.get() + (j + 3) * H1_PADDED;
        for (int i = 0; i < H1_PADDED; i += 16) {
            __m256i h = _mm256_load_si256(reinterpret_cast<const __m256i*>(&h1_q[i]));
            s0 = _mm256_add_epi32(
                s0,
                _mm256_madd_epi16(
                    h, _mm256_load_si256(reinterpret_cast<const __m256i*>(&r0[i]))));
            s1 = _mm256_add_epi32(
                s1,
                _mm256_madd_epi16(
                    h, _mm256_load_si256(reinterpret_cast<const __m256i*>(&r1[i]))));
            s2 = _mm256_add_epi32(
                s2,
                _mm256_madd_epi16(
                    h, _mm256_load_si256(reinterpret_cast<const __m256i*>(&r2[i]))));
            s3 = _mm256_add_epi32(
                s3,
                _mm256_madd_epi16(
                    h, _mm256_load_si256(reinterpret_cast<const __m256i*>(&r3[i]))));
        }
        // Horizontal sum each and store
        auto hsum = [](__m256i v) -> int32_t {
            __m128i lo = _mm256_castsi256_si128(v);
            __m128i hi = _mm256_extracti128_si256(v, 1);
            __m128i s = _mm_add_epi32(lo, hi);
            s = _mm_hadd_epi32(s, s);
            s = _mm_hadd_epi32(s, s);
            return _mm_cvtsi128_si32(s);
        };
        for (int k = 0; k < 4; ++k) {
            int32_t raw;
            switch (k) {
                case 0:
                    raw = hsum(s0);
                    break;
                case 1:
                    raw = hsum(s1);
                    break;
                case 2:
                    raw = hsum(s2);
                    break;
                default:
                    raw = hsum(s3);
                    break;
            }
            float sum = static_cast<float>(raw) * DEQUANT_SCALE + b2[j + k];
            h2[j + k] = std::max(0.0f, std::min(1.0f, sum));
        }
    }
#else
    // Layer 1: quantized int16 sparse accumulation (8 values per SSE2 op)
    alignas(16) int16_t h1_q[H1_PADDED];
    std::memcpy(h1_q, b1_q.get(), H1_PADDED * sizeof(int16_t));
    for (int k = 0; k < num_active; ++k) {
        const int16_t* row = w1_q.get() + active[k] * H1_PADDED;
        if (k + 1 < num_active) {
            _mm_prefetch(
                reinterpret_cast<const char*>(w1_q.get() + active[k + 1] * H1_PADDED),
                _MM_HINT_T0);
        }
        for (int j = 0; j < H1_PADDED; j += 8) {
            __m128i h = _mm_load_si128(reinterpret_cast<const __m128i*>(&h1_q[j]));
            __m128i r = _mm_load_si128(reinterpret_cast<const __m128i*>(&row[j]));
            _mm_store_si128(reinterpret_cast<__m128i*>(&h1_q[j]), _mm_adds_epi16(h, r));
        }
    }

    // ClippedReLU in int16
    {
        __m128i zero = _mm_setzero_si128();
        __m128i qmax = _mm_set1_epi16(static_cast<int16_t>(Q1_SCALE));
        for (int j = 0; j < H1_PADDED; j += 8) {
            __m128i h = _mm_load_si128(reinterpret_cast<const __m128i*>(&h1_q[j]));
            h = _mm_max_epi16(zero, _mm_min_epi16(qmax, h));
            _mm_store_si128(reinterpret_cast<__m128i*>(&h1_q[j]), h);
        }
    }

    // Layer 2: int16 dot product using _mm_madd_epi16
    for (int j = 0; j < HIDDEN2_SIZE; ++j) {
        __m128i sum_vec = _mm_setzero_si128();
        const int16_t* row = w2_q.get() + j * H1_PADDED;
        for (int i = 0; i < H1_PADDED; i += 8) {
            __m128i h = _mm_load_si128(reinterpret_cast<const __m128i*>(&h1_q[i]));
            __m128i w = _mm_load_si128(reinterpret_cast<const __m128i*>(&row[i]));
            sum_vec = _mm_add_epi32(sum_vec, _mm_madd_epi16(h, w));
        }
        // Horizontal sum of 4 int32s
        __m128i hi = _mm_shuffle_epi32(sum_vec, _MM_SHUFFLE(1, 0, 3, 2));
        sum_vec = _mm_add_epi32(sum_vec, hi);
        hi = _mm_shuffle_epi32(sum_vec, _MM_SHUFFLE(0, 1, 0, 1));
        sum_vec = _mm_add_epi32(sum_vec, hi);
        float sum =
            static_cast<float>(_mm_cvtsi128_si32(sum_vec)) * DEQUANT_SCALE + b2[j];
        h2[j] = std::max(0.0f, std::min(1.0f, sum));
    }
#endif

    // Layer 3: single output (tanh)
    float logit = b3[0];
    for (int i = 0; i < HIDDEN2_SIZE; ++i) logit += h2[i] * w3[i];

    // Fast tanh approximation
    float x = logit;
    if (x > 4.0f)
        x = 1.0f;
    else if (x < -4.0f)
        x = -1.0f;
    else {
        float x2 = x * x;
        x = x * (27.0f + x2) / (27.0f + 9.0f * x2);
    }
    float stm_eval = x * MATE_VALUE;

    bool white_to_move = board.board.sideToMove() == chess::Color::WHITE;
    return white_to_move ? stm_eval : -stm_eval;
}
