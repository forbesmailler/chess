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

    alignas(32) int16_t h1_q[H1_PADDED];
    std::memcpy(h1_q, b1_q.get(), H1_PADDED * sizeof(int16_t));

    // Fused feature extraction + Layer 1 sparse accumulation.
    // Accumulate directly as features are discovered (no intermediate array).
    const auto& b = board.board;
    bool white_to_move = b.sideToMove() == chess::Color::WHITE;
    const int16_t* w1 = w1_q.get();

    auto acc_add = [&](int feature) {
        const int16_t* row = w1 + feature * H1_PADDED;
#ifdef __AVX2__
        for (int j = 0; j < H1_PADDED; j += 16) {
            __m256i h = _mm256_load_si256(reinterpret_cast<const __m256i*>(&h1_q[j]));
            __m256i r = _mm256_load_si256(reinterpret_cast<const __m256i*>(&row[j]));
            _mm256_store_si256(reinterpret_cast<__m256i*>(&h1_q[j]),
                               _mm256_adds_epi16(h, r));
        }
#else
        for (int j = 0; j < H1_PADDED; j += 8) {
            __m128i h = _mm_load_si128(reinterpret_cast<const __m128i*>(&h1_q[j]));
            __m128i r = _mm_load_si128(reinterpret_cast<const __m128i*>(&row[j]));
            _mm_store_si128(reinterpret_cast<__m128i*>(&h1_q[j]),
                            _mm_adds_epi16(h, r));
        }
#endif
    };

    static constexpr chess::PieceType PIECE_TYPES[] = {
        chess::PieceType::PAWN, chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
        chess::PieceType::ROOK, chess::PieceType::QUEEN,  chess::PieceType::KING};

    auto own_color = white_to_move ? chess::Color::WHITE : chess::Color::BLACK;
    auto opp_color = white_to_move ? chess::Color::BLACK : chess::Color::WHITE;

    for (int pt = 0; pt < 6; ++pt) {
        int own_base = pt * 64;
        int opp_base = 384 + pt * 64;

        auto own_pieces = b.pieces(PIECE_TYPES[pt], own_color);
        while (own_pieces) {
            int sq = static_cast<int>(own_pieces.pop());
            if (!white_to_move) sq ^= 56;
            acc_add(own_base + sq);
        }

        auto opp_pieces = b.pieces(PIECE_TYPES[pt], opp_color);
        while (opp_pieces) {
            int sq = static_cast<int>(opp_pieces.pop());
            if (!white_to_move) sq ^= 56;
            acc_add(opp_base + sq);
        }
    }

    int cr = b.castlingRights().hashIndex();
    if (white_to_move) {
        if (cr & 1) acc_add(768);
        if (cr & 2) acc_add(769);
        if (cr & 4) acc_add(770);
        if (cr & 8) acc_add(771);
    } else {
        if (cr & 4) acc_add(768);
        if (cr & 8) acc_add(769);
        if (cr & 1) acc_add(770);
        if (cr & 2) acc_add(771);
    }

    if (b.enpassantSq() != chess::Square::NO_SQ) acc_add(772);

    // Forward from Layer 1 output through Layer 2+3 (with fused clamp)
    float stm_eval = forward_from_accumulator(h1_q);

    return white_to_move ? stm_eval : -stm_eval;
}

// --- Incremental accumulator implementation ---

void NNUEModel::compute_accumulator(const ChessBoard& board, int16_t* acc,
                                    bool as_white) const {
    const auto& b = board.board;
    const int16_t* w1 = w1_q.get();

    std::memcpy(acc, b1_q.get(), H1_PADDED * sizeof(int16_t));

    static constexpr chess::PieceType PIECE_TYPES[] = {
        chess::PieceType::PAWN, chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
        chess::PieceType::ROOK, chess::PieceType::QUEEN,  chess::PieceType::KING};

    auto own_color = as_white ? chess::Color::WHITE : chess::Color::BLACK;
    auto opp_color = as_white ? chess::Color::BLACK : chess::Color::WHITE;

    for (int pt = 0; pt < 6; ++pt) {
        int own_base = pt * 64;
        int opp_base = 384 + pt * 64;

        auto own_pieces = b.pieces(PIECE_TYPES[pt], own_color);
        while (own_pieces) {
            int sq = static_cast<int>(own_pieces.pop());
            if (!as_white) sq ^= 56;
            accumulate_add(acc, own_base + sq);
        }

        auto opp_pieces = b.pieces(PIECE_TYPES[pt], opp_color);
        while (opp_pieces) {
            int sq = static_cast<int>(opp_pieces.pop());
            if (!as_white) sq ^= 56;
            accumulate_add(acc, opp_base + sq);
        }
    }

    int cr = b.castlingRights().hashIndex();
    if (as_white) {
        if (cr & 1) accumulate_add(acc, 768);
        if (cr & 2) accumulate_add(acc, 769);
        if (cr & 4) accumulate_add(acc, 770);
        if (cr & 8) accumulate_add(acc, 771);
    } else {
        if (cr & 4) accumulate_add(acc, 768);
        if (cr & 8) accumulate_add(acc, 769);
        if (cr & 1) accumulate_add(acc, 770);
        if (cr & 2) accumulate_add(acc, 771);
    }

    if (b.enpassantSq() != chess::Square::NO_SQ) accumulate_add(acc, 772);
}

void NNUEModel::accumulate_add(int16_t* acc, int feature) const {
    const int16_t* row = w1_q.get() + feature * H1_PADDED;
#ifdef __AVX2__
    for (int j = 0; j < H1_PADDED; j += 16) {
        __m256i h = _mm256_load_si256(reinterpret_cast<const __m256i*>(&acc[j]));
        __m256i r = _mm256_load_si256(reinterpret_cast<const __m256i*>(&row[j]));
        _mm256_store_si256(reinterpret_cast<__m256i*>(&acc[j]),
                           _mm256_adds_epi16(h, r));
    }
#else
    for (int j = 0; j < H1_PADDED; j += 8) {
        __m128i h = _mm_load_si128(reinterpret_cast<const __m128i*>(&acc[j]));
        __m128i r = _mm_load_si128(reinterpret_cast<const __m128i*>(&row[j]));
        _mm_store_si128(reinterpret_cast<__m128i*>(&acc[j]), _mm_adds_epi16(h, r));
    }
#endif
}

void NNUEModel::accumulate_sub(int16_t* acc, int feature) const {
    const int16_t* row = w1_q.get() + feature * H1_PADDED;
#ifdef __AVX2__
    for (int j = 0; j < H1_PADDED; j += 16) {
        __m256i h = _mm256_load_si256(reinterpret_cast<const __m256i*>(&acc[j]));
        __m256i r = _mm256_load_si256(reinterpret_cast<const __m256i*>(&row[j]));
        _mm256_store_si256(reinterpret_cast<__m256i*>(&acc[j]),
                           _mm256_subs_epi16(h, r));
    }
#else
    for (int j = 0; j < H1_PADDED; j += 8) {
        __m128i h = _mm_load_si128(reinterpret_cast<const __m128i*>(&acc[j]));
        __m128i r = _mm_load_si128(reinterpret_cast<const __m128i*>(&row[j]));
        _mm_store_si128(reinterpret_cast<__m128i*>(&acc[j]), _mm_subs_epi16(h, r));
    }
#endif
}

void NNUEModel::init_accumulator(const ChessBoard& board) const {
    acc_ply = 0;
    auto& acc = acc_stack[0];
    compute_accumulator(board, acc.white, true);
    compute_accumulator(board, acc.black, false);
    acc.castling_hash = board.board.castlingRights().hashIndex();
    acc.has_ep = board.board.enpassantSq() != chess::Square::NO_SQ;
    acc.computed = true;
}

void NNUEModel::push_accumulator() const {
    if (acc_ply < 0 || acc_ply + 1 >= ACC_STACK_SIZE) return;
    auto& src = acc_stack[acc_ply];
    auto& dst = acc_stack[acc_ply + 1];
    std::memcpy(dst.white, src.white, H1_PADDED * sizeof(int16_t));
    std::memcpy(dst.black, src.black, H1_PADDED * sizeof(int16_t));
    dst.castling_hash = src.castling_hash;
    dst.has_ep = src.has_ep;
    dst.computed = src.computed;
    ++acc_ply;
}

void NNUEModel::pop_accumulator() const {
    if (acc_ply > 0) --acc_ply;
}

void NNUEModel::update_accumulator(chess::Move move, chess::Piece moved_piece,
                                   chess::Piece captured_piece,
                                   const ChessBoard& board_after) const {
    if (acc_ply < 0) return;
    auto& acc = acc_stack[acc_ply];

    auto moving_color = moved_piece.color();
    bool mover_is_white = moving_color == chess::Color::WHITE;
    int from_sq = move.from().index();
    int to_sq = move.to().index();
    int pt = static_cast<int>(moved_piece.type());
    auto move_type = move.typeOf();

    if (move_type == chess::Move::CASTLING) {
        bool king_side = move.to() > move.from();
        int king_from = from_sq;
        int rook_from = to_sq;
        int king_to =
            chess::Square::castling_king_square(king_side, moving_color).index();
        int rook_to =
            chess::Square::castling_rook_square(king_side, moving_color).index();

        int king_pt = static_cast<int>(chess::PieceType::KING);
        int rook_pt = static_cast<int>(chess::PieceType::ROOK);

        if (mover_is_white) {
            accumulate_sub(acc.white, king_pt * 64 + king_from);
            accumulate_add(acc.white, king_pt * 64 + king_to);
            accumulate_sub(acc.white, rook_pt * 64 + rook_from);
            accumulate_add(acc.white, rook_pt * 64 + rook_to);
            accumulate_sub(acc.black, 384 + king_pt * 64 + (king_from ^ 56));
            accumulate_add(acc.black, 384 + king_pt * 64 + (king_to ^ 56));
            accumulate_sub(acc.black, 384 + rook_pt * 64 + (rook_from ^ 56));
            accumulate_add(acc.black, 384 + rook_pt * 64 + (rook_to ^ 56));
        } else {
            accumulate_sub(acc.black, king_pt * 64 + (king_from ^ 56));
            accumulate_add(acc.black, king_pt * 64 + (king_to ^ 56));
            accumulate_sub(acc.black, rook_pt * 64 + (rook_from ^ 56));
            accumulate_add(acc.black, rook_pt * 64 + (rook_to ^ 56));
            accumulate_sub(acc.white, 384 + king_pt * 64 + king_from);
            accumulate_add(acc.white, 384 + king_pt * 64 + king_to);
            accumulate_sub(acc.white, 384 + rook_pt * 64 + rook_from);
            accumulate_add(acc.white, 384 + rook_pt * 64 + rook_to);
        }
    } else if (move_type == chess::Move::ENPASSANT) {
        int cap_sq = move.to().ep_square().index();
        int pawn_pt = static_cast<int>(chess::PieceType::PAWN);

        if (mover_is_white) {
            accumulate_sub(acc.white, pawn_pt * 64 + from_sq);
            accumulate_add(acc.white, pawn_pt * 64 + to_sq);
            accumulate_sub(acc.white, 384 + pawn_pt * 64 + cap_sq);
            accumulate_sub(acc.black, 384 + pawn_pt * 64 + (from_sq ^ 56));
            accumulate_add(acc.black, 384 + pawn_pt * 64 + (to_sq ^ 56));
            accumulate_sub(acc.black, pawn_pt * 64 + (cap_sq ^ 56));
        } else {
            accumulate_sub(acc.black, pawn_pt * 64 + (from_sq ^ 56));
            accumulate_add(acc.black, pawn_pt * 64 + (to_sq ^ 56));
            accumulate_sub(acc.black, 384 + pawn_pt * 64 + (cap_sq ^ 56));
            accumulate_sub(acc.white, 384 + pawn_pt * 64 + from_sq);
            accumulate_add(acc.white, 384 + pawn_pt * 64 + to_sq);
            accumulate_sub(acc.white, pawn_pt * 64 + cap_sq);
        }
    } else if (move_type == chess::Move::PROMOTION) {
        int pawn_pt = static_cast<int>(chess::PieceType::PAWN);
        int promo_pt = static_cast<int>(move.promotionType());

        if (mover_is_white) {
            accumulate_sub(acc.white, pawn_pt * 64 + from_sq);
            accumulate_add(acc.white, promo_pt * 64 + to_sq);
            accumulate_sub(acc.black, 384 + pawn_pt * 64 + (from_sq ^ 56));
            accumulate_add(acc.black, 384 + promo_pt * 64 + (to_sq ^ 56));
        } else {
            accumulate_sub(acc.black, pawn_pt * 64 + (from_sq ^ 56));
            accumulate_add(acc.black, promo_pt * 64 + (to_sq ^ 56));
            accumulate_sub(acc.white, 384 + pawn_pt * 64 + from_sq);
            accumulate_add(acc.white, 384 + promo_pt * 64 + to_sq);
        }

        if (captured_piece != chess::Piece::NONE) {
            int cap_pt = static_cast<int>(captured_piece.type());
            if (mover_is_white) {
                accumulate_sub(acc.white, 384 + cap_pt * 64 + to_sq);
                accumulate_sub(acc.black, cap_pt * 64 + (to_sq ^ 56));
            } else {
                accumulate_sub(acc.black, 384 + cap_pt * 64 + (to_sq ^ 56));
                accumulate_sub(acc.white, cap_pt * 64 + to_sq);
            }
        }
    } else {
        // Normal move
        if (mover_is_white) {
            accumulate_sub(acc.white, pt * 64 + from_sq);
            accumulate_add(acc.white, pt * 64 + to_sq);
            accumulate_sub(acc.black, 384 + pt * 64 + (from_sq ^ 56));
            accumulate_add(acc.black, 384 + pt * 64 + (to_sq ^ 56));
        } else {
            accumulate_sub(acc.black, pt * 64 + (from_sq ^ 56));
            accumulate_add(acc.black, pt * 64 + (to_sq ^ 56));
            accumulate_sub(acc.white, 384 + pt * 64 + from_sq);
            accumulate_add(acc.white, 384 + pt * 64 + to_sq);
        }

        if (captured_piece != chess::Piece::NONE) {
            int cap_pt = static_cast<int>(captured_piece.type());
            if (mover_is_white) {
                accumulate_sub(acc.white, 384 + cap_pt * 64 + to_sq);
                accumulate_sub(acc.black, cap_pt * 64 + (to_sq ^ 56));
            } else {
                accumulate_sub(acc.black, 384 + cap_pt * 64 + (to_sq ^ 56));
                accumulate_sub(acc.white, cap_pt * 64 + to_sq);
            }
        }
    }

    // Castling features 768-771: diff old vs new
    int old_cr = acc.castling_hash;
    int new_cr = board_after.board.castlingRights().hashIndex();
    if (old_cr != new_cr) {
        // White perspective: bits 1,2,4,8 → features 768,769,770,771
        // Black perspective: bits 4,8,1,2 → features 768,769,770,771
        int diff = old_cr ^ new_cr;
        // White perspective mapping: bit i → feature 768 + bit_to_feat[i]
        // bit 1(0x1) → 768, bit 2(0x2) → 769, bit 4(0x4) → 770, bit 8(0x8) → 771
        static constexpr int W_FEAT[] = {768, 769, 770, 771};
        // Black perspective: bit 4→768, bit 8→769, bit 1→770, bit 2→771
        static constexpr int B_FEAT[] = {770, 771, 768, 769};
        for (int bi = 0; bi < 4; ++bi) {
            int mask = 1 << bi;
            if (!(diff & mask)) continue;
            bool was_set = old_cr & mask;
            if (was_set) {
                accumulate_sub(acc.white, W_FEAT[bi]);
                accumulate_sub(acc.black, B_FEAT[bi]);
            } else {
                accumulate_add(acc.white, W_FEAT[bi]);
                accumulate_add(acc.black, B_FEAT[bi]);
            }
        }
    }

    // EP feature (772): diff old vs new
    bool new_ep = board_after.board.enpassantSq() != chess::Square::NO_SQ;
    if (acc.has_ep && !new_ep) {
        accumulate_sub(acc.white, 772);
        accumulate_sub(acc.black, 772);
    } else if (!acc.has_ep && new_ep) {
        accumulate_add(acc.white, 772);
        accumulate_add(acc.black, 772);
    }

    acc.castling_hash = new_cr;
    acc.has_ep = new_ep;
    acc.computed = true;
}

void NNUEModel::update_accumulator_null_move(const ChessBoard& board_after) const {
    if (acc_ply < 0) return;
    auto& acc = acc_stack[acc_ply];
    // Null move: no pieces change, just EP resets. Castling unchanged.
    // Remove old EP if it was set (null move always clears EP)
    if (acc.has_ep) {
        accumulate_sub(acc.white, 772);
        accumulate_sub(acc.black, 772);
        acc.has_ep = false;
    }
    acc.computed = true;
}

float NNUEModel::forward_from_accumulator(const int16_t* h1_acc) const {
    alignas(32) float h2[HIDDEN2_SIZE];

    // Fused ClippedReLU + Layer 2: clamp h1 on-the-fly during dot product
    // (no copy, no separate clamp pass)
#ifdef __AVX2__
    {
        __m256i zero = _mm256_setzero_si256();
        __m256i qmax = _mm256_set1_epi16(static_cast<int16_t>(Q1_SCALE));
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
                __m256i h = _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(&h1_acc[i]));
                h = _mm256_max_epi16(zero, _mm256_min_epi16(qmax, h));
                s0 = _mm256_add_epi32(
                    s0, _mm256_madd_epi16(
                            h, _mm256_load_si256(
                                   reinterpret_cast<const __m256i*>(&r0[i]))));
                s1 = _mm256_add_epi32(
                    s1, _mm256_madd_epi16(
                            h, _mm256_load_si256(
                                   reinterpret_cast<const __m256i*>(&r1[i]))));
                s2 = _mm256_add_epi32(
                    s2, _mm256_madd_epi16(
                            h, _mm256_load_si256(
                                   reinterpret_cast<const __m256i*>(&r2[i]))));
                s3 = _mm256_add_epi32(
                    s3, _mm256_madd_epi16(
                            h, _mm256_load_si256(
                                   reinterpret_cast<const __m256i*>(&r3[i]))));
            }
            __m256i h01 = _mm256_hadd_epi32(s0, s1);
            __m256i h23 = _mm256_hadd_epi32(s2, s3);
            __m256i h0123 = _mm256_hadd_epi32(h01, h23);
            __m128i lo = _mm256_castsi256_si128(h0123);
            __m128i hi = _mm256_extracti128_si256(h0123, 1);
            __m128i sums = _mm_add_epi32(lo, hi);
            __m128 f = _mm_cvtepi32_ps(sums);
            f = _mm_add_ps(_mm_mul_ps(f, _mm_set1_ps(DEQUANT_SCALE)),
                           _mm_loadu_ps(b2.get() + j));
            f = _mm_max_ps(_mm_setzero_ps(), _mm_min_ps(_mm_set1_ps(1.0f), f));
            _mm_storeu_ps(h2 + j, f);
        }
    }
#else
    {
        __m128i zero = _mm_setzero_si128();
        __m128i qmax = _mm_set1_epi16(static_cast<int16_t>(Q1_SCALE));
        for (int j = 0; j < HIDDEN2_SIZE; ++j) {
            __m128i sum_vec = _mm_setzero_si128();
            const int16_t* row = w2_q.get() + j * H1_PADDED;
            for (int i = 0; i < H1_PADDED; i += 8) {
                __m128i h = _mm_load_si128(
                    reinterpret_cast<const __m128i*>(&h1_acc[i]));
                h = _mm_max_epi16(zero, _mm_min_epi16(qmax, h));
                __m128i w = _mm_load_si128(
                    reinterpret_cast<const __m128i*>(&row[i]));
                sum_vec = _mm_add_epi32(sum_vec, _mm_madd_epi16(h, w));
            }
            __m128i hi_s = _mm_shuffle_epi32(sum_vec, _MM_SHUFFLE(1, 0, 3, 2));
            sum_vec = _mm_add_epi32(sum_vec, hi_s);
            hi_s = _mm_shuffle_epi32(sum_vec, _MM_SHUFFLE(0, 1, 0, 1));
            sum_vec = _mm_add_epi32(sum_vec, hi_s);
            float sum =
                static_cast<float>(_mm_cvtsi128_si32(sum_vec)) * DEQUANT_SCALE +
                b2[j];
            h2[j] = std::max(0.0f, std::min(1.0f, sum));
        }
    }
#endif

    float logit;
#ifdef __AVX2__
    {
        __m256 fma_acc = _mm256_setzero_ps();
        for (int i = 0; i < HIDDEN2_SIZE; i += 8)
            fma_acc = _mm256_fmadd_ps(_mm256_load_ps(h2 + i),
                                      _mm256_loadu_ps(w3.get() + i), fma_acc);
        __m128 lo128 = _mm256_castps256_ps128(fma_acc);
        __m128 hi128 = _mm256_extractf128_ps(fma_acc, 1);
        __m128 sum128 = _mm_add_ps(lo128, hi128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        logit = _mm_cvtss_f32(sum128) + b3[0];
    }
#else
    logit = b3[0];
    for (int i = 0; i < HIDDEN2_SIZE; ++i) logit += h2[i] * w3[i];
#endif

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
    return x * MATE_VALUE;
}

float NNUEModel::predict_from_accumulator(const ChessBoard& board) const {
    if (!loaded || acc_ply < 0) return predict(board);

    const auto& acc = acc_stack[acc_ply];
    bool white_to_move = board.turn() == ChessBoard::WHITE;
    const int16_t* stm_acc = white_to_move ? acc.white : acc.black;

    float stm_eval = forward_from_accumulator(stm_acc);
    return white_to_move ? stm_eval : -stm_eval;
}
