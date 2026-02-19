#include "opening_book.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <random>

// Polyglot random values for en passant files (indices 772-779 of RANDOM_ARRAY)
static constexpr uint64_t POLYGLOT_EP_KEYS[8] = {
    0x70CC73D90BC26E24, 0xE21A6B35DF0C3AD7, 0x003A93D8B2806962, 0x1C99DED33CB890A1,
    0xCF3145DE0ADD4289, 0xD0E4427A5514FB72, 0x77C621CC9FB3A483, 0x67A34DAC4356550B,
};

bool OpeningBook::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    return load(file);
}

bool OpeningBook::load(std::istream& stream) {
    char magic[4];
    stream.read(magic, 4);
    if (std::memcmp(magic, "BOOK", 4) != 0) return false;

    uint32_t version, num_positions, num_moves;
    stream.read(reinterpret_cast<char*>(&version), 4);
    stream.read(reinterpret_cast<char*>(&num_positions), 4);
    stream.read(reinterpret_cast<char*>(&num_moves), 4);

    if (version != 1) return false;

    positions_.resize(num_positions);
    stream.read(reinterpret_cast<char*>(positions_.data()),
                num_positions * sizeof(Position));

    moves_.resize(num_moves);
    stream.read(reinterpret_cast<char*>(moves_.data()), num_moves * sizeof(BookMove));

    if (!stream) {
        positions_.clear();
        moves_.clear();
        return false;
    }

    loaded_ = true;
    return true;
}

uint64_t OpeningBook::polyglot_hash(const chess::Board& board) {
    uint64_t key = board.hash();

    auto ep = board.enpassantSq();
    if (ep != chess::Square::NO_SQ) {
        // chess-library always hashes EP when set.
        // Polyglot only hashes EP when a pseudo-legal capture exists.
        // If no adjacent capturing pawn exists, remove the EP hash.
        auto stm = board.sideToMove();
        int ep_file = static_cast<int>(ep.file());
        // The capturing pawn sits on rank 3 (if black captures) or rank 4 (if white)
        int pawn_rank = (stm == chess::Color::BLACK) ? 3 : 4;

        auto stm_pawns = board.pieces(chess::PieceType::PAWN, stm);
        uint64_t adj_mask = 0;
        if (ep_file > 0) adj_mask |= (1ULL << (pawn_rank * 8 + ep_file - 1));
        if (ep_file < 7) adj_mask |= (1ULL << (pawn_rank * 8 + ep_file + 1));

        if (!(stm_pawns & chess::Bitboard(adj_mask))) {
            key ^= POLYGLOT_EP_KEYS[ep_file];
        }
    }

    return key;
}

const OpeningBook::Position* OpeningBook::find_position(uint64_t hash) const {
    auto it =
        std::lower_bound(positions_.begin(), positions_.end(), hash,
                         [](const Position& p, uint64_t h) { return p.hash < h; });
    if (it != positions_.end() && it->hash == hash) return &(*it);
    return nullptr;
}

chess::Move OpeningBook::to_chess_move(const chess::Board& board, const BookMove& bm) {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);

    auto from = chess::Square(bm.from);
    auto to = chess::Square(bm.to);

    for (const auto& move : moves) {
        if (move.from() != from || move.to() != to) continue;

        if (bm.promo == 0) {
            if (move.typeOf() != chess::Move::PROMOTION) return move;
        } else {
            if (move.typeOf() == chess::Move::PROMOTION) {
                chess::PieceType pt;
                switch (bm.promo) {
                    case 2:
                        pt = chess::PieceType::KNIGHT;
                        break;
                    case 3:
                        pt = chess::PieceType::BISHOP;
                        break;
                    case 4:
                        pt = chess::PieceType::ROOK;
                        break;
                    case 5:
                        pt = chess::PieceType::QUEEN;
                        break;
                    default:
                        continue;
                }
                if (move.promotionType() == pt) return move;
            }
        }
    }

    return chess::Move::NO_MOVE;
}

std::optional<chess::Move> OpeningBook::probe(const chess::Board& board) const {
    if (!loaded_) return std::nullopt;

    uint64_t hash = polyglot_hash(board);
    const Position* pos = find_position(hash);
    if (!pos || pos->count == 0) return std::nullopt;

    // Weighted random selection
    thread_local std::mt19937 rng{std::random_device{}()};

    uint32_t total_weight = 0;
    for (uint16_t i = 0; i < pos->count; ++i) {
        total_weight += moves_[pos->offset + i].weight;
    }

    if (total_weight == 0) return std::nullopt;

    std::uniform_int_distribution<uint32_t> dist(0, total_weight - 1);
    uint32_t pick = dist(rng);

    uint32_t cumulative = 0;
    for (uint16_t i = 0; i < pos->count; ++i) {
        cumulative += moves_[pos->offset + i].weight;
        if (pick < cumulative) {
            auto move = to_chess_move(board, moves_[pos->offset + i]);
            if (move != chess::Move::NO_MOVE) return move;
            break;
        }
    }

    return std::nullopt;
}
