#pragma once

#include <chess.hpp>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

class OpeningBook {
   public:
    bool load(const std::string& path);
    bool load(std::istream& stream);

    std::optional<chess::Move> probe(const chess::Board& board) const;

    bool is_loaded() const { return loaded_; }
    size_t size() const { return positions_.size(); }

    static uint64_t polyglot_hash(const chess::Board& board);

   private:
    struct Position {
        uint64_t hash;
        uint32_t offset;
        uint16_t count;
        uint16_t reserved;
    };

    struct BookMove {
        uint8_t from;
        uint8_t to;
        uint8_t promo;
        uint8_t weight;
    };

    std::vector<Position> positions_;
    std::vector<BookMove> moves_;
    bool loaded_ = false;

    const Position* find_position(uint64_t hash) const;
    static chess::Move to_chess_move(const chess::Board& board, const BookMove& bm);
};
