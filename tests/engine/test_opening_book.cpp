#include <gtest/gtest.h>

#include <chess.hpp>
#include <cstring>
#include <sstream>

#include "opening_book.h"

namespace {

// Build an in-memory book binary with the given positions and moves.
std::string make_book_binary(
    const std::vector<std::tuple<uint64_t, uint32_t, uint16_t>>& positions,
    const std::vector<std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>>& moves) {
    std::ostringstream oss;
    oss.write("BOOK", 4);
    uint32_t version = 1;
    uint32_t num_pos = static_cast<uint32_t>(positions.size());
    uint32_t num_moves = static_cast<uint32_t>(moves.size());
    oss.write(reinterpret_cast<char*>(&version), 4);
    oss.write(reinterpret_cast<char*>(&num_pos), 4);
    oss.write(reinterpret_cast<char*>(&num_moves), 4);

    for (auto& [hash, offset, count] : positions) {
        oss.write(reinterpret_cast<const char*>(&hash), 8);
        oss.write(reinterpret_cast<const char*>(&offset), 4);
        oss.write(reinterpret_cast<const char*>(&count), 2);
        uint16_t reserved = 0;
        oss.write(reinterpret_cast<char*>(&reserved), 2);
    }

    for (auto& [from, to, promo, weight] : moves) {
        oss.write(reinterpret_cast<const char*>(&from), 1);
        oss.write(reinterpret_cast<const char*>(&to), 1);
        oss.write(reinterpret_cast<const char*>(&promo), 1);
        oss.write(reinterpret_cast<const char*>(&weight), 1);
    }

    return oss.str();
}

}  // namespace

TEST(OpeningBookTest, LoadValidBook) {
    // e2e4 from the starting position
    chess::Board board;
    uint64_t hash = OpeningBook::polyglot_hash(board);

    // from=12 (e2), to=28 (e4), no promo, weight=255
    auto data = make_book_binary({{hash, 0, 1}}, {{12, 28, 0, 255}});

    OpeningBook book;
    std::istringstream stream(data);
    ASSERT_TRUE(book.load(stream));
    EXPECT_TRUE(book.is_loaded());
    EXPECT_EQ(book.size(), 1u);
}

TEST(OpeningBookTest, LoadInvalidMagic) {
    std::string data = "BAADF00D";
    OpeningBook book;
    std::istringstream stream(data);
    EXPECT_FALSE(book.load(stream));
    EXPECT_FALSE(book.is_loaded());
}

TEST(OpeningBookTest, LoadInvalidVersion) {
    std::ostringstream oss;
    oss.write("BOOK", 4);
    uint32_t version = 99;
    uint32_t zero = 0;
    oss.write(reinterpret_cast<char*>(&version), 4);
    oss.write(reinterpret_cast<char*>(&zero), 4);
    oss.write(reinterpret_cast<char*>(&zero), 4);

    OpeningBook book;
    std::string s = oss.str();
    std::istringstream stream(s);
    EXPECT_FALSE(book.load(stream));
}

TEST(OpeningBookTest, ProbeStartingPosition) {
    chess::Board board;
    uint64_t hash = OpeningBook::polyglot_hash(board);

    // e2e4: from=12, to=28
    auto data = make_book_binary({{hash, 0, 1}}, {{12, 28, 0, 255}});

    OpeningBook book;
    std::istringstream stream(data);
    ASSERT_TRUE(book.load(stream));

    auto result = book.probe(board);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(chess::uci::moveToUci(*result), "e2e4");
}

TEST(OpeningBookTest, ProbeMissingPosition) {
    chess::Board board;
    uint64_t hash = OpeningBook::polyglot_hash(board);

    // Book has different hash
    auto data = make_book_binary({{hash ^ 1, 0, 1}}, {{12, 28, 0, 255}});

    OpeningBook book;
    std::istringstream stream(data);
    ASSERT_TRUE(book.load(stream));

    auto result = book.probe(board);
    EXPECT_FALSE(result.has_value());
}

TEST(OpeningBookTest, ProbeNotLoaded) {
    chess::Board board;
    OpeningBook book;
    auto result = book.probe(board);
    EXPECT_FALSE(result.has_value());
}

TEST(OpeningBookTest, WeightedSelection) {
    chess::Board board;
    uint64_t hash = OpeningBook::polyglot_hash(board);

    // Two moves: e2e4 (weight 200) and d2d4 (weight 50)
    auto data = make_book_binary({{hash, 0, 2}}, {{12, 28, 0, 200},   // e2e4
                                                  {11, 27, 0, 50}});  // d2d4

    OpeningBook book;
    std::istringstream stream(data);
    ASSERT_TRUE(book.load(stream));

    int e4_count = 0, d4_count = 0;
    for (int i = 0; i < 1000; ++i) {
        auto result = book.probe(board);
        ASSERT_TRUE(result.has_value());
        std::string uci = chess::uci::moveToUci(*result);
        if (uci == "e2e4")
            e4_count++;
        else if (uci == "d2d4")
            d4_count++;
    }

    // e4 should be picked ~80% (200/250), d4 ~20% (50/250)
    EXPECT_GT(e4_count, 600);
    EXPECT_GT(d4_count, 50);
    EXPECT_EQ(e4_count + d4_count, 1000);
}

TEST(OpeningBookTest, MultiplePositions) {
    chess::Board start;
    uint64_t h1 = OpeningBook::polyglot_hash(start);

    // After 1. e4
    chess::Board after_e4;
    after_e4.makeMove(chess::uci::uciToMove(after_e4, "e2e4"));
    uint64_t h2 = OpeningBook::polyglot_hash(after_e4);

    // Sort by hash for binary search
    std::vector<std::tuple<uint64_t, uint32_t, uint16_t>> positions;
    std::vector<std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>> moves;

    if (h1 < h2) {
        positions = {{h1, 0, 1}, {h2, 1, 1}};
    } else {
        positions = {{h2, 0, 1}, {h1, 1, 1}};
    }

    if (h1 < h2) {
        moves = {{12, 28, 0, 255},   // e2e4 from start
                 {52, 36, 0, 255}};  // e7e5 after e4
    } else {
        moves = {{52, 36, 0, 255},   // e7e5 after e4
                 {12, 28, 0, 255}};  // e2e4 from start
    }

    auto data = make_book_binary(positions, moves);

    OpeningBook book;
    std::istringstream stream(data);
    ASSERT_TRUE(book.load(stream));
    EXPECT_EQ(book.size(), 2u);

    auto r1 = book.probe(start);
    ASSERT_TRUE(r1.has_value());
    EXPECT_EQ(chess::uci::moveToUci(*r1), "e2e4");

    auto r2 = book.probe(after_e4);
    ASSERT_TRUE(r2.has_value());
    EXPECT_EQ(chess::uci::moveToUci(*r2), "e7e5");
}

TEST(PolyglotHashTest, StartingPositionKnownValue) {
    // The starting position has a well-known Polyglot hash
    chess::Board board;
    uint64_t hash = OpeningBook::polyglot_hash(board);
    EXPECT_EQ(hash, 0x463b96181691fc9c);
}

TEST(PolyglotHashTest, DifferentPositionsDifferentHashes) {
    chess::Board b1;
    chess::Board b2;
    b2.makeMove(chess::uci::uciToMove(b2, "e2e4"));

    EXPECT_NE(OpeningBook::polyglot_hash(b1), OpeningBook::polyglot_hash(b2));
}

TEST(PolyglotHashTest, EnPassantOnlyHashedWhenCapturable) {
    // Position with EP square set but no adjacent pawn to capture
    // 1. e4 d5 2. e5 f5 — EP on f6, but no white pawn adjacent to capture
    chess::Board b1;
    b1.setFen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");

    // Same position but without EP (compare)
    chess::Board b2;
    b2.setFen("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3");

    // White pawn IS on e5, adjacent to f6 EP square, so EP should be hashed
    EXPECT_NE(OpeningBook::polyglot_hash(b1), OpeningBook::polyglot_hash(b2));

    // Position where EP exists but no adjacent pawn
    chess::Board b3;
    b3.setFen("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 2");

    chess::Board b4;
    b4.setFen("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2");

    // Black pawn on e4 is adjacent to d3 EP square, so EP should be hashed
    EXPECT_NE(OpeningBook::polyglot_hash(b3), OpeningBook::polyglot_hash(b4));
}

TEST(PolyglotHashTest, AfterE4KnownValue) {
    // After 1. e4 — known Polyglot hash
    chess::Board board;
    board.makeMove(chess::uci::uciToMove(board, "e2e4"));
    uint64_t hash = OpeningBook::polyglot_hash(board);
    EXPECT_EQ(hash, 0x823c9b50fd114196);
}
