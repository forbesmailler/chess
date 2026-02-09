#include <gtest/gtest.h>

#include "chess_board.h"

// --- square_from_string boundary values ---

TEST(ChessBoardEdge, SquareFromStringBoundaries) {
    EXPECT_EQ(ChessBoard::square_from_string("a1"), 0);
    EXPECT_EQ(ChessBoard::square_from_string("h1"), 7);
    EXPECT_EQ(ChessBoard::square_from_string("a8"), 56);
    EXPECT_EQ(ChessBoard::square_from_string("h8"), 63);
}

TEST(ChessBoardEdge, SquareFromStringInvalid) {
    EXPECT_EQ(ChessBoard::square_from_string(""), -1);
    EXPECT_EQ(ChessBoard::square_from_string("a"), -1);
    EXPECT_EQ(ChessBoard::square_from_string("abc"), -1);
    EXPECT_EQ(ChessBoard::square_from_string("i1"), -1);  // file out of range
    EXPECT_EQ(ChessBoard::square_from_string("a0"), -1);  // rank out of range
    EXPECT_EQ(ChessBoard::square_from_string("a9"), -1);  // rank out of range
    EXPECT_EQ(ChessBoard::square_from_string("z5"), -1);  // file out of range
}

// --- square_to_string boundary values ---

TEST(ChessBoardEdge, SquareToStringBoundaries) {
    EXPECT_EQ(ChessBoard::square_to_string(0), "a1");
    EXPECT_EQ(ChessBoard::square_to_string(7), "h1");
    EXPECT_EQ(ChessBoard::square_to_string(56), "a8");
    EXPECT_EQ(ChessBoard::square_to_string(63), "h8");
}

TEST(ChessBoardEdge, SquareToStringOutOfRange) {
    EXPECT_EQ(ChessBoard::square_to_string(-1), "");
    EXPECT_EQ(ChessBoard::square_to_string(64), "");
    EXPECT_EQ(ChessBoard::square_to_string(-100), "");
    EXPECT_EQ(ChessBoard::square_to_string(100), "");
}

// --- piece_type_at boundary values ---

TEST(ChessBoardEdge, PieceTypeAtOutOfRange) {
    ChessBoard board;
    EXPECT_EQ(board.piece_type_at(-1), ChessBoard::NONE);
    EXPECT_EQ(board.piece_type_at(64), ChessBoard::NONE);
    EXPECT_EQ(board.piece_type_at(-100), ChessBoard::NONE);
    EXPECT_EQ(board.piece_type_at(100), ChessBoard::NONE);
}

TEST(ChessBoardEdge, PieceTypeAtAllStartingPieces) {
    ChessBoard board;
    // White back rank
    EXPECT_EQ(board.piece_type_at(0), ChessBoard::ROOK);    // a1
    EXPECT_EQ(board.piece_type_at(1), ChessBoard::KNIGHT);  // b1
    EXPECT_EQ(board.piece_type_at(2), ChessBoard::BISHOP);  // c1
    EXPECT_EQ(board.piece_type_at(3), ChessBoard::QUEEN);   // d1
    EXPECT_EQ(board.piece_type_at(4), ChessBoard::KING);    // e1
    EXPECT_EQ(board.piece_type_at(5), ChessBoard::BISHOP);  // f1
    EXPECT_EQ(board.piece_type_at(6), ChessBoard::KNIGHT);  // g1
    EXPECT_EQ(board.piece_type_at(7), ChessBoard::ROOK);    // h1
    // White pawns
    for (int sq = 8; sq < 16; ++sq) {
        EXPECT_EQ(board.piece_type_at(sq), ChessBoard::PAWN) << "sq=" << sq;
    }
    // Empty squares (ranks 3-6)
    for (int sq = 16; sq < 48; ++sq) {
        EXPECT_EQ(board.piece_type_at(sq), ChessBoard::NONE) << "sq=" << sq;
    }
}

TEST(ChessBoardEdge, PieceTypeAtByString) {
    ChessBoard board;
    EXPECT_EQ(board.piece_type_at("e1"), ChessBoard::KING);
    EXPECT_EQ(board.piece_type_at("e2"), ChessBoard::PAWN);
    EXPECT_EQ(board.piece_type_at("e4"), ChessBoard::NONE);
    // Invalid string delegates to square_from_string returning -1
    EXPECT_EQ(board.piece_type_at(""), ChessBoard::NONE);
    EXPECT_EQ(board.piece_type_at("z9"), ChessBoard::NONE);
}

// --- get_capture_moves ---

TEST(ChessBoardEdge, NoCapturesFromStart) {
    ChessBoard board;
    auto captures = board.get_capture_moves();
    EXPECT_EQ(captures.size(), 0u);
}

TEST(ChessBoardEdge, CapturesAvailable) {
    // Position where white pawn on e4 can capture black pawn on d5
    ChessBoard board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
    auto captures = board.get_capture_moves();
    EXPECT_GE(captures.size(), 1u);

    // All returned moves should be captures
    for (const auto& move : captures) {
        EXPECT_TRUE(board.is_capture_move(move))
            << "Non-capture in capture list: " << move.uci();
    }
}

// --- is_capture_move ---

TEST(ChessBoardEdge, IsCaptureMove) {
    // Position with captures available
    ChessBoard board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
    auto all_moves = board.get_legal_moves();

    int capture_count = 0;
    for (const auto& move : all_moves) {
        if (board.is_capture_move(move)) {
            capture_count++;
        }
    }
    // exd5 is a capture, ed6 (en passant) is a capture
    EXPECT_GE(capture_count, 1);
}

// --- is_in_check ---

TEST(ChessBoardEdge, InCheckWhite) {
    // White king in check from black queen
    ChessBoard board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    EXPECT_TRUE(board.is_in_check(ChessBoard::WHITE));
    EXPECT_FALSE(board.is_in_check(ChessBoard::BLACK));
}

TEST(ChessBoardEdge, NotInCheck) {
    ChessBoard board;
    EXPECT_FALSE(board.is_in_check(ChessBoard::WHITE));
    EXPECT_FALSE(board.is_in_check(ChessBoard::BLACK));
}

TEST(ChessBoardEdge, InCheckWrongSide) {
    // White is in check but we ask about Black's check status
    // is_in_check only works for the side to move, so asking about BLACK returns false
    ChessBoard board("rnb1kbnr/pppp1ppp/4p3/8/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    EXPECT_FALSE(board.is_in_check(ChessBoard::BLACK));
}

// --- is_draw ---

TEST(ChessBoardEdge, InsufficientMaterialDraw) {
    // King vs King — insufficient material draw
    ChessBoard board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    EXPECT_TRUE(board.is_draw());
    EXPECT_TRUE(board.is_game_over());
    EXPECT_FALSE(board.is_checkmate());
    EXPECT_FALSE(board.is_stalemate());
}

// --- make_move with invalid UCI ---

TEST(ChessBoardEdge, MakeInvalidMove) {
    ChessBoard board;
    auto invalid_move = ChessBoard::Move::from_uci("z9z9");  // nonsense square
    EXPECT_FALSE(board.make_move(invalid_move));
}

TEST(ChessBoardEdge, MakeEmptyUciMove) {
    ChessBoard board;
    auto empty_move = ChessBoard::Move::from_uci("");
    EXPECT_FALSE(board.make_move(empty_move));
}

// --- unmake with empty history ---

TEST(ChessBoardEdge, UnmakeWithNoHistory) {
    ChessBoard board;
    std::string fen_before = board.to_fen();
    auto move = ChessBoard::Move::from_uci("e2e4");
    // Unmake without ever making a move — should be a no-op
    board.unmake_move(move);
    EXPECT_EQ(board.to_fen(), fen_before);
}

// --- castling rights loss ---

TEST(ChessBoardEdge, CastlingRightsLostAfterKingMove) {
    ChessBoard board("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    auto rights = board.get_castling_rights();
    EXPECT_TRUE(rights.white_kingside);
    EXPECT_TRUE(rights.white_queenside);

    board.make_move(ChessBoard::Move::from_uci("e1e2"));
    rights = board.get_castling_rights();
    EXPECT_FALSE(rights.white_kingside);
    EXPECT_FALSE(rights.white_queenside);
    // Black still has castling rights
    EXPECT_TRUE(rights.black_kingside);
    EXPECT_TRUE(rights.black_queenside);
}

// --- Move::is_promotion ---

TEST(ChessBoardEdge, PromotionDetection) {
    // Pawn on 7th rank about to promote
    ChessBoard board("8/P7/8/8/8/8/8/4K2k w - - 0 1");
    auto moves = board.get_legal_moves();

    int promotion_count = 0;
    for (const auto& move : moves) {
        if (move.is_promotion()) promotion_count++;
    }
    // a7a8 promotes to Q, R, B, N = 4 promotions
    EXPECT_EQ(promotion_count, 4);
}

// --- Move::from() / to() ---

TEST(ChessBoardEdge, MoveFromTo) {
    ChessBoard board;
    auto moves = board.get_legal_moves();
    // Find e2e4
    for (const auto& move : moves) {
        if (move.uci() == "e2e4") {
            EXPECT_EQ(move.from(), ChessBoard::square_from_string("e2"));
            EXPECT_EQ(move.to(), ChessBoard::square_from_string("e4"));
            return;
        }
    }
    FAIL() << "e2e4 not found in legal moves";
}

// --- piece_count ---

TEST(ChessBoardEdge, PieceCountAfterCapture) {
    ChessBoard board("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2");
    EXPECT_EQ(board.piece_count(), 32);
    board.make_move(ChessBoard::Move::from_uci("e4d5"));  // exd5
    EXPECT_EQ(board.piece_count(), 31);
}

TEST(ChessBoardEdge, PieceCountKingVsKing) {
    ChessBoard board("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    EXPECT_EQ(board.piece_count(), 2);
}

// --- load_fen / to_fen round-trip ---

TEST(ChessBoardEdge, FenRoundtrip) {
    std::string fen =
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    ChessBoard board(fen);
    EXPECT_EQ(board.to_fen(), fen);
}

TEST(ChessBoardEdge, LoadFenReplacesPosition) {
    ChessBoard board;
    board.load_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    EXPECT_EQ(board.piece_count(), 2);
    EXPECT_EQ(board.turn(), ChessBoard::WHITE);
}

// --- hash ---

TEST(ChessBoardEdge, DifferentPositionsDifferentHash) {
    ChessBoard board1;
    ChessBoard board2("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    EXPECT_NE(board1.hash(), board2.hash());
}

TEST(ChessBoardEdge, SamePositionSameHash) {
    ChessBoard board1;
    ChessBoard board2;
    EXPECT_EQ(board1.hash(), board2.hash());
}
