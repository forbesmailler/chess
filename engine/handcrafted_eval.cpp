#include "handcrafted_eval.h"

#include <algorithm>
#include <cmath>

#include "generated_config.h"

// Mirror square vertically (for black PST lookups)
static constexpr int mirror(int sq) { return sq ^ 56; }

static constexpr int rank_of(int sq) { return sq / 8; }
static constexpr int file_of(int sq) { return sq % 8; }

float handcrafted_evaluate(const ChessBoard& board) {
    if (board.is_checkmate())
        return board.turn() == ChessBoard::WHITE ? -config::MATE_VALUE
                                                 : config::MATE_VALUE;
    if (board.is_stalemate() || board.is_draw()) return 0.0f;

    int mg_score = 0;
    int eg_score = 0;
    int phase = 0;
    int white_bishops = 0;
    int black_bishops = 0;

    // Pawn file tracking for structure analysis
    // pawn_files[color][file] = count of pawns on that file
    int pawn_files[2][8] = {};
    // pawn_rank_min/max[color][file] = min/max rank of pawns on that file
    // (for passed pawn detection)
    int pawn_rank_min[2][8];
    int pawn_rank_max[2][8];
    for (int f = 0; f < 8; ++f) {
        pawn_rank_min[0][f] = 8;
        pawn_rank_min[1][f] = 8;
        pawn_rank_max[0][f] = -1;
        pawn_rank_max[1][f] = -1;
    }

    int king_sq[2] = {-1, -1};

    // First pass: collect pawn structure data and king squares
    const auto& b = board.board;
    for (int color = 0; color < 2; ++color) {
        auto c = color == 0 ? chess::Color::WHITE : chess::Color::BLACK;

        // King square
        king_sq[color] = b.kingSq(c).index();

        // Pawn locations
        auto pawns = b.pieces(chess::PieceType::PAWN, c);
        while (pawns) {
            int sq = static_cast<int>(pawns.pop());
            int f = file_of(sq);
            int r = rank_of(sq);
            pawn_files[color][f]++;
            pawn_rank_min[color][f] = std::min(pawn_rank_min[color][f], r);
            pawn_rank_max[color][f] = std::max(pawn_rank_max[color][f], r);
        }
    }

    // Evaluate each piece
    static constexpr chess::PieceType PIECE_TYPES[] = {
        chess::PieceType::PAWN, chess::PieceType::KNIGHT, chess::PieceType::BISHOP,
        chess::PieceType::ROOK, chess::PieceType::QUEEN,  chess::PieceType::KING};

    for (int color = 0; color < 2; ++color) {
        auto c = color == 0 ? chess::Color::WHITE : chess::Color::BLACK;
        int sign = color == 0 ? 1 : -1;

        for (int pt = 0; pt < 6; ++pt) {
            auto pieces = b.pieces(PIECE_TYPES[pt], c);
            while (pieces) {
                int sq = static_cast<int>(pieces.pop());
                int pst_sq = color == 0 ? sq : mirror(sq);

                // Material
                mg_score += sign * config::eval::MATERIAL_MG[pt];
                eg_score += sign * config::eval::MATERIAL_EG[pt];

                // PST
                mg_score += sign * config::eval::PST_MG[pt][pst_sq];
                eg_score += sign * config::eval::PST_EG[pt][pst_sq];

                // Phase
                phase += config::eval::PHASE_WEIGHT[pt];

                // Bishop pair tracking
                if (pt == 2) {  // BISHOP
                    if (color == 0)
                        white_bishops++;
                    else
                        black_bishops++;
                }

                // Pawn structure bonuses
                if (pt == 0) {  // PAWN
                    int f = file_of(sq);
                    int r = rank_of(sq);

                    // Passed pawn: no enemy pawns on same or adjacent files
                    // that can block or capture
                    bool passed = true;
                    int enemy = 1 - color;
                    for (int af = std::max(0, f - 1); af <= std::min(7, f + 1); ++af) {
                        if (pawn_files[enemy][af] > 0) {
                            if (color == 0) {
                                // White pawn: enemy pawn must be on higher rank
                                if (pawn_rank_max[enemy][af] > r) passed = false;
                            } else {
                                if (pawn_rank_min[enemy][af] < r) passed = false;
                            }
                        }
                    }
                    if (passed) {
                        int dist = color == 0 ? r : (7 - r);
                        int bonus = config::eval::pawn_structure::PASSED_BASE +
                                    dist * dist *
                                        config::eval::pawn_structure::PASSED_RANK_SCALE;
                        mg_score +=
                            sign *
                            (bonus / config::eval::pawn_structure::PASSED_MG_DIVISOR);
                        eg_score += sign * bonus;
                    }

                    // Isolated pawn: no friendly pawns on adjacent files
                    bool isolated = true;
                    if (f > 0 && pawn_files[color][f - 1] > 0) isolated = false;
                    if (f < 7 && pawn_files[color][f + 1] > 0) isolated = false;
                    if (isolated) {
                        mg_score -= sign * config::eval::pawn_structure::ISOLATED_MG;
                        eg_score -= sign * config::eval::pawn_structure::ISOLATED_EG;
                    }

                    // Doubled pawn: more than one pawn on this file
                    if (pawn_files[color][f] > 1) {
                        mg_score -= sign * config::eval::pawn_structure::DOUBLED_MG;
                        eg_score -= sign * config::eval::pawn_structure::DOUBLED_EG;
                    }
                }

                // Rook on open/semi-open file
                if (pt == 3) {  // ROOK
                    int f = file_of(sq);
                    bool own_pawns = pawn_files[color][f] > 0;
                    bool enemy_pawns = pawn_files[1 - color][f] > 0;
                    if (!own_pawns && !enemy_pawns) {
                        mg_score += sign * config::eval::rook_file::OPEN_MG;
                        eg_score += sign * config::eval::rook_file::OPEN_EG;
                    } else if (!own_pawns) {
                        mg_score += sign * config::eval::rook_file::SEMI_OPEN_MG;
                        eg_score += sign * config::eval::rook_file::SEMI_OPEN_EG;
                    }
                }

                // Mobility (for non-pawn, non-king pieces)
                if (pt >= 1 && pt <= 4) {
                    chess::Bitboard attacks;
                    auto sq_typed = static_cast<chess::Square>(sq);
                    auto occ = b.occ();
                    switch (pt) {
                        case 1:
                            attacks = chess::attacks::knight(sq_typed);
                            break;
                        case 2:
                            attacks = chess::attacks::bishop(sq_typed, occ);
                            break;
                        case 3:
                            attacks = chess::attacks::rook(sq_typed, occ);
                            break;
                        case 4:
                            attacks = chess::attacks::queen(sq_typed, occ);
                            break;
                    }
                    // Don't count squares occupied by own pieces
                    auto own_pieces = b.us(c);
                    attacks &= ~own_pieces;
                    int mob = attacks.count();
                    mg_score += sign * mob * config::eval::MOBILITY_BONUS[pt];
                    eg_score += sign * mob * config::eval::MOBILITY_BONUS[pt];
                }
            }
        }
    }

    // Bishop pair bonus
    if (white_bishops >= 2) {
        mg_score += config::eval::bishop_pair::BONUS_MG;
        eg_score += config::eval::bishop_pair::BONUS_EG;
    }
    if (black_bishops >= 2) {
        mg_score -= config::eval::bishop_pair::BONUS_MG;
        eg_score -= config::eval::bishop_pair::BONUS_EG;
    }

    // King pawn shield
    for (int color = 0; color < 2; ++color) {
        int sign = color == 0 ? 1 : -1;
        int ksq = king_sq[color];
        int kf = file_of(ksq);
        int kr = rank_of(ksq);

        int shield = 0;
        // Check pawns on 2nd/3rd rank near king
        int shield_rank1 = color == 0 ? kr + 1 : kr - 1;
        int shield_rank2 = color == 0 ? kr + 2 : kr - 2;

        for (int f = std::max(0, kf - 1); f <= std::min(7, kf + 1); ++f) {
            if (pawn_files[color][f] > 0) {
                int pr_min = pawn_rank_min[color][f];
                int pr_max = pawn_rank_max[color][f];
                int closest = color == 0 ? pr_min : pr_max;
                if (closest == shield_rank1 || closest == shield_rank2) {
                    shield++;
                }
            }
        }
        mg_score += sign * shield * config::eval::king_safety::SHIELD_BONUS_MG;
    }

    // Tapered eval
    if (phase > config::eval::TOTAL_PHASE) phase = config::eval::TOTAL_PHASE;
    int mg_phase = phase;
    int eg_phase = config::eval::TOTAL_PHASE - phase;

    float eval = static_cast<float>(mg_score * mg_phase + eg_score * eg_phase) /
                 config::eval::TOTAL_PHASE;

    // Scale to NNUE-compatible range: [-MATE_VALUE, +MATE_VALUE]
    float scaled =
        (2.0f / (1.0f + std::exp(-eval / config::eval::SIGMOID_SCALE)) - 1.0f) *
        config::MATE_VALUE;
    return scaled;
}
