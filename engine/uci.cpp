#include "uci.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "chess_engine.h"

UCIHandler::UCIHandler(BaseEngine* engine, std::shared_ptr<OpeningBook> book)
    : engine(engine), book(std::move(book)) {}

UCIHandler::~UCIHandler() {
    if (search_thread.joinable()) {
        engine->stop_search();
        search_thread.join();
    }
}

void UCIHandler::run() {
    std::string line;
    while (std::getline(std::cin, line)) {
        // Trim trailing \r for Windows
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;

        if (line == "uci") {
            handle_uci();
        } else if (line == "isready") {
            handle_isready();
        } else if (line == "ucinewgame") {
            handle_ucinewgame();
        } else if (line.substr(0, 8) == "position") {
            handle_position(line);
        } else if (line.substr(0, 2) == "go") {
            handle_go(line);
        } else if (line == "stop") {
            handle_stop();
        } else if (line == "quit") {
            handle_stop();
            return;
        }
        // Ignore unknown commands per UCI spec
    }
}

void UCIHandler::handle_uci() {
    std::cout << "id name ForbesChess" << std::endl;
    std::cout << "id author Forbes" << std::endl;
    std::cout << "uciok" << std::endl;
}

void UCIHandler::handle_isready() { std::cout << "readyok" << std::endl; }

void UCIHandler::handle_ucinewgame() {
    handle_stop();
    if (auto* ce = dynamic_cast<ChessEngine*>(engine)) {
        ce->clear_caches();
    }
    board = ChessBoard();
}

void UCIHandler::handle_position(const std::string& line) {
    std::istringstream ss(line);
    std::string token;
    ss >> token;  // "position"
    ss >> token;

    if (token == "startpos") {
        board = ChessBoard();
        ss >> token;  // possibly "moves"
    } else if (token == "fen") {
        std::string fen;
        // Read 6 FEN fields
        for (int i = 0; i < 6 && ss >> token; i++) {
            if (token == "moves") break;
            if (!fen.empty()) fen += " ";
            fen += token;
        }
        board = ChessBoard(fen);
        // If we broke out early on "moves", token is already "moves"
        // Otherwise read the next token
        if (token != "moves") ss >> token;
    }

    // Apply moves
    if (token == "moves") {
        while (ss >> token) {
            board.make_move(ChessBoard::Move::from_uci(token));
        }
    }
}

void UCIHandler::handle_go(const std::string& line) {
    handle_stop();

    std::istringstream ss(line);
    std::string token;
    ss >> token;  // "go"

    int wtime = 0, btime = 0, winc = 0, binc = 0;
    int movestogo = 0, movetime = 0;
    bool infinite = false;

    while (ss >> token) {
        if (token == "wtime")
            ss >> wtime;
        else if (token == "btime")
            ss >> btime;
        else if (token == "winc")
            ss >> winc;
        else if (token == "binc")
            ss >> binc;
        else if (token == "movestogo")
            ss >> movestogo;
        else if (token == "movetime")
            ss >> movetime;
        else if (token == "infinite")
            infinite = true;
    }

    TimeControl tc{};
    if (movetime > 0) {
        tc.time_left_ms = movetime;
        tc.increment_ms = 0;
        tc.moves_to_go = 1;
    } else if (infinite) {
        // Very large time budget; engine will run until "stop"
        tc.time_left_ms = 1000000;
        tc.increment_ms = 0;
        tc.moves_to_go = 0;
    } else {
        bool white_to_move = (board.turn() == ChessBoard::WHITE);
        tc.time_left_ms = white_to_move ? wtime : btime;
        tc.increment_ms = white_to_move ? winc : binc;
        tc.moves_to_go = movestogo;
    }

    searching = true;
    search_thread = std::thread(&UCIHandler::search_and_print, this, tc);
}

void UCIHandler::handle_stop() {
    if (searching) {
        engine->stop_search();
    }
    if (search_thread.joinable()) {
        search_thread.join();
    }
    searching = false;
}

void UCIHandler::search_and_print(TimeControl tc) {
    // Try opening book first
    if (book && book->is_loaded()) {
        auto book_move = book->probe(board.board);
        if (book_move) {
            std::string uci = chess::uci::moveToUci(*book_move);
            std::cout << "bestmove " << uci << std::endl;
            searching = false;
            return;
        }
    }

    auto result = engine->get_best_move(board, tc);

    std::string score_str = format_score(result.score);
    long long time_ms = result.time_used.count();

    std::cout << "info depth " << result.depth << " score " << score_str << " nodes "
              << result.nodes_searched << " time " << time_ms;
    if (time_ms > 0) {
        long long nps = static_cast<long long>(result.nodes_searched) * 1000 / time_ms;
        std::cout << " nps " << nps;
    }
    std::cout << std::endl;

    std::cout << "bestmove " << result.best_move.uci() << std::endl;
    searching = false;
}

std::string UCIHandler::format_score(float score) {
    constexpr float MATE_THRESHOLD =
        config::MATE_VALUE - config::search::MATE_THRESHOLD_MARGIN;

    if (std::abs(score) >= MATE_THRESHOLD) {
        // Convert to mate-in-N plies, then to moves
        int plies_to_mate = static_cast<int>(config::MATE_VALUE - std::abs(score));
        int moves = (plies_to_mate + 1) / 2;
        if (score < 0) moves = -moves;
        return "mate " + std::to_string(moves);
    }

    return "cp " + std::to_string(static_cast<int>(score));
}
