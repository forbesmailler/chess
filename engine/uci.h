#pragma once

#include <memory>
#include <string>
#include <thread>

#include "base_engine.h"
#include "chess_board.h"
#include "opening_book.h"

class UCIHandler {
   public:
    UCIHandler(BaseEngine* engine, std::shared_ptr<OpeningBook> book = nullptr);
    ~UCIHandler();

    void run();

   private:
    BaseEngine* engine;
    std::shared_ptr<OpeningBook> book;
    ChessBoard board;

    std::thread search_thread;
    bool searching = false;

    void handle_uci();
    void handle_isready();
    void handle_ucinewgame();
    void handle_position(const std::string& line);
    void handle_go(const std::string& line);
    void handle_stop();

    void search_and_print(TimeControl tc);
    std::string format_score(float score);
};
