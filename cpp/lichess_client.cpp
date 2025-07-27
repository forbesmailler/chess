#include "lichess_client.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <sstream>
#include <thread>

using json = nlohmann::json;

LichessClient::CurlGlobalInit LichessClient::curl_init;

LichessClient::CurlGlobalInit::CurlGlobalInit() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

LichessClient::CurlGlobalInit::~CurlGlobalInit() {
    curl_global_cleanup();
}

LichessClient::LichessClient(const std::string& token) 
    : token(token), base_url("https://lichess.org/api") {}

LichessClient::~LichessClient() = default;

bool LichessClient::get_account_info(AccountInfo& info) {
    auto response = make_request(base_url + "/account");
    
    if (response.status_code != 200) {
        std::cerr << "Failed to get account info: " << response.status_code << std::endl;
        return false;
    }
    
    try {
        auto j = json::parse(response.data);
        info.id = j["id"];
        info.username = j["username"];
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing account info: " << e.what() << std::endl;
        return false;
    }
}

bool LichessClient::accept_challenge(const std::string& challenge_id) {
    std::string url = base_url + "/challenge/" + challenge_id + "/accept";
    auto response = make_request(url, "POST");
    
    return response.status_code == 200;
}

bool LichessClient::make_move(const std::string& game_id, const std::string& uci_move) {
    std::string url = base_url + "/bot/game/" + game_id + "/move/" + uci_move;
    
    // Retry logic similar to Python version
    const int max_attempts = 5;
    for (int attempt = 1; attempt <= max_attempts; attempt++) {
        auto response = make_request(url, "POST");
        
        if (response.status_code == 200) {
            if (attempt > 1) {
                std::cout << "Game " << game_id << ": succeeded on attempt " << attempt << std::endl;
            }
            return true;
        }
        
        std::cerr << "Attempt " << attempt << "/" << max_attempts 
                  << " - Game " << game_id << ": could not play move " << uci_move 
                  << " (status: " << response.status_code << ")" << std::endl;
        
        if (attempt < max_attempts) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    std::cerr << "Game " << game_id << ": failed to play move " << uci_move 
              << " after " << max_attempts << " attempts" << std::endl;
    return false;
}

void LichessClient::stream_events(std::function<void(const GameEvent&)> callback) {
    std::string url = base_url + "/stream/event";
    
    auto line_processor = [callback](const std::string& line) {
        if (line.empty()) return;
        
        try {
            auto j = json::parse(line);
            GameEvent event;
            event.type = j["type"];
            
            if (event.type == "challenge") {
                event.challenge_id = j["challenge"]["id"];
            } else if (event.type == "gameStart") {
                event.game_id = j["game"]["id"];
            }
            
            callback(event);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing event: " << e.what() << std::endl;
        }
    };
    
    make_request(url, "GET", "", true); // This needs to be implemented as streaming
}

void LichessClient::stream_game(const std::string& game_id, std::function<void(const GameEvent&)> callback) {
    std::string url = base_url + "/bot/game/stream/" + game_id;
    
    auto line_processor = [callback, game_id](const std::string& line) {
        if (line.empty()) return;
        
        try {
            auto j = json::parse(line);
            GameEvent event;
            event.game_id = game_id;
            event.type = j["type"];
            
            if (event.type == "gameFull") {
                event.moves = j["state"]["moves"];
                event.status = j["state"]["status"];
                // Determine color - this would need more logic
            } else if (event.type == "gameState") {
                event.moves = j["moves"];
                event.status = j["status"];
            }
            
            callback(event);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing game event: " << e.what() << std::endl;
        }
    };
    
    make_request(url, "GET", "", true); // This needs to be implemented as streaming
}

LichessClient::HttpResponse LichessClient::make_request(const std::string& url, 
                                                       const std::string& method,
                                                       const std::string& data,
                                                       bool stream) {
    CURL* curl = curl_easy_init();
    HttpResponse response;
    
    if (!curl) {
        response.status_code = 0;
        return response;
    }
    
    // Set headers
    struct curl_slist* headers = nullptr;
    std::string auth_header = "Authorization: Bearer " + token;
    headers = curl_slist_append(headers, auth_header.c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response.data);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    
    if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        if (!data.empty()) {
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        }
    }
    
    CURLcode res = curl_easy_perform(curl);
    
    if (res == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.status_code);
    } else {
        response.status_code = 0;
        std::cerr << "CURL error: " << curl_easy_strerror(res) << std::endl;
    }
    
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    
    return response;
}

size_t LichessClient::write_callback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t total_size = size * nmemb;
    data->append(static_cast<char*>(contents), total_size);
    return total_size;
}

size_t LichessClient::stream_callback(void* contents, size_t size, size_t nmemb,
                                    std::function<void(const std::string&)>* callback) {
    size_t total_size = size * nmemb;
    std::string line(static_cast<char*>(contents), total_size);
    (*callback)(line);
    return total_size;
}
