#include "lichess_client.h"

#include <curl/curl.h>

#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>

using json = nlohmann::json;

LichessClient::CurlGlobalInit LichessClient::curl_init;

LichessClient::CurlGlobalInit::CurlGlobalInit() { curl_global_init(CURL_GLOBAL_DEFAULT); }

LichessClient::CurlGlobalInit::~CurlGlobalInit() { curl_global_cleanup(); }

LichessClient::LichessClient(const std::string& token)
    : token(token), base_url("https://lichess.org/api") {}

LichessClient::~LichessClient() = default;

bool LichessClient::get_account_info(AccountInfo& info) {
    auto response = make_request(base_url + "/account");

    std::cout << "[DEBUG] Account info response code: " << response.status_code << std::endl;
    std::cout << "[DEBUG] Account info response: " << response.data << std::endl;

    if (response.status_code != 200) {
        std::cerr << "Failed to get account info: " << response.status_code << std::endl;
        return false;
    }

    try {
        auto j = json::parse(response.data);
        info.id = j["id"];
        info.username = j["username"];
        info.is_bot = j.value("title", "") == "BOT";
        info.title = j.value("title", "");

        std::cout << "[DEBUG] Account: " << info.username << " (ID: " << info.id << ")"
                  << std::endl;
        std::cout << "[DEBUG] Is bot: " << (info.is_bot ? "YES" : "NO") << std::endl;
        std::cout << "[DEBUG] Title: " << info.title << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing account info: " << e.what() << std::endl;
        return false;
    }
}

bool LichessClient::accept_challenge(const std::string& challenge_id) {
    auto response = make_request(base_url + "/challenge/" + challenge_id + "/accept", "POST");
    if (response.status_code != 200) {
        std::cout << "Challenge accept failed with status " << response.status_code << ": "
                  << response.data << std::endl;
    }
    return response.status_code == 200;
}

bool LichessClient::make_move(const std::string& game_id, const std::string& uci_move) {
    auto response = make_request(base_url + "/bot/game/" + game_id + "/move/" + uci_move, "POST");
    if (response.status_code != 200) {
        std::cout << "Make move failed with status " << response.status_code << ": "
                  << response.data << std::endl;
    }
    return response.status_code == 200;
}

bool LichessClient::accept_draw(const std::string& game_id) {
    auto response = make_request(base_url + "/bot/game/" + game_id + "/draw/yes", "POST");
    return response.status_code == 200;
}

bool LichessClient::decline_draw(const std::string& game_id) {
    auto response = make_request(base_url + "/bot/game/" + game_id + "/draw/no", "POST");
    return response.status_code == 200;
}

bool LichessClient::offer_draw(const std::string& game_id) {
    auto response = make_request(base_url + "/bot/game/" + game_id + "/draw/yes", "POST");
    return response.status_code == 200;
}

bool LichessClient::test_connectivity() {
    std::cout << "[DEBUG] Testing basic connectivity..." << std::endl;

    // First test basic HTTP connectivity to Google
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cout << "[DEBUG] CURL init failed" << std::endl;
        return false;
    }

    std::string response_data;
    curl_easy_setopt(curl, CURLOPT_URL, "http://www.google.com");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);  // HEAD request only

    CURLcode res = curl_easy_perform(curl);
    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        std::cout << "[DEBUG] Basic connectivity test failed: " << curl_easy_strerror(res)
                  << std::endl;
        return false;
    }

    std::cout << "[DEBUG] Basic connectivity OK, testing Lichess..." << std::endl;

    // Now test Lichess connectivity
    auto response = make_request("https://lichess.org/api");
    if (response.status_code == 0) {
        std::cout << "[DEBUG] Lichess connectivity test failed" << std::endl;
        return false;
    }

    std::cout << "[DEBUG] Lichess connectivity OK (status: " << response.status_code << ")"
              << std::endl;
    return true;
}

void LichessClient::stream_events(std::function<void(const GameEvent&)> callback) {
    stream_lines(base_url + "/stream/event", [callback](const std::string& line) {
        if (line.empty()) return;

        // Debug: log all incoming events
        std::cout << "[DEBUG] Received event line: " << line << std::endl;

        try {
            auto j = json::parse(line);
            GameEvent event;
            event.type = j["type"];

            std::cout << "[DEBUG] Parsed event type: " << event.type << std::endl;

            if (event.type == "challenge") {
                event.challenge_id = j["challenge"]["id"];
                std::cout << "[DEBUG] Challenge ID: " << event.challenge_id << std::endl;
            } else if (event.type == "gameStart") {
                event.game_id = j["game"]["id"];
                std::cout << "[DEBUG] Game ID: " << event.game_id << std::endl;
            }

            callback(event);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing event: " << e.what() << std::endl;
            std::cerr << "Raw line: " << line << std::endl;
        }
    });
}

void LichessClient::stream_game(const std::string& game_id,
                                std::function<void(const GameEvent&)> callback) {
    stream_lines(base_url + "/bot/game/stream/" + game_id, [callback](const std::string& line) {
        if (line.empty()) return;

        try {
            auto j = json::parse(line);
            GameEvent event;
            event.type = j["type"];

            if (event.type == "gameState") {
                event.moves = j.value("moves", "");
                event.status = j.value("status", "started");
                event.draw_offer = j.value("wdraw", false) || j.value("bdraw", false);
                event.wtime = j.value("wtime", 0);
                event.btime = j.value("btime", 0);
                event.winc = j.value("winc", 0);
                event.binc = j.value("binc", 0);
            } else if (event.type == "gameFull") {
                if (j.contains("state")) {
                    event.moves = j["state"].value("moves", "");
                    event.status = j["state"].value("status", "started");
                    event.draw_offer =
                        j["state"].value("wdraw", false) || j["state"].value("bdraw", false);
                    event.wtime = j["state"].value("wtime", 0);
                    event.btime = j["state"].value("btime", 0);
                    event.winc = j["state"].value("winc", 0);
                    event.binc = j["state"].value("binc", 0);
                }
                if (j.contains("white")) {
                    event.white_id = j["white"].value("id", "");
                }
                if (j.contains("black")) {
                    event.black_id = j["black"].value("id", "");
                }
            }

            callback(event);
        } catch (const std::exception& e) {
            std::cerr << "Error parsing game event: " << e.what() << std::endl;
        }
    });
}

// CURL callback for writing response data
size_t LichessClient::write_callback(void* contents, size_t size, size_t nmemb, std::string* data) {
    size_t total_size = size * nmemb;
    data->append(static_cast<char*>(contents), total_size);
    return total_size;
}

// CURL callback for streaming data line by line
size_t LichessClient::stream_callback(void* contents, size_t size, size_t nmemb,
                                      StreamData* stream_data) {
    size_t total_size = size * nmemb;
    std::string chunk(static_cast<char*>(contents), total_size);

    stream_data->buffer += chunk;

    // Process complete lines
    size_t pos = 0;
    while ((pos = stream_data->buffer.find('\n')) != std::string::npos) {
        std::string line = stream_data->buffer.substr(0, pos);
        // Remove carriage return if present
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        stream_data->callback(line);
        stream_data->buffer.erase(0, pos + 1);
    }

    return total_size;
}

LichessClient::HttpResponse LichessClient::make_request(const std::string& url,
                                                        const std::string& method,
                                                        const std::string& data, bool stream) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cout << "CURL init failed" << std::endl;
        return {"", 0};
    }

    std::string response_data;
    long response_code = 0;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);

    // Set timeouts - longer for better reliability
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);         // 30 seconds total timeout
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);  // 10 seconds to connect

    // Enable keep-alive for regular requests too
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);

    // Set User-Agent
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Lichess-Bot-CPP/1.0");

    // Set headers
    struct curl_slist* headers = nullptr;
    std::string auth_header = "Authorization: Bearer " + token;
    headers = curl_slist_append(headers, auth_header.c_str());
    headers = curl_slist_append(headers, "Accept: application/json");

    if (method == "POST") {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        if (!data.empty()) {
            headers = curl_slist_append(headers, "Content-Type: application/json");
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, data.c_str());
        } else {
            // For empty POST requests, set content length to 0
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, "");
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, 0L);
        }
    }

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // Follow redirects
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);

    // Perform the request
    CURLcode res = curl_easy_perform(curl);

    if (res == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    } else {
        std::cout << "[ERROR] CURL error for " << url << ": " << curl_easy_strerror(res)
                  << std::endl;
        response_code = 0;  // Set response code to 0 on error
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return {response_data, static_cast<int>(response_code)};
}

void LichessClient::stream_lines(const std::string& url,
                                 std::function<void(const std::string&)> callback) {
    std::cout << "[DEBUG] Starting stream to URL: " << url << std::endl;

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cout << "[DEBUG] CURL init failed for streaming" << std::endl;
        return;
    }

    StreamData stream_data;
    stream_data.callback = callback;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stream_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream_data);

    // Set longer timeouts for streaming connections
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);           // No timeout for streaming
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);   // 30 seconds to connect
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT, 1L);   // Minimum 1 byte/sec
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, 300L);  // For 5 minutes

    // Enable keep-alive
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE, 120L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPINTVL, 60L);

    // Set headers
    struct curl_slist* headers = nullptr;
    std::string auth_header = "Authorization: Bearer " + token;
    headers = curl_slist_append(headers, auth_header.c_str());
    headers = curl_slist_append(headers, "Accept: text/plain");
    headers = curl_slist_append(headers, "Cache-Control: no-cache");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    // Follow redirects
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    std::cout << "[DEBUG] Starting CURL request for streaming..." << std::endl;

    // Perform the request
    CURLcode res = curl_easy_perform(curl);

    std::cout << "[DEBUG] CURL stream finished with result: " << curl_easy_strerror(res)
              << std::endl;

    // Check HTTP response code
    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    std::cout << "[DEBUG] Stream HTTP response code: " << response_code << std::endl;

    // Process any remaining data in buffer
    if (!stream_data.buffer.empty()) {
        // Remove final carriage return if present
        if (stream_data.buffer.back() == '\r') {
            stream_data.buffer.pop_back();
        }
        stream_data.callback(stream_data.buffer);
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
}
