#include "lichess_client.h"

#include <curl/curl.h>

#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <thread>

#include "generated_config.h"

using json = nlohmann::json;

LichessClient::CurlGlobalInit LichessClient::curl_init;

LichessClient::CurlGlobalInit::CurlGlobalInit() {
    curl_global_init(CURL_GLOBAL_DEFAULT);
}

LichessClient::CurlGlobalInit::~CurlGlobalInit() { curl_global_cleanup(); }

LichessClient::LichessClient(const std::string& token)
    : token(token), base_url(config::bot::LICHESS_BASE_URL) {}

LichessClient::~LichessClient() = default;

bool LichessClient::get_account_info(AccountInfo& info) {
    auto response = make_request(base_url + "/account");

    if (response.status_code != 200) {
        std::cerr << "Failed to get account info: " << response.status_code
                  << std::endl;
        return false;
    }

    try {
        auto j = json::parse(response.data);
        info.id = j["id"];
        info.username = j["username"];
        info.is_bot = j.value("title", "") == "BOT";
        info.title = j.value("title", "");
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing account info: " << e.what() << std::endl;
        return false;
    }
}

bool LichessClient::accept_challenge(const std::string& challenge_id) {
    auto response =
        make_request(base_url + "/challenge/" + challenge_id + "/accept", "POST");
    if (response.status_code != 200) {
        std::cout << "Challenge accept failed with status " << response.status_code
                  << ": " << response.data << std::endl;
    }
    return response.status_code == 200;
}

bool LichessClient::make_move(const std::string& game_id, const std::string& uci_move) {
    auto response =
        make_request(base_url + "/bot/game/" + game_id + "/move/" + uci_move, "POST");
    if (response.status_code != 200) {
        std::cout << "Make move failed with status " << response.status_code << ": "
                  << response.data << std::endl;
    }
    return response.status_code == 200;
}

bool LichessClient::accept_draw(const std::string& game_id) {
    auto response =
        make_request(base_url + "/bot/game/" + game_id + "/draw/yes", "POST");
    return response.status_code == 200;
}

bool LichessClient::decline_draw(const std::string& game_id) {
    auto response =
        make_request(base_url + "/bot/game/" + game_id + "/draw/no", "POST");
    return response.status_code == 200;
}

bool LichessClient::offer_draw(const std::string& game_id) {
    auto response =
        make_request(base_url + "/bot/game/" + game_id + "/draw/yes", "POST");
    return response.status_code == 200;
}

bool LichessClient::test_connectivity() {
    CURL* curl = curl_easy_init();
    if (!curl) return false;

    std::string response_data;
    std::string connectivity_url(config::bot::CONNECTIVITY_TEST_URL);
    curl_easy_setopt(curl, CURLOPT_URL, connectivity_url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, config::curl::CONNECTIVITY_TIMEOUT);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT,
                     config::curl::CONNECTIVITY_CONNECT_TIMEOUT);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);  // HEAD request only

    CURLcode res = curl_easy_perform(curl);
    long response_code;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) return false;

    auto response = make_request("https://lichess.org/api");
    return response.status_code != 0;
}

void LichessClient::stream_events(std::function<void(const GameEvent&)> callback) {
    stream_lines(base_url + "/stream/event", [callback](const std::string& line) {
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
            std::cerr << "Raw line: " << line << std::endl;
        }
    });
}

void LichessClient::stream_game(const std::string& game_id,
                                std::function<void(const GameEvent&)> callback) {
    stream_lines(
        base_url + "/bot/game/stream/" + game_id, [callback](const std::string& line) {
            if (line.empty()) return;

            try {
                auto j = json::parse(line);
                GameEvent event;
                event.type = j["type"];

                if (event.type == "gameState") {
                    event.moves = j.value("moves", "");
                    event.status = j.value("status", "started");
                    event.draw_offer =
                        j.value("wdraw", false) || j.value("bdraw", false);
                    event.wtime = j.value("wtime", 0);
                    event.btime = j.value("btime", 0);
                    event.winc = j.value("winc", 0);
                    event.binc = j.value("binc", 0);
                } else if (event.type == "gameFull") {
                    if (j.contains("state")) {
                        event.moves = j["state"].value("moves", "");
                        event.status = j["state"].value("status", "started");
                        event.draw_offer = j["state"].value("wdraw", false) ||
                                           j["state"].value("bdraw", false);
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

size_t LichessClient::write_callback(void* contents, size_t size, size_t nmemb,
                                     std::string* data) {
    size_t total_size = size * nmemb;
    data->append(static_cast<char*>(contents), total_size);
    return total_size;
}

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
                                                        const std::string& data,
                                                        bool stream) {
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

    // Set timeouts
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, config::curl::REQUEST_TIMEOUT);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT,
                     config::curl::REQUEST_CONNECT_TIMEOUT);

    // Enable keep-alive for regular requests too
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);

    // Set User-Agent
    std::string user_agent(config::bot::USER_AGENT);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, user_agent.c_str());

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
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, config::curl::MAX_REDIRECTS);

    // Perform the request
    CURLcode res = curl_easy_perform(curl);

    if (res == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response_code);
    } else {
        std::cerr << "CURL error for " << url << ": " << curl_easy_strerror(res)
                  << std::endl;
        response_code = 0;
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return {response_data, response_code};
}

void LichessClient::stream_lines(const std::string& url,
                                 std::function<void(const std::string&)> callback) {
    CURL* curl = curl_easy_init();
    if (!curl) return;

    StreamData stream_data;
    stream_data.callback = callback;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, stream_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream_data);

    // Set longer timeouts for streaming connections
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 0L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT,
                     config::curl::STREAM_CONNECT_TIMEOUT);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_LIMIT,
                     config::curl::STREAM_LOW_SPEED_LIMIT);
    curl_easy_setopt(curl, CURLOPT_LOW_SPEED_TIME, config::curl::STREAM_LOW_SPEED_TIME);

    // Enable keep-alive
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPIDLE, config::curl::STREAM_KEEPALIVE_IDLE);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPINTVL,
                     config::curl::STREAM_KEEPALIVE_INTERVAL);

    // Set headers
    struct curl_slist* headers = nullptr;
    std::string auth_header = "Authorization: Bearer " + token;
    headers = curl_slist_append(headers, auth_header.c_str());
    headers = curl_slist_append(headers, "Accept: text/plain");
    headers = curl_slist_append(headers, "Cache-Control: no-cache");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

    CURLcode res = curl_easy_perform(curl);

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
