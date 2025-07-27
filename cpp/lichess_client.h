#pragma once
#include <string>
#include <functional>
#include <memory>

class LichessClient {
public:
    explicit LichessClient(const std::string& token);
    ~LichessClient();
    
    struct AccountInfo {
        std::string id;
        std::string username;
        bool is_bot = false;
        std::string title;
    };
    
    struct GameEvent {
        std::string type;
        std::string game_id;
        std::string challenge_id;
        std::string moves;
        std::string status;
        bool is_white = false;
        int wtime = 0;
        int btime = 0;
        int winc = 0;
        int binc = 0;
        std::string white_id;
        std::string black_id;
    };
    
    bool get_account_info(AccountInfo& info);
    bool accept_challenge(const std::string& challenge_id);
    bool make_move(const std::string& game_id, const std::string& uci_move);
    
    void stream_events(std::function<void(const GameEvent&)> callback);
    void stream_game(const std::string& game_id, std::function<void(const GameEvent&)> callback);
    
private:
    std::string token;
    std::string base_url;
    
    struct CurlGlobalInit {
        CurlGlobalInit();
        ~CurlGlobalInit();
    };
    
    static CurlGlobalInit curl_init;
    
    struct HttpResponse {
        std::string data;
        long status_code;
    };
    
    struct StreamData {
        std::string buffer;
        std::function<void(const std::string&)> callback;
    };
    
    HttpResponse make_request(const std::string& url, const std::string& method = "GET", 
                            const std::string& data = "", bool stream = false);
    
    void stream_lines(const std::string& url, std::function<void(const std::string&)> callback);
    
    static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* data);
    static size_t stream_callback(void* contents, size_t size, size_t nmemb, StreamData* stream_data);
};
