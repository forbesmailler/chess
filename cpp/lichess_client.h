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
    };
    
    struct GameEvent {
        std::string type;
        std::string game_id;
        std::string challenge_id;
        std::string moves;
        std::string status;
        bool is_white;
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
    
    HttpResponse make_request(const std::string& url, const std::string& method = "GET", 
                            const std::string& data = "", bool stream = false);
    
    static size_t write_callback(void* contents, size_t size, size_t nmemb, std::string* data);
    static size_t stream_callback(void* contents, size_t size, size_t nmemb, 
                                std::function<void(const std::string&)>* callback);
};
