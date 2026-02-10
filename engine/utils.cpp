#include "utils.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {
std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    return oss.str();
}
}  // namespace

std::vector<std::string> Utils::split_string(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;

    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}

void Utils::log_info(const std::string& message) {
    std::cout << "[INFO " << get_timestamp() << "] " << message << std::endl;
}

void Utils::log_error(const std::string& message) {
    std::cerr << "[ERROR " << get_timestamp() << "] " << message << std::endl;
}

void Utils::log_warning(const std::string& message) {
    std::cout << "[WARNING " << get_timestamp() << "] " << message << std::endl;
}
