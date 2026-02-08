#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
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

std::string Utils::trim(const std::string& str) {
    auto start = str.begin();
    auto end = str.end();

    while (start != end && std::isspace(*start)) ++start;
    while (end != start && std::isspace(*(end - 1))) --end;

    return {start, end};
}

bool Utils::string_ends_with(const std::string& str, const std::string& suffix) {
    return suffix.length() <= str.length() &&
           str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
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

std::vector<float> Utils::softmax(const std::vector<float>& logits) {
    float max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<float> result(logits.size());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = std::exp(logits[i] - max_val);
        sum += result[i];
    }
    for (auto& v : result) v /= sum;
    return result;
}
