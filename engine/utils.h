#pragma once
#include <string>
#include <vector>

class Utils {
   public:
    static std::vector<std::string> split_string(const std::string& str,
                                                 char delimiter);
    static void log_info(const std::string& message);
    static void log_error(const std::string& message);
    static void log_warning(const std::string& message);
};
