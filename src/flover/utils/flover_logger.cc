#include "logger.h"
#include <iostream>
#include <chrono>
#include <ctime>
#include <string>

void log_impl(std::ostringstream& oss) {}

template<typename T, typename... Args>
void log_impl(std::ostringstream& oss, T const& first, Args const&... rest) {
    oss << first;
    log_impl(oss, rest...);
}

template<typename... Args>
void log(Args const&... args) {
    auto now = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::ctime(&now_t);

    std::string time_str = ss.str();
    time_str.pop_back();

    std::ostringstream oss;
    oss << "[" << time_str << "] ";
    
    log_impl(oss, args...);

    std::cout << oss.str() << "\n";
}