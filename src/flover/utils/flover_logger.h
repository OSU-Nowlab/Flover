#pragma once
#include <sstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <string>

// Base function
inline void log_impl(std::ostringstream& oss) {}

// Variadic template function
template<typename T, typename... Args>
inline void log_impl(std::ostringstream& oss, T const& first, Args const&... rest) {
    oss << first;
    log_impl(oss, rest...);
}

// Log function
template<typename... Args>
void log(Args const&... args) {
    // Get the current time
    auto now = std::chrono::system_clock::now();

    // Convert it to a time_t object
    auto now_t = std::chrono::system_clock::to_time_t(now);

    // Convert the time_t to a string
    std::stringstream ss;
    ss << std::ctime(&now_t);

    // Remove the trailing newline added by ctime
    std::string time_str = ss.str();
    time_str.pop_back();

    // Get the number of milliseconds from the epoch
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    
    // Create an ostringstream for the log message
    std::ostringstream oss;
    oss << "[" << time_str << "." << millis << "] ";
    
    log_impl(oss, args...);

    // Output the log message
    std::cout << oss.str() << "\n" << std::flush;
}
