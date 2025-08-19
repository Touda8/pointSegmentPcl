#pragma once
#include <iostream>
#include <string>

class Logger {
public:
    enum LogLevel {
        DEBUG,
        INFO,
        WARNING,
        ERROR
    };

    static void setDebugMode(bool enable);
    static void log(LogLevel level, const std::string& message);
    static void debug(const std::string& message);
    static void info(const std::string& message);
    static void warning(const std::string& message);
    static void error(const std::string& message);

private:
    static bool debugMode;
    static std::string levelToString(LogLevel level);
};
