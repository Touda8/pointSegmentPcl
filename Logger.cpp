#include "Logger.h"
#include <iostream>
#include <string>

bool Logger::debugMode = false;

void Logger::setDebugMode(bool enable) {
    debugMode = enable;
#ifdef _DEBUG
    debugMode = true;  // 在Debug模式下强制启用
#endif
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level == DEBUG && !debugMode) {
        return;  // Debug消息在非Debug模式下不显示
    }
    
    std::cout << "[" << levelToString(level) << "] " << message << std::endl;
}

void Logger::debug(const std::string& message) {
    log(DEBUG, message);
}

void Logger::info(const std::string& message) {
    log(INFO, message);
}

void Logger::warning(const std::string& message) {
    log(WARNING, message);
}

void Logger::error(const std::string& message) {
    log(ERROR, message);
}

std::string Logger::levelToString(LogLevel level) {
    switch (level) {
        case DEBUG:   return "DEBUG";
        case INFO:    return "INFO";
        case WARNING: return "WARNING";
        case ERROR:   return "ERROR";
        default:      return "UNKNOWN";
    }
}
