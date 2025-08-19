#include <iostream>
#include <string>
#include <vector>
#include "Logger.h"
#include "FileConverter.h"

void printUsage(const std::string& programName) {
    std::cout << "Usage: " << programName << " <input_path> <output_path>" << std::endl;
    std::cout << "Converts 3D model files to PLY format" << std::endl;
    std::cout << "Supported input formats: .stl, .step, .stp" << std::endl;
    std::cout << "Output format: .ply" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " model.stl output.ply" << std::endl;
    std::cout << "  " << programName << " design.step converted.ply" << std::endl;
}

int main(int argc, char* argv[]) {
    // 设置日志模式（在Debug编译时自动启用详细日志）
#ifdef _DEBUG
    Logger::setDebugMode(true);
    Logger::debug("Running in DEBUG mode");
#else
    Logger::setDebugMode(false);
#endif

    Logger::info("File Converter Tool - STL/STEP to PLY");
    Logger::info("Built with PCL and VTK libraries");

    // 检查命令行参数
    if (argc != 3) {
        Logger::error("Invalid number of arguments");
        printUsage(argv[0]);
        return 1;
    }

    std::string inputPath = argv[1];
    std::string outputPath = argv[2];

    Logger::info("Input file: " + inputPath);
    Logger::info("Output file: " + outputPath);

    // 创建文件转换器
    FileConverter converter;

    // 执行转换
    bool success = converter.convertFile(inputPath, outputPath);

    if (success) {
        Logger::info("File conversion completed successfully!");
        return 0;
    } else {
        Logger::error("File conversion failed!");
        return 1;
    }
}
