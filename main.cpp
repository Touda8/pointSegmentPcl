#include <iostream>
#include <string>
#include <vector>
#include "Logger.h"
#include "FileConverter.h"
#include "DepthImageGenerator.h"

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

    Logger::info("Point Cloud Processing Tool");
    Logger::info("Built with PCL, VTK and OpenCV libraries");

    /* 
    // 文件转换功能已注释掉
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
    */

    // 新功能：点云到深度图像转换
    Logger::info("Starting point cloud to depth image conversion...");
    
    std::string inputCloudPath = "point/input/pallet_ref.ply";
    std::string outputImageDir = "output";
    
    Logger::info("Input point cloud: " + inputCloudPath);
    Logger::info("Output directory: " + outputImageDir);
    
    // 创建深度图像生成器
    DepthImageGenerator depthGenerator;
    
    // 设置参数（可选，使用默认值也可以）
    depthGenerator.setImageSize(1920, 1080);
    depthGenerator.setDepthRange(-2000.0f, -1000.0f);
    depthGenerator.setCameraPosition(0.0f, 0.0f, 0.0f);
    depthGenerator.setAutoFitPlane(true); // 启用修复后的自动平面拟合
    
    // 启用深度信息可视化
    depthGenerator.setDepthVisualization(true);
    depthGenerator.setDepthColorMap(cv::COLORMAP_JET); // 使用JET颜色映射
    
    // 添加示例ROI（可选）
    ImageROI roi1;
    roi1.name = "Sample_ROI_1";
    roi1.boundingBox = cv::Rect(400, 300, 200, 150);
    roi1.color = cv::Scalar(0, 255, 0); // 绿色
    int roiId1 = depthGenerator.addROI(roi1);
    
    ImageROI roi2;
    roi2.name = "Sample_ROI_2";
    roi2.boundingBox = cv::Rect(800, 500, 300, 200);
    roi2.color = cv::Scalar(255, 0, 0); // 蓝色
    int roiId2 = depthGenerator.addROI(roi2);
    
    Logger::info("Added sample ROIs with IDs: " + std::to_string(roiId1) + ", " + std::to_string(roiId2));
    
    // 生成深度图像
    bool success = depthGenerator.generateDepthImage(inputCloudPath, outputImageDir);
    
    if (success) {
        Logger::info("Depth image conversion completed successfully!");
        
        // 演示点云分割功能（未来扩展）
        // std::string segmentedOutputPath = outputImageDir + "/segmented_roi_" + std::to_string(roiId1) + ".ply";
        // depthGenerator.segmentPointCloudByROI(roiId1, segmentedOutputPath);
        
        return 0;
    } else {
        Logger::error("Depth image conversion failed!");
        return 1;
    }
}
