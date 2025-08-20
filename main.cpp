#include <iostream>
#include <string>
#include <vector>
#include "Logger.h"
#include "FileConverter.h"
#include "TemplatePointCloudProcessor.h"
#include "PointCloudDepthRenderer.h"

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

    // 新功能：模板点云预处理和点云渲染
    Logger::info("Starting template point cloud preprocessing...");
    
    std::string templateCloudPath = "point/input/newModel.ply";
    std::string configPath = "output/template_config.ini";
    
    Logger::info("Template point cloud: " + templateCloudPath);
    Logger::info("Configuration output: " + configPath);
    
    // 创建模板预处理器
    TemplatePointCloudProcessor templateProcessor;
    
    // 设置基本参数
    templateProcessor.setImageSize(1920, 1080);
    templateProcessor.setDepthRange(-2000.0f, -1000.0f);
    
    // 进行模板点云预处理 - 这是核心功能
    bool templateSuccess = templateProcessor.processTemplatePointCloud(templateCloudPath, configPath);
    
    if (templateSuccess) {
        Logger::info("Template preprocessing completed successfully!");
        Logger::info("Configuration file generated: " + configPath);
        Logger::info("Template processing finished. Configuration file can be used for future point cloud processing.");
        
        // 示例：使用生成的配置文件进行点云渲染
        Logger::info("Testing point cloud depth rendering with generated configuration...");
        
        PointCloudDepthRenderer renderer;
        std::string testPointCloudPath = "input/sheet1/ball_PointCloud_20250603_141316.ply";
        std::string renderOutputDir = "output/render_test";
        
        bool renderSuccess = renderer.processPointCloud(testPointCloudPath, renderOutputDir, configPath);
        
        if (renderSuccess) {
            Logger::info("Point cloud rendering test completed successfully!");
            
            // 示例：添加ROI进行区域分割
            Logger::info("Adding example ROIs for region segmentation...");
            int roi1 = renderer.addROI(cv::Rect(300, 200, 400, 300), "Object_Region", cv::Scalar(0, 255, 0));
            int roi2 = renderer.addROI(cv::Rect(800, 400, 300, 250), "Background_Region", cv::Scalar(255, 0, 0));
            
            // 生成带ROI的深度图
            cv::Mat colorDepthImage = renderer.getLastColorDepthImage();
            if (!colorDepthImage.empty()) {
                cv::Mat roiOverlayImage = renderer.generateROIOverlayImage(colorDepthImage);
                std::string roiImagePath = renderOutputDir + "/depth_with_rois.png";
                cv::imwrite(roiImagePath, roiOverlayImage);
                Logger::info("ROI overlay image saved to: " + roiImagePath);
            }
            
            // 提取ROI点云
            if (renderer.saveROIPointCloud(roi1, renderOutputDir + "/object_region_points.ply")) {
                Logger::info("Object region point cloud saved");
            }
            if (renderer.saveROIPointCloud(roi2, renderOutputDir + "/background_region_points.ply")) {
                Logger::info("Background region point cloud saved");
            }
            
            // 显示ROI统计信息
            std::vector<int> roi1Points = renderer.getROIPointIndices(roi1);
            std::vector<int> roi2Points = renderer.getROIPointIndices(roi2);
            Logger::info("ROI 1 contains " + std::to_string(roi1Points.size()) + " points");
            Logger::info("ROI 2 contains " + std::to_string(roi2Points.size()) + " points");
        }
        
        return 0;
    } else {
        Logger::error("Template preprocessing failed!");
        return 1;
    }
}
