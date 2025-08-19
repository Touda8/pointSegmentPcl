#include "FileConverter.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <vtkSTLReader.h>
#include <vtkPLYWriter.h>
#include <vtkPolyData.h>
#include <vtkSmartPointer.h>
#include <vtkPoints.h>
#include <vtkPointData.h>

FileConverter::FileConverter() {
    Logger::debug("FileConverter initialized");
}

FileConverter::~FileConverter() {
    Logger::debug("FileConverter destroyed");
}

bool FileConverter::convertFile(const std::string& inputPath, const std::string& outputPath) {
    Logger::debug("Starting file conversion: " + inputPath + " -> " + outputPath);
    
    // 检查输入文件是否存在
    if (!fileExists(inputPath)) {
        Logger::error("Input file does not exist: " + inputPath);
        return false;
    }
    
    // 获取文件扩展名
    std::string inputExt = getFileExtension(inputPath);
    std::string outputExt = getFileExtension(outputPath);
    
    // 检查输出格式是否为PLY
    if (toLower(outputExt) != ".ply") {
        Logger::error("Output file must be in PLY format, got: " + outputExt);
        return false;
    }
    
    // 根据输入文件类型选择转换方法
    if (toLower(inputExt) == ".stl") {
        return convertSTLtoPLY(inputPath, outputPath);
    }
    else if (toLower(inputExt) == ".step" || toLower(inputExt) == ".stp") {
        return convertSTEPtoPLY(inputPath, outputPath);
    }
    else {
        Logger::error("Unsupported input file format: " + inputExt);
        Logger::info("Supported formats: .stl, .step, .stp");
        return false;
    }
}

std::string FileConverter::getFileExtension(const std::string& filePath) {
    size_t dotPos = filePath.find_last_of('.');
    if (dotPos != std::string::npos) {
        return filePath.substr(dotPos);
    }
    return "";
}

bool FileConverter::convertSTLtoPLY(const std::string& inputPath, const std::string& outputPath) {
    try {
        Logger::info("Converting STL to PLY (Point Cloud format): " + inputPath);
        
        // 确保输出目录存在
        size_t pos = outputPath.find_last_of("\\/");
        if (pos != std::string::npos) {
            std::string dir = outputPath.substr(0, pos);
            // 创建目录（使用系统命令）
            std::string mkdirCmd = "mkdir \"" + dir + "\" 2>nul";
            system(mkdirCmd.c_str());
        }
        
        // 使用VTK读取STL文件
        vtkSmartPointer<vtkSTLReader> stlReader = vtkSmartPointer<vtkSTLReader>::New();
        stlReader->SetFileName(inputPath.c_str());
        stlReader->Update();
        
        vtkPolyData* polyData = stlReader->GetOutput();
        if (polyData->GetNumberOfPoints() == 0) {
            Logger::error("Failed to read STL file or file is empty: " + inputPath);
            return false;
        }
        
        Logger::debug("STL file loaded successfully, mesh vertices: " + 
                     std::to_string(polyData->GetNumberOfPoints()));
        
        // 将VTK PolyData转换为PCL点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        if (!vtkPolyDataToPCLPointCloud(polyData, pointCloud)) {
            Logger::error("Failed to convert mesh to point cloud");
            return false;
        }
        
        Logger::debug("Point cloud created with " + std::to_string(pointCloud->size()) + " points");
        
        // 使用PCL保存PLY文件（点云格式）
        if (pcl::io::savePLYFileASCII(outputPath, *pointCloud) == -1) {
            Logger::error("Failed to save PLY file: " + outputPath);
            return false;
        }
        
        Logger::info("Successfully converted STL to PLY point cloud: " + outputPath);
        Logger::info("Point cloud contains " + std::to_string(pointCloud->size()) + " points");
        return true;
        
    } catch (const std::exception& e) {
        Logger::error("Exception during STL to PLY conversion: " + std::string(e.what()));
        return false;
    } catch (...) {
        Logger::error("Unknown exception during STL to PLY conversion");
        return false;
    }
}

bool FileConverter::convertSTEPtoPLY(const std::string& inputPath, const std::string& outputPath) {
    Logger::error("STEP file conversion is not currently supported");
    Logger::info("STEP files require additional libraries like OpenCASCADE or FreeCAD");
    Logger::info("Consider converting STEP to STL first using CAD software, then use this tool");
    Logger::info("Input file: " + inputPath);
    Logger::info("Intended output: " + outputPath);
    
    return false;
}

std::string FileConverter::toLower(const std::string& str) {
    std::string lowerStr = str;
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
    return lowerStr;
}

bool FileConverter::vtkPolyDataToPCLPointCloud(vtkPolyData* polyData, pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud) {
    if (!polyData || polyData->GetNumberOfPoints() == 0) {
        Logger::error("Invalid or empty polydata");
        return false;
    }
    
    vtkPoints* points = polyData->GetPoints();
    if (!points) {
        Logger::error("No points found in polydata");
        return false;
    }
    
    // 清空点云
    pointCloud->clear();
    pointCloud->width = points->GetNumberOfPoints();
    pointCloud->height = 1;
    pointCloud->is_dense = true;
    pointCloud->resize(pointCloud->width * pointCloud->height);
    
    // 将VTK点复制到PCL点云
    for (vtkIdType i = 0; i < points->GetNumberOfPoints(); ++i) {
        double point[3];
        points->GetPoint(i, point);
        
        pcl::PointXYZ& pclPoint = pointCloud->points[i];
        pclPoint.x = static_cast<float>(point[0]);
        pclPoint.y = static_cast<float>(point[1]);
        pclPoint.z = static_cast<float>(point[2]);
    }
    
    Logger::debug("Converted " + std::to_string(pointCloud->size()) + " mesh vertices to point cloud");
    return true;
}

bool FileConverter::fileExists(const std::string& filePath) {
    // 使用传统方法检查文件是否存在，兼容性更好
    std::ifstream file(filePath);
    return file.good();
}
