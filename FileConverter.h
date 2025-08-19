#pragma once
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <vtkSTLReader.h>
#include <vtkTriangleFilter.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include "Logger.h"

class FileConverter {
public:
    FileConverter();
    ~FileConverter();

    // 主要转换接口
    bool convertFile(const std::string& inputPath, const std::string& outputPath);

private:
    // 获取文件扩展名
    std::string getFileExtension(const std::string& filePath);
    
    // 转换STL文件到PLY
    bool convertSTLtoPLY(const std::string& inputPath, const std::string& outputPath);
    
    // 转换STEP文件到PLY（目前不支持，会给出提示）
    bool convertSTEPtoPLY(const std::string& inputPath, const std::string& outputPath);
    
    // 将VTK PolyData转换为PCL点云
    bool vtkPolyDataToPCLPointCloud(vtkPolyData* polyData, pcl::PointCloud<pcl::PointXYZ>::Ptr& pointCloud);
    
    // 将文件扩展名转换为小写
    std::string toLower(const std::string& str);
    
    // 检查文件是否存在
    bool fileExists(const std::string& filePath);
};
