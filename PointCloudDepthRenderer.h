#pragma once

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <map>
#include "Logger.h"

// ROI结构体定义
struct ROI {
    int id;
    cv::Rect rect;
    std::string name;
    cv::Scalar color;
    std::vector<int> pointIndices;  // 该ROI包含的点云索引
};

// 像素到点云的映射结构
struct PixelMapping {
    int pointIndex;      // 对应的点云索引
    float depth;         // 深度值
    cv::Point3f worldCoord;  // 世界坐标
    bool isValid;        // 是否有效
    
    PixelMapping() : pointIndex(-1), depth(0.0f), worldCoord(0, 0, 0), isValid(false) {}
};

class PointCloudDepthRenderer {
public:
    PointCloudDepthRenderer();
    ~PointCloudDepthRenderer();
    
    // 配置管理
    bool loadConfiguration(const std::string& configPath);
    bool saveConfiguration(const std::string& configPath) const;
    
    // 主要处理函数
    bool processPointCloud(const std::string& pointCloudPath, 
                          const std::string& outputDir,
                          const std::string& configPath = "");
    
    // 深度图生成
    bool generateDepthImages(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                           cv::Mat& grayDepthImage,
                           cv::Mat& colorDepthImage,
                           cv::Mat& depthInfoImage);
    
    // ROI相关接口
    int addROI(const cv::Rect& rect, const std::string& name = "", const cv::Scalar& color = cv::Scalar(0, 255, 0));
    bool removeROI(int roiId);
    void clearAllROIs();
    std::vector<ROI> getAllROIs() const;
    
    // ROI点云分割接口
    pcl::PointCloud<pcl::PointXYZ>::Ptr extractROIPointCloud(int roiId) const;
    std::vector<int> getROIPointIndices(int roiId) const;
    bool saveROIPointCloud(int roiId, const std::string& outputPath) const;
    
    // ROI可视化
    void drawROIs(cv::Mat& image) const;
    cv::Mat generateROIOverlayImage(const cv::Mat& baseImage) const;
    
    // 深度图可视化设置
    void setColorMap(int colorMapType);
    void setDepthRange(float minDepth, float maxDepth);
    void setImageSize(int width, int height);
    
    // 获取处理结果
    cv::Mat getLastGrayDepthImage() const { return lastGrayDepthImage_.clone(); }
    cv::Mat getLastColorDepthImage() const { return lastColorDepthImage_.clone(); }
    cv::Mat getLastDepthInfoImage() const { return lastDepthInfoImage_.clone(); }
    
    // 像素到点云的映射查询
    PixelMapping getPixelMapping(int x, int y) const;
    bool isPixelValid(int x, int y) const;
    
    // 配置参数访问
    bool hasValidConfiguration() const { return configLoaded_; }
    
private:
    // 核心处理函数
    bool loadPointCloud(const std::string& filePath, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    bool projectToDepthImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, cv::Mat& depthImage);
    void transformPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    // 深度图生成函数
    cv::Mat generateColorDepthImage(const cv::Mat& depthImage);
    cv::Mat generateGrayscaleDepthImage(const cv::Mat& depthImage);
    cv::Mat generateDepthInfoImage(const cv::Mat& depthImage);
    
    // 像素到点云映射
    void updatePixelToPointMapping(int u, int v, int pointIndex, float depth, const cv::Point3f& worldCoord);
    void clearPixelToPointMapping();
    
    // ROI处理
    void updateROIPointIndices();
    bool isPointInROI(const cv::Point& point, const ROI& roi) const;
    
    // 配置文件解析
    bool parseConfigurationFile(const std::string& configPath);
    
    // 输出管理
    bool createOutputDirectory(const std::string& dirPath);
    bool saveDepthImage(const cv::Mat& image, const std::string& outputPath);
    
private:
    // 相机参数
    float cameraX_, cameraY_, cameraZ_;
    float cameraPitch_, cameraYaw_, cameraRoll_;
    
    // 图像参数
    int imageWidth_, imageHeight_;
    float focalLength_;
    float minDepth_, maxDepth_;
    
    // 平面参数
    Eigen::Vector3f planeNormal_;
    Eigen::Vector3f planeCenter_;
    float planeDistance_;
    
    // 深度图设置
    int colorMapType_;
    bool usePlaneRelativeDepth_;
    
    // ROI管理
    std::vector<ROI> rois_;
    int nextROIId_;
    
    // 像素到点云映射
    std::vector<std::vector<PixelMapping>> pixelToPointMap_;
    
    // 当前处理的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr currentPointCloud_;
    
    // 最后生成的深度图
    cv::Mat lastGrayDepthImage_;
    cv::Mat lastColorDepthImage_;
    cv::Mat lastDepthInfoImage_;
    cv::Mat lastRawDepthImage_;
    
    // 状态标志
    bool configLoaded_;
    bool pointCloudLoaded_;
    
    // 变换矩阵
    Eigen::Matrix4f transformMatrix_;
};
