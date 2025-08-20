#pragma once
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/random_sample.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "Logger.h"

// ROI结构定义
struct ImageROI {
    int id;
    std::string name;
    cv::Rect boundingBox;  // ROI的边界框
    std::vector<cv::Point> contour;  // ROI的轮廓（支持不规则形状）
    double minDepth, maxDepth;  // ROI内的深度范围
    cv::Scalar color;  // ROI显示颜色
    
    ImageROI() : id(-1), minDepth(0), maxDepth(0), color(cv::Scalar(0,255,0)) {}
};

// 像素到点云的映射信息
struct PixelToPointMapping {
    int pointIndex;  // 原始点云中的点索引
    float depth;     // 深度值
    cv::Point3f worldCoord;  // 世界坐标
    
    PixelToPointMapping() : pointIndex(-1), depth(0) {}
};

class TemplatePointCloudProcessor {
public:
    TemplatePointCloudProcessor();
    ~TemplatePointCloudProcessor();

    // 核心功能：模板点云预处理 - 拟合平面并生成配置文件
    bool processTemplatePointCloud(const std::string& templateCloudPath, const std::string& configOutputPath);
    
    // 辅助功能：使用现有配置将点云转换为深度图像（用于验证或后续处理）
    bool generateDepthImage(const std::string& inputCloudPath, const std::string& outputDir);

    // 设置图像参数
    void setImageSize(int width, int height);
    void setDepthRange(float minDepth, float maxDepth);
    void setCameraPosition(float x, float y, float z);
    void setCameraOrientation(float pitch, float yaw, float roll);
    void setAutoFitPlane(bool enable); // 启用/禁用自动平面拟合
    
    // 深度信息可视化
    void setDepthVisualization(bool enable);  // 启用深度信息显示
    void setDepthColorMap(int colorMapType);  // 设置深度颜色映射类型
    void setUsePlaneRelativeDepth(bool enable);  // 设置是否使用平面相对深度
    
    // ROI管理功能
    int addROI(const ImageROI& roi);  // 添加ROI，返回ROI ID
    bool removeROI(int roiId);        // 删除ROI
    void clearAllROIs();              // 清除所有ROI
    std::vector<ImageROI> getAllROIs() const;  // 获取所有ROI
    bool saveROIs(const std::string& filePath) const;  // 保存ROI到文件
    bool loadROIs(const std::string& filePath);        // 从文件加载ROI
    
    // 未来扩展：点云分割接口（当前为预留接口）
    bool segmentPointCloudByROI(int roiId, const std::string& outputPath);
    std::vector<int> getPointIndicesInROI(int roiId) const;  // 获取ROI内的点索引
    
    // 配置管理（模板预处理的核心输出）
    bool saveCameraConfiguration(const std::string& configPath) const;  // 保存模板分析结果到配置文件
    bool loadCameraConfiguration(const std::string& configPath);  // 加载模板配置用于后续处理
    bool hasSavedConfiguration() const;  // 检查是否有已保存的模板配置
    void setAutoSaveConfig(bool enable) { autoSaveConfig_ = enable; }  // 设置是否自动保存模板配置
    bool getAutoSaveConfig() const { return autoSaveConfig_; }  // 获取自动保存配置状态

private:
    // 加载点云
    bool loadPointCloud(const std::string& filePath, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    // 将3D点云投影到2D深度图像
    bool projectToDepthImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, cv::Mat& depthImage);
    
    // 变换点云到相机坐标系
    void transformPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    // 平面拟合相关方法
    bool fitPlaneToPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    bool fitPlaneRobustPCA(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud); // 改进的PCA方法
    bool fitPlaneSimple(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud); // 简单统计方法
    void computeCameraFromPlane();
    void adjustDepthRangeForTransformedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    void adjustPlaneRelativeDepthRange(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);  // 调整平面相对深度范围
    
    // 点云预处理方法
    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    // 保存深度图像
    bool saveDepthImage(const cv::Mat& depthImage, const std::string& outputPath);
    
    // 创建输出目录
    bool createOutputDirectory(const std::string& dirPath);
    
    // 归一化深度值到0-255范围
    void normalizeDepthImage(cv::Mat& depthImage);
    
    // 深度信息可视化相关
    cv::Mat generateColorDepthImage(const cv::Mat& depthImage);  // 生成彩色深度图
    cv::Mat generateEnhancedGrayscaleDepthImage(const cv::Mat& depthImage);  // 生成增强对比度的灰度深度图
    cv::Mat generateDepthInfoImage(const cv::Mat& depthImage);   // 生成带深度信息的图像
    void drawDepthLegend(cv::Mat& image, double minVal, double maxVal);  // 绘制深度图例
    
    // ROI相关内部方法
    void drawROIs(cv::Mat& image) const;  // 在图像上绘制ROI
    bool isPointInROI(const cv::Point& point, const ImageROI& roi) const;  // 判断点是否在ROI内
    
    // 像素到点云映射相关
    void updatePixelToPointMapping(int u, int v, int pointIndex, float depth, const cv::Point3f& worldCoord);
    void clearPixelToPointMapping();

private:
    int imageWidth_;
    int imageHeight_;
    float minDepth_;
    float maxDepth_;
    float cameraX_, cameraY_, cameraZ_;
    float cameraPitch_, cameraYaw_, cameraRoll_;
    
    // 相机内参（焦距等）
    float focalLength_;
    float principalPointX_, principalPointY_;
    
    // 平面拟合相关
    bool autoFitPlane_;
    bool autoSaveConfig_;  // 是否自动保存配置
    Eigen::Vector3f planeNormal_;
    Eigen::Vector3f planeCenter_;
    float planeDistance_;
    
    // 深度信息可视化相关
    bool enableDepthVisualization_;
    int depthColorMapType_;  // OpenCV颜色映射类型
    bool usePlaneRelativeDepth_;  // 是否使用平面相对深度
    float planeRelativeMinDepth_, planeRelativeMaxDepth_;  // 相对于拟合平面的深度范围
    
    // 模板处理模式标志
    bool isTemplateProcessingMode_;  // 模板处理模式下不进行深度过滤
    
    // ROI管理
    std::vector<ImageROI> rois_;
    int nextROIId_;
    
    // 像素到点云的映射（用于未来的点云分割）
    std::vector<std::vector<PixelToPointMapping>> pixelToPointMap_;  // [row][col]
    pcl::PointCloud<pcl::PointXYZ>::Ptr originalPointCloud_;  // 保存原始点云用于分割
};
