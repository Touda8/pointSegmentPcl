#pragma once
#include <string>
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

class DepthImageGenerator {
public:
    DepthImageGenerator();
    ~DepthImageGenerator();

    // 主要功能接口：将点云转换为深度图像
    bool generateDepthImage(const std::string& inputCloudPath, const std::string& outputDir);

    // 设置图像参数
    void setImageSize(int width, int height);
    void setDepthRange(float minDepth, float maxDepth);
    void setCameraPosition(float x, float y, float z);
    void setCameraOrientation(float pitch, float yaw, float roll);
    void setAutoFitPlane(bool enable); // 启用/禁用自动平面拟合

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
    
    // 点云预处理方法
    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    
    // 保存深度图像
    bool saveDepthImage(const cv::Mat& depthImage, const std::string& outputPath);
    
    // 创建输出目录
    bool createOutputDirectory(const std::string& dirPath);
    
    // 归一化深度值到0-255范围
    void normalizeDepthImage(cv::Mat& depthImage);

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
    Eigen::Vector3f planeNormal_;
    Eigen::Vector3f planeCenter_;
    float planeDistance_;
};
