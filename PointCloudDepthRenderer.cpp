#include "PointCloudDepthRenderer.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

PointCloudDepthRenderer::PointCloudDepthRenderer() 
    : cameraX_(0.0f), cameraY_(0.0f), cameraZ_(-2000.0f),
      cameraPitch_(0.0f), cameraYaw_(0.0f), cameraRoll_(0.0f),
      imageWidth_(1920), imageHeight_(1080),
      focalLength_(1536.0f),
      minDepth_(-2000.0f), maxDepth_(-1000.0f),
      planeNormal_(0, 0, 1), planeCenter_(0, 0, 0), planeDistance_(0),
      colorMapType_(cv::COLORMAP_PLASMA),
      usePlaneRelativeDepth_(true),
      nextROIId_(1),
      configLoaded_(false),
      pointCloudLoaded_(false),
      currentPointCloud_(new pcl::PointCloud<pcl::PointXYZ>())
{
    transformMatrix_ = Eigen::Matrix4f::Identity();
    
    // 初始化像素到点云映射
    pixelToPointMap_.resize(imageHeight_);
    for (int i = 0; i < imageHeight_; ++i) {
        pixelToPointMap_[i].resize(imageWidth_);
    }
    
    Logger::debug("PointCloudDepthRenderer initialized");
}

PointCloudDepthRenderer::~PointCloudDepthRenderer() {
    Logger::debug("PointCloudDepthRenderer destroyed");
}

bool PointCloudDepthRenderer::loadConfiguration(const std::string& configPath) {
    Logger::info("Loading configuration from: " + configPath);
    
    if (!parseConfigurationFile(configPath)) {
        Logger::error("Failed to parse configuration file: " + configPath);
        return false;
    }
    
    configLoaded_ = true;
    Logger::info("Configuration loaded successfully");
    return true;
}

bool PointCloudDepthRenderer::parseConfigurationFile(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        Logger::error("Cannot open configuration file: " + configPath);
        return false;
    }
    
    std::string line;
    std::string currentSection;
    
    while (std::getline(file, line)) {
        // 移除空格和注释
        line.erase(0, line.find_first_not_of(" \t"));
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        
        // 检查是否是节（section）
        if (line[0] == '[' && line.back() == ']') {
            currentSection = line.substr(1, line.length() - 2);
            continue;
        }
        
        // 解析键值对
        size_t equalPos = line.find('=');
        if (equalPos == std::string::npos) continue;
        
        std::string key = line.substr(0, equalPos);
        std::string value = line.substr(equalPos + 1);
        
        // 移除空格
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        
        // 根据节解析不同的参数
        if (currentSection == "Camera") {
            if (key == "positionX") cameraX_ = std::stof(value);
            else if (key == "positionY") cameraY_ = std::stof(value);
            else if (key == "positionZ") cameraZ_ = std::stof(value);
            else if (key == "pitch") cameraPitch_ = std::stof(value);
            else if (key == "yaw") cameraYaw_ = std::stof(value);
            else if (key == "roll") cameraRoll_ = std::stof(value);
        }
        else if (currentSection == "Image") {
            if (key == "width") imageWidth_ = std::stoi(value);
            else if (key == "height") imageHeight_ = std::stoi(value);
            else if (key == "focalLength") focalLength_ = std::stof(value);
        }
        else if (currentSection == "Depth") {
            if (key == "minDepth") minDepth_ = std::stof(value);
            else if (key == "maxDepth") maxDepth_ = std::stof(value);
            else if (key == "colorMapType") colorMapType_ = std::stoi(value);
            else if (key == "usePlaneRelativeDepth") usePlaneRelativeDepth_ = (value == "true" || value == "1");
        }
        else if (currentSection == "Plane") {
            if (key == "normalX") planeNormal_.x() = std::stof(value);
            else if (key == "normalY") planeNormal_.y() = std::stof(value);
            else if (key == "normalZ") planeNormal_.z() = std::stof(value);
            else if (key == "centerX") planeCenter_.x() = std::stof(value);
            else if (key == "centerY") planeCenter_.y() = std::stof(value);
            else if (key == "centerZ") planeCenter_.z() = std::stof(value);
            else if (key == "distance") planeDistance_ = std::stof(value);
        }
    }
    
    // 更新像素映射大小
    pixelToPointMap_.resize(imageHeight_);
    for (int i = 0; i < imageHeight_; ++i) {
        pixelToPointMap_[i].resize(imageWidth_);
    }
    
    // 计算变换矩阵
    float pitch = cameraPitch_ * M_PI / 180.0f;
    float yaw = cameraYaw_ * M_PI / 180.0f;
    float roll = cameraRoll_ * M_PI / 180.0f;
    
    Eigen::Matrix3f rotation = 
        (Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
         Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
         Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX())).toRotationMatrix();
    
    transformMatrix_.block<3,3>(0,0) = rotation.transpose();
    transformMatrix_(0,3) = -cameraX_;
    transformMatrix_(1,3) = -cameraY_;
    transformMatrix_(2,3) = -cameraZ_;
    
    Logger::debug("Configuration parsed successfully");
    return true;
}

bool PointCloudDepthRenderer::processPointCloud(const std::string& pointCloudPath, 
                                               const std::string& outputDir,
                                               const std::string& configPath) {
    Logger::info("Processing point cloud: " + pointCloudPath);
    
    // 如果提供了配置路径，加载配置
    if (!configPath.empty() && !loadConfiguration(configPath)) {
        Logger::error("Failed to load configuration");
        return false;
    }
    
    if (!configLoaded_) {
        Logger::error("No valid configuration loaded");
        return false;
    }
    
    // 加载点云
    if (!loadPointCloud(pointCloudPath, currentPointCloud_)) {
        Logger::error("Failed to load point cloud");
        return false;
    }
    
    pointCloudLoaded_ = true;
    
    // 创建输出目录
    if (!createOutputDirectory(outputDir)) {
        Logger::error("Failed to create output directory");
        return false;
    }
    
    // 生成深度图
    cv::Mat grayDepthImage, colorDepthImage, depthInfoImage;
    if (!generateDepthImages(currentPointCloud_, grayDepthImage, colorDepthImage, depthInfoImage)) {
        Logger::error("Failed to generate depth images");
        return false;
    }
    
    // 保存深度图
    std::string baseName = fs::path(pointCloudPath).stem().string();
    
    std::string grayPath = outputDir + "/" + baseName + "_depth_gray.png";
    std::string colorPath = outputDir + "/" + baseName + "_depth_color.png";
    std::string infoPath = outputDir + "/" + baseName + "_depth_info.png";
    
    if (!saveDepthImage(grayDepthImage, grayPath) ||
        !saveDepthImage(colorDepthImage, colorPath) ||
        !saveDepthImage(depthInfoImage, infoPath)) {
        Logger::error("Failed to save depth images");
        return false;
    }
    
    Logger::info("Point cloud processing completed successfully");
    Logger::info("Generated depth images:");
    Logger::info("  Gray: " + grayPath);
    Logger::info("  Color: " + colorPath);
    Logger::info("  Info: " + infoPath);
    
    return true;
}

bool PointCloudDepthRenderer::generateDepthImages(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                 cv::Mat& grayDepthImage,
                                                 cv::Mat& colorDepthImage,
                                                 cv::Mat& depthInfoImage) {
    Logger::debug("Generating depth images");
    
    // 清理像素映射
    clearPixelToPointMapping();
    
    // 创建点云副本用于变换
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*cloud, *transformedCloud);
    
    // 应用变换
    transformPointCloud(transformedCloud);
    
    // 投影到深度图
    cv::Mat rawDepthImage;
    if (!projectToDepthImage(transformedCloud, rawDepthImage)) {
        Logger::error("Failed to project point cloud to depth image");
        return false;
    }
    
    // 保存原始深度图
    lastRawDepthImage_ = rawDepthImage.clone();
    
    // 生成不同类型的深度图
    grayDepthImage = generateGrayscaleDepthImage(rawDepthImage);
    colorDepthImage = generateColorDepthImage(rawDepthImage);
    depthInfoImage = generateDepthInfoImage(rawDepthImage);
    
    // 保存结果
    lastGrayDepthImage_ = grayDepthImage.clone();
    lastColorDepthImage_ = colorDepthImage.clone();
    lastDepthInfoImage_ = depthInfoImage.clone();
    
    // 更新ROI点索引
    updateROIPointIndices();
    
    Logger::debug("Depth images generated successfully");
    return true;
}

bool PointCloudDepthRenderer::loadPointCloud(const std::string& filePath, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    Logger::debug("Loading point cloud: " + filePath);
    
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(filePath, *cloud) == -1) {
        Logger::error("Could not read PLY file: " + filePath);
        return false;
    }
    
    // 移除无效点
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
    
    Logger::debug("Point cloud loaded successfully. Points: " + std::to_string(cloud->size()));
    return true;
}

void PointCloudDepthRenderer::transformPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    pcl::transformPointCloud(*cloud, *cloud, transformMatrix_);
}

bool PointCloudDepthRenderer::projectToDepthImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, cv::Mat& depthImage) {
    Logger::debug("Projecting point cloud to depth image");
    
    depthImage = cv::Mat::zeros(imageHeight_, imageWidth_, CV_32F);
    
    float cx = imageWidth_ / 2.0f;
    float cy = imageHeight_ / 2.0f;
    
    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& point = cloud->points[i];
        
        if (point.z >= 0) continue; // 忽略相机后面的点
        
        // 投影到图像平面
        float u = focalLength_ * point.x / (-point.z) + cx;
        float v = focalLength_ * point.y / (-point.z) + cy;
        
        int ui = static_cast<int>(u);
        int vi = static_cast<int>(v);
        
        if (ui >= 0 && ui < imageWidth_ && vi >= 0 && vi < imageHeight_) {
            float depth = usePlaneRelativeDepth_ ? 
                -(planeNormal_.dot(Eigen::Vector3f(point.x, point.y, point.z)) - planeDistance_) :
                -point.z;
            
            if (depth >= minDepth_ && depth <= maxDepth_) {
                if (depthImage.at<float>(vi, ui) == 0 || depth < depthImage.at<float>(vi, ui)) {
                    depthImage.at<float>(vi, ui) = depth;
                    updatePixelToPointMapping(ui, vi, i, depth, cv::Point3f(point.x, point.y, point.z));
                }
            }
        }
    }
    
    Logger::debug("Point cloud projected to depth image");
    return true;
}

cv::Mat PointCloudDepthRenderer::generateGrayscaleDepthImage(const cv::Mat& depthImage) {
    cv::Mat grayImage = cv::Mat::zeros(depthImage.size(), CV_8U);
    
    double minVal, maxVal;
    cv::minMaxLoc(depthImage, &minVal, &maxVal, nullptr, nullptr, depthImage > 0);
    
    if (maxVal > minVal) {
        for (int y = 0; y < depthImage.rows; ++y) {
            for (int x = 0; x < depthImage.cols; ++x) {
                float depth = depthImage.at<float>(y, x);
                if (depth > 0) {
                    int grayValue = static_cast<int>(255 * (depth - minVal) / (maxVal - minVal));
                    grayImage.at<uchar>(y, x) = cv::saturate_cast<uchar>(grayValue);
                }
            }
        }
    }
    
    return grayImage;
}

cv::Mat PointCloudDepthRenderer::generateColorDepthImage(const cv::Mat& depthImage) {
    cv::Mat grayImage = generateGrayscaleDepthImage(depthImage);
    cv::Mat colorImage;
    cv::applyColorMap(grayImage, colorImage, colorMapType_);
    
    // 将无效像素设为黑色
    for (int y = 0; y < depthImage.rows; ++y) {
        for (int x = 0; x < depthImage.cols; ++x) {
            if (depthImage.at<float>(y, x) == 0) {
                colorImage.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
            }
        }
    }
    
    return colorImage;
}

cv::Mat PointCloudDepthRenderer::generateDepthInfoImage(const cv::Mat& depthImage) {
    cv::Mat infoImage = generateColorDepthImage(depthImage);
    
    // 添加深度信息文本
    double minVal, maxVal;
    cv::minMaxLoc(depthImage, &minVal, &maxVal, nullptr, nullptr, depthImage > 0);
    
    std::string depthInfo = "Depth Range: " + 
        std::to_string(static_cast<int>(minVal)) + " ~ " + 
        std::to_string(static_cast<int>(maxVal)) + " mm";
    
    cv::putText(infoImage, depthInfo, cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    
    // 绘制ROI
    drawROIs(infoImage);
    
    return infoImage;
}

void PointCloudDepthRenderer::updatePixelToPointMapping(int u, int v, int pointIndex, float depth, const cv::Point3f& worldCoord) {
    if (v >= 0 && v < imageHeight_ && u >= 0 && u < imageWidth_) {
        pixelToPointMap_[v][u].pointIndex = pointIndex;
        pixelToPointMap_[v][u].depth = depth;
        pixelToPointMap_[v][u].worldCoord = worldCoord;
        pixelToPointMap_[v][u].isValid = true;
    }
}

void PointCloudDepthRenderer::clearPixelToPointMapping() {
    for (int y = 0; y < imageHeight_; ++y) {
        for (int x = 0; x < imageWidth_; ++x) {
            pixelToPointMap_[y][x] = PixelMapping();
        }
    }
}

// ROI相关实现
int PointCloudDepthRenderer::addROI(const cv::Rect& rect, const std::string& name, const cv::Scalar& color) {
    ROI roi;
    roi.id = nextROIId_++;
    roi.rect = rect;
    roi.name = name.empty() ? ("ROI_" + std::to_string(roi.id)) : name;
    roi.color = color;
    
    rois_.push_back(roi);
    
    // 如果已经有点云数据，立即更新点索引
    if (pointCloudLoaded_) {
        updateROIPointIndices();
    }
    
    Logger::debug("Added ROI: " + roi.name + " (ID: " + std::to_string(roi.id) + ")");
    return roi.id;
}

bool PointCloudDepthRenderer::removeROI(int roiId) {
    auto it = std::find_if(rois_.begin(), rois_.end(), 
                          [roiId](const ROI& roi) { return roi.id == roiId; });
    
    if (it != rois_.end()) {
        Logger::debug("Removed ROI: " + it->name + " (ID: " + std::to_string(roiId) + ")");
        rois_.erase(it);
        return true;
    }
    
    Logger::warning("ROI with ID " + std::to_string(roiId) + " not found");
    return false;
}

void PointCloudDepthRenderer::clearAllROIs() {
    rois_.clear();
    Logger::debug("Cleared all ROIs");
}

std::vector<ROI> PointCloudDepthRenderer::getAllROIs() const {
    return rois_;
}

void PointCloudDepthRenderer::updateROIPointIndices() {
    if (!pointCloudLoaded_) return;
    
    for (auto& roi : rois_) {
        roi.pointIndices.clear();
        
        for (int y = roi.rect.y; y < roi.rect.y + roi.rect.height && y < imageHeight_; ++y) {
            for (int x = roi.rect.x; x < roi.rect.x + roi.rect.width && x < imageWidth_; ++x) {
                const auto& mapping = pixelToPointMap_[y][x];
                if (mapping.isValid) {
                    roi.pointIndices.push_back(mapping.pointIndex);
                }
            }
        }
        
        Logger::debug("ROI " + roi.name + " contains " + std::to_string(roi.pointIndices.size()) + " points");
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudDepthRenderer::extractROIPointCloud(int roiId) const {
    pcl::PointCloud<pcl::PointXYZ>::Ptr roiCloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    auto it = std::find_if(rois_.begin(), rois_.end(), 
                          [roiId](const ROI& roi) { return roi.id == roiId; });
    
    if (it == rois_.end() || !pointCloudLoaded_) {
        Logger::warning("ROI with ID " + std::to_string(roiId) + " not found or no point cloud loaded");
        return roiCloud;
    }
    
    for (int pointIndex : it->pointIndices) {
        if (pointIndex >= 0 && pointIndex < currentPointCloud_->size()) {
            roiCloud->push_back(currentPointCloud_->points[pointIndex]);
        }
    }
    
    Logger::debug("Extracted ROI point cloud with " + std::to_string(roiCloud->size()) + " points");
    return roiCloud;
}

std::vector<int> PointCloudDepthRenderer::getROIPointIndices(int roiId) const {
    auto it = std::find_if(rois_.begin(), rois_.end(), 
                          [roiId](const ROI& roi) { return roi.id == roiId; });
    
    if (it != rois_.end()) {
        return it->pointIndices;
    }
    
    return std::vector<int>();
}

bool PointCloudDepthRenderer::saveROIPointCloud(int roiId, const std::string& outputPath) const {
    auto roiCloud = extractROIPointCloud(roiId);
    
    if (roiCloud->empty()) {
        Logger::error("ROI point cloud is empty, cannot save");
        return false;
    }
    
    if (pcl::io::savePLYFile(outputPath, *roiCloud) == -1) {
        Logger::error("Failed to save ROI point cloud to: " + outputPath);
        return false;
    }
    
    Logger::info("ROI point cloud saved to: " + outputPath);
    return true;
}

void PointCloudDepthRenderer::drawROIs(cv::Mat& image) const {
    for (const auto& roi : rois_) {
        cv::rectangle(image, roi.rect, roi.color, 2);
        
        // 绘制ROI名称
        cv::Point textPos(roi.rect.x, roi.rect.y - 5);
        if (textPos.y < 20) textPos.y = roi.rect.y + 20;
        
        cv::putText(image, roi.name, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.7, roi.color, 2);
    }
}

cv::Mat PointCloudDepthRenderer::generateROIOverlayImage(const cv::Mat& baseImage) const {
    cv::Mat overlayImage = baseImage.clone();
    drawROIs(overlayImage);
    return overlayImage;
}

// 设置函数
void PointCloudDepthRenderer::setColorMap(int colorMapType) {
    colorMapType_ = colorMapType;
    Logger::debug("Color map type set to: " + std::to_string(colorMapType));
}

void PointCloudDepthRenderer::setDepthRange(float minDepth, float maxDepth) {
    minDepth_ = minDepth;
    maxDepth_ = maxDepth;
    Logger::debug("Depth range set to: " + std::to_string(minDepth) + " ~ " + std::to_string(maxDepth));
}

void PointCloudDepthRenderer::setImageSize(int width, int height) {
    imageWidth_ = width;
    imageHeight_ = height;
    
    // 重新调整像素映射大小
    pixelToPointMap_.resize(imageHeight_);
    for (int i = 0; i < imageHeight_; ++i) {
        pixelToPointMap_[i].resize(imageWidth_);
    }
    
    Logger::debug("Image size set to: " + std::to_string(width) + "x" + std::to_string(height));
}

PixelMapping PointCloudDepthRenderer::getPixelMapping(int x, int y) const {
    if (y >= 0 && y < imageHeight_ && x >= 0 && x < imageWidth_) {
        return pixelToPointMap_[y][x];
    }
    return PixelMapping();
}

bool PointCloudDepthRenderer::isPixelValid(int x, int y) const {
    if (y >= 0 && y < imageHeight_ && x >= 0 && x < imageWidth_) {
        return pixelToPointMap_[y][x].isValid;
    }
    return false;
}

bool PointCloudDepthRenderer::createOutputDirectory(const std::string& dirPath) {
    try {
        fs::create_directories(dirPath);
        return true;
    } catch (const std::exception& e) {
        Logger::error("Failed to create directory: " + dirPath + " - " + e.what());
        return false;
    }
}

bool PointCloudDepthRenderer::saveDepthImage(const cv::Mat& image, const std::string& outputPath) {
    if (cv::imwrite(outputPath, image)) {
        Logger::debug("Saved depth image: " + outputPath);
        return true;
    } else {
        Logger::error("Failed to save depth image: " + outputPath);
        return false;
    }
}

bool PointCloudDepthRenderer::saveConfiguration(const std::string& configPath) const {
    std::ofstream file(configPath);
    if (!file.is_open()) {
        Logger::error("Cannot create configuration file: " + configPath);
        return false;
    }
    
    file << "# Point Cloud Depth Renderer Configuration\n\n";
    
    file << "[Camera]\n";
    file << "positionX=" << cameraX_ << "\n";
    file << "positionY=" << cameraY_ << "\n";
    file << "positionZ=" << cameraZ_ << "\n";
    file << "pitch=" << cameraPitch_ << "\n";
    file << "yaw=" << cameraYaw_ << "\n";
    file << "roll=" << cameraRoll_ << "\n\n";
    
    file << "[Image]\n";
    file << "width=" << imageWidth_ << "\n";
    file << "height=" << imageHeight_ << "\n";
    file << "focalLength=" << focalLength_ << "\n\n";
    
    file << "[Depth]\n";
    file << "minDepth=" << minDepth_ << "\n";
    file << "maxDepth=" << maxDepth_ << "\n";
    file << "colorMapType=" << colorMapType_ << "\n";
    file << "usePlaneRelativeDepth=" << (usePlaneRelativeDepth_ ? "true" : "false") << "\n\n";
    
    file << "[Plane]\n";
    file << "normalX=" << planeNormal_.x() << "\n";
    file << "normalY=" << planeNormal_.y() << "\n";
    file << "normalZ=" << planeNormal_.z() << "\n";
    file << "centerX=" << planeCenter_.x() << "\n";
    file << "centerY=" << planeCenter_.y() << "\n";
    file << "centerZ=" << planeCenter_.z() << "\n";
    file << "distance=" << planeDistance_ << "\n";
    
    file.close();
    Logger::info("Configuration saved to: " + configPath);
    return true;
}
