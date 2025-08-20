#include "TemplatePointCloudProcessor.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>
#include <chrono>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>

TemplatePointCloudProcessor::TemplatePointCloudProcessor()
    : imageWidth_(640)
    , imageHeight_(480)
    , minDepth_(0.1f)
    , maxDepth_(10.0f)
    , cameraX_(0.0f)
    , cameraY_(0.0f)
    , cameraZ_(2.0f)
    , cameraPitch_(0.0f)
    , cameraYaw_(0.0f)
    , cameraRoll_(0.0f)
    , focalLength_(525.0f)
    , principalPointX_(320.0f)
    , principalPointY_(240.0f)
    , autoFitPlane_(true)
    , autoSaveConfig_(true)
    , planeNormal_(0.0f, 0.0f, 1.0f)
    , planeCenter_(0.0f, 0.0f, 0.0f)
    , planeDistance_(0.0f)
    , enableDepthVisualization_(true)
    , depthColorMapType_(cv::COLORMAP_JET)
    , usePlaneRelativeDepth_(false)  // 默认使用相机深度
    , planeRelativeMinDepth_(-1000.0f)
    , planeRelativeMaxDepth_(1000.0f)
    , isTemplateProcessingMode_(false)  // 默认非模板处理模式
    , nextROIId_(1)
    , originalPointCloud_(new pcl::PointCloud<pcl::PointXYZ>)
{
    Logger::debug("TemplatePointCloudProcessor initialized with default parameters");
    Logger::debug("Image size: " + std::to_string(imageWidth_) + "x" + std::to_string(imageHeight_));
    Logger::debug("Depth range: " + std::to_string(minDepth_) + " - " + std::to_string(maxDepth_));
    Logger::debug("Auto fit plane: " + std::string(autoFitPlane_ ? "enabled" : "disabled"));
    Logger::debug("Depth visualization: " + std::string(enableDepthVisualization_ ? "enabled" : "disabled"));
}

// 核心功能：模板点云预处理 - 拟合平面并生成配置文件
bool TemplatePointCloudProcessor::processTemplatePointCloud(const std::string& templateCloudPath, const std::string& configOutputPath) {
    try {
        Logger::info("Starting template point cloud preprocessing...");
        Logger::info("Template cloud: " + templateCloudPath);
        Logger::info("Configuration output: " + configOutputPath);

        // 启用模板处理模式（不进行深度过滤）
        isTemplateProcessingMode_ = true;
        Logger::debug("Template processing mode enabled - depth filtering disabled for template analysis");

        // 加载模板点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (!loadPointCloud(templateCloudPath, cloud)) {
            Logger::error("Failed to load template point cloud: " + templateCloudPath);
            isTemplateProcessingMode_ = false;  // 恢复正常模式
            return false;
        }

        Logger::info("Template point cloud loaded successfully with " + std::to_string(cloud->size()) + " points");

        // 强制启用平面拟合（这是模板预处理的核心）
        Logger::info("Fitting plane to template point cloud...");
        
        // 使用改进的鲁棒PCA方法拟合平面
        bool planeFitSuccess = false;
        if (fitPlaneRobustPCA(cloud)) {
            planeFitSuccess = true;
            Logger::info("Plane fitting successful using Robust PCA method");
        } else {
            Logger::warning("Robust PCA plane fitting failed, trying simple method...");
            if (fitPlaneSimple(cloud)) {
                planeFitSuccess = true;
                Logger::info("Plane fitting successful using simple method");
            } else {
                Logger::error("All plane fitting methods failed");
                return false;
            }
        }

        if (planeFitSuccess) {
            // 计算相机变换参数
            computeCameraFromPlane();
            
            // 调整深度范围以适应模板点云
            adjustPlaneRelativeDepthRange(cloud);
            
            Logger::debug("Template analysis results:");
            Logger::debug("Plane normal: (" + std::to_string(planeNormal_(0)) + ", " + 
                         std::to_string(planeNormal_(1)) + ", " + std::to_string(planeNormal_(2)) + ")");
            Logger::debug("Plane center: (" + std::to_string(planeCenter_(0)) + ", " + 
                         std::to_string(planeCenter_(1)) + ", " + std::to_string(planeCenter_(2)) + ")");
            Logger::debug("Camera position: (" + std::to_string(cameraX_) + ", " + 
                         std::to_string(cameraY_) + ", " + std::to_string(cameraZ_) + ")");

            // 生成模板深度图
            Logger::info("Generating template depth image...");
            
            // 变换点云到相机坐标系
            transformPointCloud(cloud);
            
            // 生成深度图像
            cv::Mat depthImage;
            if (projectToDepthImage(cloud, depthImage)) {
                Logger::info("Template depth image generated successfully");
                
                // 提取配置文件目录作为输出目录
                std::string outputDir = configOutputPath.substr(0, configOutputPath.find_last_of("/\\"));
                if (outputDir == configOutputPath) {
                    outputDir = ".";  // 如果没有找到目录分隔符，使用当前目录
                }
                
                // 保存原始深度数据
                std::string rawOutputPath = outputDir + "/template_depth_raw.tiff";
                cv::imwrite(rawOutputPath, depthImage);
                Logger::info("Raw template depth data saved to: " + rawOutputPath);
                
                // 如果启用了深度可视化，保存彩色深度图
                if (enableDepthVisualization_) {
                    cv::Mat colorDepthImage = generateColorDepthImage(depthImage);
                    if (!colorDepthImage.empty()) {
                        std::string colorOutputPath = outputDir + "/template_depth_color.png";
                        cv::imwrite(colorOutputPath, colorDepthImage);
                        Logger::info("Template color depth image saved to: " + colorOutputPath);
                    }
                    
                    cv::Mat enhancedGrayImage = generateEnhancedGrayscaleDepthImage(depthImage);
                    if (!enhancedGrayImage.empty()) {
                        std::string grayOutputPath = outputDir + "/template_depth_enhanced_gray.png";
                        cv::imwrite(grayOutputPath, enhancedGrayImage);
                        Logger::info("Template enhanced grayscale depth image saved to: " + grayOutputPath);
                    }
                }
            } else {
                Logger::warning("Failed to generate template depth image, but continuing with configuration save");
            }

            // 保存模板分析结果到配置文件
            if (saveCameraConfiguration(configOutputPath)) {
                Logger::info("Template processing completed successfully!");
                Logger::info("Configuration saved to: " + configOutputPath);
                Logger::info("This configuration can now be used for processing similar point clouds");
                
                // 恢复正常模式
                isTemplateProcessingMode_ = false;
                return true;
            } else {
                Logger::error("Failed to save template configuration to: " + configOutputPath);
                isTemplateProcessingMode_ = false;  // 恢复正常模式
                return false;
            }
        }

        isTemplateProcessingMode_ = false;  // 恢复正常模式
        return false;
        
    } catch (const std::exception& e) {
        Logger::error("Exception in template point cloud preprocessing: " + std::string(e.what()));
        isTemplateProcessingMode_ = false;  // 恢复正常模式
        return false;
    }
}

TemplatePointCloudProcessor::~TemplatePointCloudProcessor() {
    Logger::debug("TemplatePointCloudProcessor destroyed");
}

bool TemplatePointCloudProcessor::generateDepthImage(const std::string& inputCloudPath, const std::string& outputDir) {
    try {
        Logger::info("Starting depth image generation");
        Logger::info("Input cloud: " + inputCloudPath);
        Logger::info("Output directory: " + outputDir);

        // 创建输出目录
        if (!createOutputDirectory(outputDir)) {
            Logger::error("Failed to create output directory: " + outputDir);
            return false;
        }

        // 检查是否存在已保存的配置文件
        std::string configPath = outputDir + "/camera_config.ini";
        std::ifstream configCheck(configPath);
        if (configCheck.good() && autoFitPlane_) {
            Logger::info("Found existing camera configuration, attempting to load...");
            if (loadCameraConfiguration(configPath)) {
                Logger::info("Successfully loaded existing camera configuration - skipping plane fitting");
            } else {
                Logger::warning("Failed to load existing configuration - will perform new plane fitting");
            }
        }
        configCheck.close();

        // 加载点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        if (!loadPointCloud(inputCloudPath, cloud)) {
            Logger::error("Failed to load point cloud: " + inputCloudPath);
            return false;
        }

        Logger::info("Point cloud loaded successfully with " + std::to_string(cloud->size()) + " points");

        // 如果启用了自动平面拟合，先拟合平面
        if (autoFitPlane_) {
            Logger::info("Fitting plane to point cloud...");
            
            // 使用改进的鲁棒PCA方法
            if (!fitPlaneRobustPCA(cloud)) {
                Logger::warning("Robust PCA plane fitting failed, trying simple method...");
                if (!fitPlaneSimple(cloud)) {
                    Logger::error("All plane fitting methods failed, using default orientation");
                    // 使用默认平面设置
                    planeNormal_ = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
                    planeCenter_ = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
                    planeDistance_ = 0.0f;
                }
            }
            computeCameraFromPlane();
            
            // 自动保存配置到输出目录（如果启用）
            if (autoSaveConfig_) {
                std::string configPath = outputDir + "/camera_config.ini";
                if (saveCameraConfiguration(configPath)) {
                    Logger::info("Auto-saved camera configuration for future use");
                } else {
                    Logger::warning("Failed to auto-save camera configuration");
                }
            }
        }

        // 调整平面相对深度范围
        adjustPlaneRelativeDepthRange(cloud);

        // 变换点云到相机坐标系
        transformPointCloud(cloud);

        // 如果使用了平面拟合，自动调整深度范围以适应变换后的点云
        if (autoFitPlane_) {
            adjustDepthRangeForTransformedCloud(cloud);
        }

        // 生成深度图像
        cv::Mat depthImage;
        if (!projectToDepthImage(cloud, depthImage)) {
            Logger::error("Failed to project point cloud to depth image");
            return false;
        }

        // 归一化深度图像
        normalizeDepthImage(depthImage);

        // 保存深度图像
        std::string outputPath = outputDir + "/depth_image.png";
        if (!saveDepthImage(depthImage, outputPath)) {
            Logger::error("Failed to save depth image: " + outputPath);
            return false;
        }

        Logger::info("Depth image generated successfully: " + outputPath);
        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception in depth image generation: " + std::string(e.what()));
        return false;
    } catch (...) {
        Logger::error("Unknown exception in depth image generation");
        return false;
    }
}

void TemplatePointCloudProcessor::setImageSize(int width, int height) {
    imageWidth_ = width;
    imageHeight_ = height;
    principalPointX_ = width / 2.0f;
    principalPointY_ = height / 2.0f;
    // 根据图像尺寸调整焦距
    focalLength_ = width * 0.8f;  // 经验值，约为图像宽度的0.8倍
    
    // 初始化像素到点云的映射数组
    pixelToPointMap_.clear();
    pixelToPointMap_.resize(height, std::vector<PixelToPointMapping>(width));
    
    Logger::debug("Image size set to: " + std::to_string(width) + "x" + std::to_string(height));
    Logger::debug("Focal length adjusted to: " + std::to_string(focalLength_));
    Logger::debug("Pixel to point mapping array initialized");
}

void TemplatePointCloudProcessor::setDepthRange(float minDepth, float maxDepth) {
    minDepth_ = minDepth;
    maxDepth_ = maxDepth;
    Logger::debug("Depth range set to: " + std::to_string(minDepth) + " - " + std::to_string(maxDepth));
}

void TemplatePointCloudProcessor::setCameraPosition(float x, float y, float z) {
    cameraX_ = x;
    cameraY_ = y;
    cameraZ_ = z;
    Logger::debug("Camera position set to: (" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")");
}

void TemplatePointCloudProcessor::setCameraOrientation(float pitch, float yaw, float roll) {
    cameraPitch_ = pitch;
    cameraYaw_ = yaw;
    cameraRoll_ = roll;
    Logger::debug("Camera orientation set to: pitch=" + std::to_string(pitch) + ", yaw=" + std::to_string(yaw) + ", roll=" + std::to_string(roll));
}

void TemplatePointCloudProcessor::setAutoFitPlane(bool enable) {
    autoFitPlane_ = enable;
    Logger::debug("Auto fit plane " + std::string(enable ? "enabled" : "disabled"));
}

bool TemplatePointCloudProcessor::loadPointCloud(const std::string& filePath, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    try {
        if (pcl::io::loadPLYFile<pcl::PointXYZ>(filePath, *cloud) == -1) {
            Logger::error("Could not read PLY file: " + filePath);
            return false;
        }

        // 移除NaN点
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

        // 保存原始点云的副本用于未来的点云分割
        *originalPointCloud_ = *cloud;
        Logger::debug("Original point cloud saved for future segmentation");

        // 分析点云坐标范围
        if (!cloud->empty()) {
            float minX = cloud->points[0].x, maxX = cloud->points[0].x;
            float minY = cloud->points[0].y, maxY = cloud->points[0].y;
            float minZ = cloud->points[0].z, maxZ = cloud->points[0].z;
            
            for (const auto& point : cloud->points) {
                minX = std::min(minX, point.x);
                maxX = std::max(maxX, point.x);
                minY = std::min(minY, point.y);
                maxY = std::max(maxY, point.y);
                minZ = std::min(minZ, point.z);
                maxZ = std::max(maxZ, point.z);
            }
            
            Logger::debug("Point cloud coordinate ranges:");
            Logger::debug("X: [" + std::to_string(minX) + ", " + std::to_string(maxX) + "]");
            Logger::debug("Y: [" + std::to_string(minY) + ", " + std::to_string(maxY) + "]");
            Logger::debug("Z: [" + std::to_string(minZ) + ", " + std::to_string(maxZ) + "]");
        }

        Logger::debug("Loaded point cloud with " + std::to_string(cloud->size()) + " valid points");
        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception loading point cloud: " + std::string(e.what()));
        return false;
    }
}

bool TemplatePointCloudProcessor::projectToDepthImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, cv::Mat& depthImage) {
    try {
        // 初始化深度图像
        depthImage = cv::Mat::zeros(imageHeight_, imageWidth_, CV_32F);
        
        // 清空并初始化像素到点云映射
        clearPixelToPointMapping();

        int validProjections = 0;
        int totalPoints = 0;
        int depthFilteredOut = 0;
        int imageRangeFilteredOut = 0;
        
        Logger::debug("Projection parameters:");
        Logger::debug("Min depth (abs): " + std::to_string(std::min(std::abs(minDepth_), std::abs(maxDepth_))));
        Logger::debug("Max depth (abs): " + std::to_string(std::max(std::abs(minDepth_), std::abs(maxDepth_))));
        Logger::debug("Focal length: " + std::to_string(focalLength_));
        Logger::debug("Principal point: (" + std::to_string(principalPointX_) + ", " + std::to_string(principalPointY_) + ")");
        Logger::debug("Image size: " + std::to_string(imageWidth_) + "x" + std::to_string(imageHeight_));
        
        for (size_t pointIndex = 0; pointIndex < cloud->points.size(); ++pointIndex) {
            const auto& point = cloud->points[pointIndex];
            totalPoints++;
            
            // 投影计算使用相机距离
            float cameraDepth = std::abs(point.z);
            
            // 完全移除深度过滤 - 模板点云处理需要保留所有点
            // （原有的深度过滤逻辑已移除，专注于模板点云处理）
            
            // 投影到图像平面 - 使用绝对值进行投影计算 (x和y轴交换)
            int u = static_cast<int>((point.y * focalLength_) / cameraDepth + principalPointX_);
            int v = static_cast<int>((point.x * focalLength_) / cameraDepth + principalPointY_);

            // 检查投影是否在图像范围内
            if (u >= 0 && u < imageWidth_ && v >= 0 && v < imageHeight_) {
                // Z-buffer使用相机深度
                float currentCameraDepth = depthImage.at<float>(v, u);
                if (currentCameraDepth == 0 || cameraDepth < std::abs(currentCameraDepth)) {
                    float depthValueToStore = cameraDepth;  // 默认存储相机深度
                    
                    // 如果启用了平面相对深度模式，则计算并存储平面相对深度
                    if (usePlaneRelativeDepth_ && autoFitPlane_ && planeNormal_.norm() > 0) {
                        Eigen::Vector3f pointVec(point.x, point.y, point.z);
                        Eigen::Vector3f planeToPoint = pointVec - planeCenter_;
                        depthValueToStore = planeToPoint.dot(planeNormal_);
                    }
                    
                    depthImage.at<float>(v, u) = depthValueToStore;
                    
                    // 更新像素到点云的映射
                    cv::Point3f worldCoord(point.x, point.y, point.z);
                    updatePixelToPointMapping(u, v, static_cast<int>(pointIndex), depthValueToStore, worldCoord);
                    
                    validProjections++;
                }
            } else {
                imageRangeFilteredOut++;
            }
        }

        Logger::debug("Projection statistics:");
        Logger::debug("Total points: " + std::to_string(totalPoints));
        Logger::debug("Depth filtered out: " + std::to_string(depthFilteredOut));
        Logger::debug("Image range filtered out: " + std::to_string(imageRangeFilteredOut));
        Logger::debug("Successfully projected " + std::to_string(validProjections) + " points to depth image");
        return validProjections > 0;

    } catch (const std::exception& e) {
        Logger::error("Exception in point cloud projection: " + std::string(e.what()));
        return false;
    }
}

void TemplatePointCloudProcessor::transformPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    try {
        if (autoFitPlane_) {
            // 基于平面拟合结果创建变换矩阵
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();
            
            // 计算相机坐标系的基向量
            Eigen::Vector3f zAxis = planeNormal_;  // Z轴为平面法向量
            Eigen::Vector3f xAxis = Eigen::Vector3f(1, 0, 0);
            
            // 如果Z轴与世界坐标系X轴平行，则选择Y轴作为参考
            if (std::abs(zAxis.dot(xAxis)) > 0.9f) {
                xAxis = Eigen::Vector3f(0, 1, 0);
            }
            
            // 计算X轴（右方向）
            xAxis = xAxis - (xAxis.dot(zAxis)) * zAxis;
            xAxis.normalize();
            
            // 计算Y轴（上方向）
            Eigen::Vector3f yAxis = zAxis.cross(xAxis);
            yAxis.normalize();
            
            // 创建旋转矩阵
            Eigen::Matrix3f rotation;
            rotation.col(0) = xAxis;
            rotation.col(1) = yAxis;
            rotation.col(2) = zAxis;
            
            // 设置变换矩阵
            transform.linear() = rotation.transpose(); // 从世界坐标系到相机坐标系的旋转
            transform.translation() = -rotation.transpose() * 
                Eigen::Vector3f(cameraX_, cameraY_, cameraZ_); // 平移
            
            // 应用变换
            pcl::transformPointCloud(*cloud, *cloud, transform);
            
            Logger::debug("Point cloud transformed using plane-fitted camera coordinate system");
            
        } else {
            // 原有的变换逻辑
            Eigen::Affine3f transform = Eigen::Affine3f::Identity();
            
            // 平移变换
            transform.translation() << -cameraX_, -cameraY_, -cameraZ_;
            
            // 旋转变换（简化处理，只考虑绕Z轴的旋转）
            if (cameraYaw_ != 0.0f) {
                transform.rotate(Eigen::AngleAxisf(cameraYaw_, Eigen::Vector3f::UnitZ()));
            }
            
            // 应用变换
            pcl::transformPointCloud(*cloud, *cloud, transform);
            
            Logger::debug("Point cloud transformed using manual camera coordinate system");
        }
        
        // 分析变换后的坐标范围
        if (!cloud->empty()) {
            float minX = cloud->points[0].x, maxX = cloud->points[0].x;
            float minY = cloud->points[0].y, maxY = cloud->points[0].y;
            float minZ = cloud->points[0].z, maxZ = cloud->points[0].z;
            
            for (const auto& point : cloud->points) {
                minX = std::min(minX, point.x);
                maxX = std::max(maxX, point.x);
                minY = std::min(minY, point.y);
                maxY = std::max(maxY, point.y);
                minZ = std::min(minZ, point.z);
                maxZ = std::max(maxZ, point.z);
            }
            
            Logger::debug("Transformed point cloud coordinate ranges:");
            Logger::debug("X: [" + std::to_string(minX) + ", " + std::to_string(maxX) + "]");
            Logger::debug("Y: [" + std::to_string(minY) + ", " + std::to_string(maxY) + "]");
            Logger::debug("Z: [" + std::to_string(minZ) + ", " + std::to_string(maxZ) + "]");
        }

    } catch (const std::exception& e) {
        Logger::error("Exception in point cloud transformation: " + std::string(e.what()));
    }
}

bool TemplatePointCloudProcessor::saveDepthImage(const cv::Mat& depthImage, const std::string& outputPath) {
    try {
        // 保存原始灰度深度图像
        cv::Mat depthImage8U;
        depthImage.convertTo(depthImage8U, CV_8U);
        
        if (!cv::imwrite(outputPath, depthImage8U)) {
            Logger::error("Failed to write depth image to: " + outputPath);
            return false;
        }
        Logger::info("Grayscale depth image saved to: " + outputPath);

        // 保存原始深度值（浮点格式）
        std::string rawOutputPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_raw.tiff";
        cv::imwrite(rawOutputPath, depthImage);
        Logger::info("Raw depth data saved to: " + rawOutputPath);

        // 如果启用了深度可视化，保存彩色深度图和增强灰度图
        if (enableDepthVisualization_) {
            // 保存彩色深度图
            cv::Mat colorDepthImage = generateColorDepthImage(depthImage);
            if (!colorDepthImage.empty()) {
                std::string colorOutputPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_color.png";
                cv::imwrite(colorOutputPath, colorDepthImage);
                Logger::info("Color depth image saved to: " + colorOutputPath);
            }
            
            // 保存增强对比度的灰度深度图（用于分割）
            cv::Mat enhancedGrayImage = generateEnhancedGrayscaleDepthImage(depthImage);
            if (!enhancedGrayImage.empty()) {
                std::string grayOutputPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_enhanced_gray.png";
                cv::imwrite(grayOutputPath, enhancedGrayImage);
                Logger::info("Enhanced grayscale depth image saved to: " + grayOutputPath);
            }
            
            // 保存带信息的深度图
            cv::Mat infoImage = generateDepthInfoImage(depthImage);
            if (!infoImage.empty()) {
                std::string infoOutputPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_info.png";
                cv::imwrite(infoOutputPath, infoImage);
                Logger::info("Depth info image (with ROIs and legend) saved to: " + infoOutputPath);
            }
        }

        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception saving depth image: " + std::string(e.what()));
        return false;
    }
}

bool TemplatePointCloudProcessor::createOutputDirectory(const std::string& dirPath) {
    try {
        std::string mkdirCmd = "mkdir \"" + dirPath + "\" 2>nul";
        system(mkdirCmd.c_str());
        return true;
    } catch (...) {
        return false;
    }
}

void TemplatePointCloudProcessor::normalizeDepthImage(cv::Mat& depthImage) {
    try {
        // 找到非零的最小值和最大值
        double minVal, maxVal;
        cv::Mat mask = depthImage > 0;
        cv::minMaxLoc(depthImage, &minVal, &maxVal, nullptr, nullptr, mask);

        if (maxVal > minVal) {
            // 归一化到0-255范围
            cv::Mat normalizedImage;
            depthImage.copyTo(normalizedImage);
            
            // 将非零像素归一化
            normalizedImage.setTo(0, depthImage == 0); // 保持背景为0
            cv::Mat nonZeroMask = depthImage > 0;
            normalizedImage.setTo((depthImage - minVal) * 255.0 / (maxVal - minVal), nonZeroMask);
            
            normalizedImage.copyTo(depthImage);
            
            Logger::debug("Depth image normalized from range [" + std::to_string(minVal) + ", " + std::to_string(maxVal) + "] to [0, 255]");
        }

    } catch (const std::exception& e) {
        Logger::error("Exception in depth image normalization: " + std::string(e.what()));
    }
}

bool TemplatePointCloudProcessor::fitPlaneToPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    try {
        // 方法1: 使用PCA进行平面拟合（更轻量级，避免RANSAC的内存问题）
        Logger::debug("Using PCA-based plane fitting for better memory efficiency");
        
        // 极简下采样：只使用1000个点进行拟合
        pcl::PointCloud<pcl::PointXYZ>::Ptr sample_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        int step = std::max(1, static_cast<int>(cloud->size() / 1000)); // 只保留1000个点
        
        for (size_t i = 0; i < cloud->size(); i += step) {
            sample_cloud->points.push_back(cloud->points[i]);
        }
        sample_cloud->width = sample_cloud->points.size();
        sample_cloud->height = 1;
        sample_cloud->is_dense = false;
        
        Logger::debug("Sampled " + std::to_string(sample_cloud->size()) + " points for PCA plane fitting");
        
        // 计算质心
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*sample_cloud, centroid);
        planeCenter_ = Eigen::Vector3f(centroid[0], centroid[1], centroid[2]);
        
        // 计算协方差矩阵
        Eigen::Matrix3f covariance_matrix;
        pcl::computeCovarianceMatrixNormalized(*sample_cloud, centroid, covariance_matrix);
        
        // 进行特征值分解
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
        Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();
        Eigen::Vector3f eigen_values = eigen_solver.eigenvalues();
        
        // 最小特征值对应的特征向量就是平面法向量
        planeNormal_ = eigen_vectors.col(0); // 最小特征值的特征向量
        
        // 确保法向量指向合理方向
        if (planeNormal_[2] < 0) {
            planeNormal_ = -planeNormal_;
        }
        
        // 计算平面距离
        planeDistance_ = -planeNormal_.dot(planeCenter_);
        
        Logger::debug("PCA plane fitting completed:");
        Logger::debug("Normal: (" + std::to_string(planeNormal_[0]) + ", " + 
                     std::to_string(planeNormal_[1]) + ", " + std::to_string(planeNormal_[2]) + ")");
        Logger::debug("Center: (" + std::to_string(planeCenter_[0]) + ", " + 
                     std::to_string(planeCenter_[1]) + ", " + std::to_string(planeCenter_[2]) + ")");
        Logger::debug("Distance: " + std::to_string(planeDistance_));
        Logger::debug("Eigen values: " + std::to_string(eigen_values[0]) + ", " + 
                     std::to_string(eigen_values[1]) + ", " + std::to_string(eigen_values[2]));
        
        return true;
        
    } catch (const std::exception& e) {
        Logger::error("Exception in PCA plane fitting: " + std::string(e.what()));
        return false;
    } catch (...) {
        Logger::error("Unknown exception in PCA plane fitting");
        return false;
    }
}

void TemplatePointCloudProcessor::computeCameraFromPlane() {
    try {
        Logger::debug("Computing optimal camera position from fitted plane");
        
        // 计算点云在平面上的投影范围，以确定最佳相机距离
        // 这样可以确保所有点都包含在图像中
        
        // 创建平面的坐标系
        Eigen::Vector3f zAxis = planeNormal_;  // Z轴为平面法向量
        Eigen::Vector3f xAxis = Eigen::Vector3f(1, 0, 0);
        
        // 如果Z轴与世界坐标系X轴平行，则选择Y轴作为参考
        if (std::abs(zAxis.dot(xAxis)) > 0.9f) {
            xAxis = Eigen::Vector3f(0, 1, 0);
        }
        
        // 计算平面坐标系的基向量
        xAxis = xAxis - (xAxis.dot(zAxis)) * zAxis;
        xAxis.normalize();
        Eigen::Vector3f yAxis = zAxis.cross(xAxis);
        yAxis.normalize();
        
        // 估算点云在平面坐标系中的范围
        float maxRadius = 0.0f;
        Eigen::Vector3f relativePos = planeCenter_ - planeCenter_; // 相对于平面中心
        
        // 简单估算：使用点云的边界框对角线长度
        // 这个方法比精确计算快，但足够准确
        float estimated_range = 2000.0f; // 基于之前的坐标范围估算
        maxRadius = estimated_range * 0.7f; // 保守估计
        
        // 计算相机距离：确保整个点云都在视野内
        // 使用视野角度计算所需距离
        float fov_factor = 1.2f; // 视野因子，确保有足够边界
        float camera_distance = maxRadius / std::tan(std::atan(imageWidth_ / (2.0f * focalLength_))) * fov_factor;
        
        // 设置最小和最大距离限制
        camera_distance = std::max(200.0f, std::min(camera_distance, 2000.0f));
        
        // 设置相机位置：沿着平面法向量方向
        cameraX_ = planeCenter_[0] + planeNormal_[0] * camera_distance;
        cameraY_ = planeCenter_[1] + planeNormal_[1] * camera_distance;
        cameraZ_ = planeCenter_[2] + planeNormal_[2] * camera_distance;
        
        // 计算相机朝向：相机看向平面中心
        Eigen::Vector3f viewDirection = -planeNormal_; // 与法向量相反
        
        // 计算俯仰角和偏航角
        cameraPitch_ = std::asin(-viewDirection[2]);
        cameraYaw_ = std::atan2(viewDirection[1], viewDirection[0]);
        cameraRoll_ = 0.0f; // 假设没有滚转
        
        Logger::debug("Optimized camera computed from plane:");
        Logger::debug("Position: (" + std::to_string(cameraX_) + ", " + 
                     std::to_string(cameraY_) + ", " + std::to_string(cameraZ_) + ")");
        Logger::debug("Distance from plane: " + std::to_string(camera_distance));
        Logger::debug("Estimated max radius: " + std::to_string(maxRadius));
        Logger::debug("Orientation: pitch=" + std::to_string(cameraPitch_) + 
                     ", yaw=" + std::to_string(cameraYaw_) + ", roll=" + std::to_string(cameraRoll_));
        
    } catch (const std::exception& e) {
        Logger::error("Exception in optimized camera computation: " + std::string(e.what()));
    }
}

bool TemplatePointCloudProcessor::fitPlaneSimple(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    try {
        Logger::debug("Using simple statistical method for plane fitting (memory-safe)");
        
        // 计算点云的边界框和质心
        float minX = cloud->points[0].x, maxX = cloud->points[0].x;
        float minY = cloud->points[0].y, maxY = cloud->points[0].y;
        float minZ = cloud->points[0].z, maxZ = cloud->points[0].z;
        
        double sumX = 0, sumY = 0, sumZ = 0;
        
        // 每隔大步长采样，减少计算量
        int step = std::max(1, static_cast<int>(cloud->size() / 500)); // 只用500个点
        int sampleCount = 0;
        
        for (size_t i = 0; i < cloud->size(); i += step) {
            const auto& point = cloud->points[i];
            
            minX = std::min(minX, point.x);
            maxX = std::max(maxX, point.x);
            minY = std::min(minY, point.y);
            maxY = std::max(maxY, point.y);
            minZ = std::min(minZ, point.z);
            maxZ = std::max(maxZ, point.z);
            
            sumX += point.x;
            sumY += point.y;
            sumZ += point.z;
            sampleCount++;
        }
        
        // 计算质心
        planeCenter_ = Eigen::Vector3f(
            static_cast<float>(sumX / sampleCount),
            static_cast<float>(sumY / sampleCount),
            static_cast<float>(sumZ / sampleCount)
        );
        
        // 简单假设：基于Z坐标变化最小的方向作为法向量
        float rangeX = maxX - minX;
        float rangeY = maxY - minY;
        float rangeZ = maxZ - minZ;
        
        // 找到变化最小的维度作为主法向量
        if (rangeZ <= rangeX && rangeZ <= rangeY) {
            // Z方向变化最小，法向量主要在Z方向
            planeNormal_ = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        } else if (rangeY <= rangeX && rangeY <= rangeZ) {
            // Y方向变化最小
            planeNormal_ = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
        } else {
            // X方向变化最小
            planeNormal_ = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
        }
        
        // 计算平面距离
        planeDistance_ = -planeNormal_.dot(planeCenter_);
        
        Logger::debug("Simple plane fitting completed:");
        Logger::debug("Ranges: X=" + std::to_string(rangeX) + 
                     ", Y=" + std::to_string(rangeY) + 
                     ", Z=" + std::to_string(rangeZ));
        Logger::debug("Normal: (" + std::to_string(planeNormal_[0]) + ", " + 
                     std::to_string(planeNormal_[1]) + ", " + std::to_string(planeNormal_[2]) + ")");
        Logger::debug("Center: (" + std::to_string(planeCenter_[0]) + ", " + 
                     std::to_string(planeCenter_[1]) + ", " + std::to_string(planeCenter_[2]) + ")");
        Logger::debug("Sample count: " + std::to_string(sampleCount));
        
        return true;
        
    } catch (const std::exception& e) {
        Logger::error("Exception in simple plane fitting: " + std::string(e.what()));
        return false;
    } catch (...) {
        Logger::error("Unknown exception in simple plane fitting");
        return false;
    }
}

void TemplatePointCloudProcessor::adjustDepthRangeForTransformedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    try {
        Logger::debug("Adjusting depth range based on transformed point cloud");
        
        // 快速采样计算变换后点云的深度范围
        float minZ = std::numeric_limits<float>::max();
        float maxZ = std::numeric_limits<float>::lowest();
        
        int step = std::max(1, static_cast<int>(cloud->size() / 1000)); // 采样1000个点
        
        for (size_t i = 0; i < cloud->size(); i += step) {
            float z = std::abs(cloud->points[i].z); // 使用绝对值
            minZ = std::min(minZ, z);
            maxZ = std::max(maxZ, z);
        }
        
        if (minZ < maxZ) {
            // 添加一些边界，确保包含所有点
            float margin = (maxZ - minZ) * 0.1f; // 10%的边界
            float newMinDepth = std::max(0.1f, minZ - margin);
            float newMaxDepth = maxZ + margin;
            
            // 更新深度范围
            minDepth_ = newMinDepth;
            maxDepth_ = newMaxDepth;
            
            Logger::debug("Auto-adjusted depth range to: " + 
                         std::to_string(newMinDepth) + " - " + std::to_string(newMaxDepth));
            Logger::debug("Original Z range (abs): [" + std::to_string(minZ) + ", " + std::to_string(maxZ) + "]");
        } else {
            Logger::warning("Could not determine valid depth range from transformed cloud");
        }
        
    } catch (const std::exception& e) {
        Logger::error("Exception in depth range adjustment: " + std::string(e.what()));
    } catch (...) {
        Logger::error("Unknown exception in depth range adjustment");
    }
}

void TemplatePointCloudProcessor::adjustPlaneRelativeDepthRange(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    try {
        Logger::debug("Adjusting plane relative depth range based on point cloud");
        
        if (!autoFitPlane_ || planeNormal_.norm() == 0) {
            Logger::debug("No plane fitted, using Z coordinate range");
            // 如果没有平面拟合，使用Z坐标范围
            float minZ = std::numeric_limits<float>::max();
            float maxZ = std::numeric_limits<float>::lowest();
            
            for (const auto& point : cloud->points) {
                minZ = std::min(minZ, point.z);
                maxZ = std::max(maxZ, point.z);
            }
            
            float margin = (maxZ - minZ) * 0.1f;
            planeRelativeMinDepth_ = minZ - margin;
            planeRelativeMaxDepth_ = maxZ + margin;
        } else {
            // 计算所有点相对于拟合平面的距离
            float minRelativeDepth = std::numeric_limits<float>::max();
            float maxRelativeDepth = std::numeric_limits<float>::lowest();
            
            for (const auto& point : cloud->points) {
                Eigen::Vector3f pointVec(point.x, point.y, point.z);
                Eigen::Vector3f planeToPoint = pointVec - planeCenter_;
                float relativeDepth = planeToPoint.dot(planeNormal_);
                
                minRelativeDepth = std::min(minRelativeDepth, relativeDepth);
                maxRelativeDepth = std::max(maxRelativeDepth, relativeDepth);
            }
            
            // 添加边界
            float margin = (maxRelativeDepth - minRelativeDepth) * 0.1f;
            planeRelativeMinDepth_ = minRelativeDepth - margin;
            planeRelativeMaxDepth_ = maxRelativeDepth + margin;
        }
        
        Logger::debug("Plane relative depth range set to: " + 
                     std::to_string(planeRelativeMinDepth_) + " - " + std::to_string(planeRelativeMaxDepth_));
        
    } catch (const std::exception& e) {
        Logger::error("Exception in adjusting plane relative depth range: " + std::string(e.what()));
    }
}

pcl::PointCloud<pcl::PointXYZ>::Ptr TemplatePointCloudProcessor::preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    try {
        Logger::debug("Preprocessing point cloud for robust plane fitting");
        
        // 第一步：先下采样到500万点（提高后续处理效率）
        pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        size_t target_points = 5000000; // 500万点
        
        if (cloud->size() <= target_points) {
            // 如果点云已经小于等于500万，直接使用
            *downsampled_cloud = *cloud;
            Logger::debug("Point cloud size (" + std::to_string(cloud->size()) + 
                         ") is already within target, no initial downsampling needed");
        } else {
            // 计算采样步长进行下采样
            int step = static_cast<int>(cloud->size() / target_points);
            step = std::max(1, step);
            
            Logger::debug("Initial downsampling from " + std::to_string(cloud->size()) + 
                         " to approximately " + std::to_string(target_points) + " points (step=" + std::to_string(step) + ")");
            
            for (size_t i = 0; i < cloud->size(); i += step) {
                downsampled_cloud->points.push_back(cloud->points[i]);
            }
            
            downsampled_cloud->width = downsampled_cloud->points.size();
            downsampled_cloud->height = 1;
            downsampled_cloud->is_dense = false;
            
            Logger::debug("Initial downsampled to " + std::to_string(downsampled_cloud->size()) + " points");
        }
        
        // 第二步：在下采样后的点云上进行统计滤波去除离群点
        // pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        // pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        // sor.setInputCloud(downsampled_cloud);
        // sor.setMeanK(50);  // 考虑每个点的50个邻居
        // sor.setStddevMulThresh(2.0);  // 标准差阈值
        // sor.filter(*filtered_cloud);

        // Logger::debug("Statistical filtering: " + std::to_string(downsampled_cloud->size()) + 
        //              " -> " + std::to_string(filtered_cloud->size()) + " points");

        // 直接使用下采样后的点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud = downsampled_cloud;
        
        // 第三步：进一步采样用于PCA分析（保留足够的点进行准确分析）
        pcl::PointCloud<pcl::PointXYZ>::Ptr final_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        // 目标采样数量：100K点（既能保证精度又能控制计算量）
        int target_samples = 100000;
        
        if (filtered_cloud->size() <= target_samples) {
            // 如果点数不多，直接使用过滤后的点云
            final_cloud = filtered_cloud;
        } else {
            // 使用随机采样
            pcl::RandomSample<pcl::PointXYZ> random_sample;
            random_sample.setInputCloud(filtered_cloud);
            random_sample.setSample(target_samples);
            random_sample.filter(*final_cloud);
        }
        
        Logger::debug("Final sampling: " + std::to_string(filtered_cloud->size()) + 
                     " -> " + std::to_string(final_cloud->size()) + " points");
        
        Logger::debug("Point cloud preprocessing completed. Total processing: " + 
                     std::to_string(cloud->size()) + " -> " + std::to_string(final_cloud->size()) + " points");
        
        return final_cloud;
        
    } catch (const std::exception& e) {
        Logger::error("Exception in point cloud preprocessing: " + std::string(e.what()));
        Logger::warning("Using simple downsampling fallback");
        
        // 失败时使用简单下采样
        pcl::PointCloud<pcl::PointXYZ>::Ptr fallback_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        int step = std::max(1, static_cast<int>(cloud->size() / 10000)); // 1万个点
        for (size_t i = 0; i < cloud->size(); i += step) {
            fallback_cloud->points.push_back(cloud->points[i]);
        }
        fallback_cloud->width = fallback_cloud->points.size();
        fallback_cloud->height = 1;
        fallback_cloud->is_dense = false;
        
        return fallback_cloud;
    }
}

bool TemplatePointCloudProcessor::fitPlaneRobustPCA(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    try {
        Logger::debug("Using robust PCA-based plane fitting");
        
        // 预处理点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr processed_cloud = preprocessPointCloud(cloud);
        
        if (processed_cloud->empty()) {
            Logger::error("Preprocessed cloud is empty");
            return false;
        }
        
        // 使用PCL的PCA类进行更准确的分析
        pcl::PCA<pcl::PointXYZ> pca;
        pca.setInputCloud(processed_cloud);
        
        // 获取主成分
        Eigen::Matrix3f eigen_vectors = pca.getEigenVectors();
        Eigen::Vector3f eigen_values = pca.getEigenValues();
        Eigen::Vector4f mean = pca.getMean();
        
        // 最小特征值对应的特征向量是平面法向量
        planeNormal_ = eigen_vectors.col(2); // 第三列是最小特征值的特征向量
        planeCenter_ = Eigen::Vector3f(mean[0], mean[1], mean[2]);
        
        // 确保法向量指向合理方向
        if (planeNormal_[2] < 0) {
            planeNormal_ = -planeNormal_;
        }
        
        // 计算平面距离
        planeDistance_ = -planeNormal_.dot(planeCenter_);
        
        // 分析拟合质量
        float flatness = eigen_values[2] / (eigen_values[0] + eigen_values[1] + eigen_values[2]);
        
        Logger::debug("Robust PCA plane fitting completed:");
        Logger::debug("Normal: (" + std::to_string(planeNormal_[0]) + ", " + 
                     std::to_string(planeNormal_[1]) + ", " + std::to_string(planeNormal_[2]) + ")");
        Logger::debug("Center: (" + std::to_string(planeCenter_[0]) + ", " + 
                     std::to_string(planeCenter_[1]) + ", " + std::to_string(planeCenter_[2]) + ")");
        Logger::debug("Eigen values: " + std::to_string(eigen_values[0]) + ", " + 
                     std::to_string(eigen_values[1]) + ", " + std::to_string(eigen_values[2]));
        Logger::debug("Flatness ratio: " + std::to_string(flatness));
        Logger::debug("Processed points: " + std::to_string(processed_cloud->size()));
        
        // 检查拟合质量
        if (flatness < 0.01f) {  // 平面特征明显
            Logger::debug("Good plane fitting quality detected");
            return true;
        } else {
            Logger::warning("Plane fitting quality may be poor (flatness=" + std::to_string(flatness) + ")");
            return true; // 仍然使用结果，但给出警告
        }
        
    } catch (const std::exception& e) {
        Logger::error("Exception in robust PCA plane fitting: " + std::string(e.what()));
        return false;
    } catch (...) {
        Logger::error("Unknown exception in robust PCA plane fitting");
        return false;
    }
}

// ========== 新增功能实现 ==========

// 深度信息可视化设置
void TemplatePointCloudProcessor::setDepthVisualization(bool enable) {
    enableDepthVisualization_ = enable;
    Logger::debug("Depth visualization " + std::string(enable ? "enabled" : "disabled"));
}

void TemplatePointCloudProcessor::setDepthColorMap(int colorMapType) {
    depthColorMapType_ = colorMapType;
    Logger::debug("Depth color map set to type: " + std::to_string(colorMapType));
}

void TemplatePointCloudProcessor::setUsePlaneRelativeDepth(bool enable) {
    usePlaneRelativeDepth_ = enable;
    Logger::debug("Use plane relative depth " + std::string(enable ? "enabled" : "disabled"));
}

// 生成彩色深度图
cv::Mat TemplatePointCloudProcessor::generateColorDepthImage(const cv::Mat& depthImage) {
    try {
        cv::Mat colorDepthImage;
        
        // 首先找到非零像素的深度范围
        cv::Mat mask = depthImage != 0;
        cv::Mat nonZeroDepths;
        depthImage.copyTo(nonZeroDepths, mask);
        
        double minVal, maxVal;
        cv::minMaxLoc(nonZeroDepths, &minVal, &maxVal, nullptr, nullptr, mask);
        
        Logger::debug("Depth range for color mapping: " + std::to_string(minVal) + " to " + std::to_string(maxVal));
        
        if (maxVal > minVal) {
            // 增强深度对比度的参数
            double contrastFactor = 2.0;  // 增强对比度系数
            double range = maxVal - minVal;
            double center = (maxVal + minVal) / 2.0;
            
            // 创建增强的深度图
            cv::Mat enhancedDepth;
            depthImage.copyTo(enhancedDepth);
            
            // 应用对比度增强
            for (int i = 0; i < enhancedDepth.rows; ++i) {
                for (int j = 0; j < enhancedDepth.cols; ++j) {
                    float& val = enhancedDepth.at<float>(i, j);
                    if (val != 0) {
                        // 归一化到[-1, 1]范围
                        double normalizedVal = (val - center) / (range / 2.0);
                        
                        // 应用S型曲线增强对比度
                        double sign = (normalizedVal >= 0) ? 1.0 : -1.0;
                        double absVal = std::abs(normalizedVal);
                        double enhancedVal = sign * std::pow(absVal, 1.0 / contrastFactor);
                        
                        // 进一步放大范围
                        enhancedVal = std::tanh(enhancedVal * 1.5);  // 使用tanh函数进一步增强对比度
                        
                        // 重新映射回原始范围并扩展
                        val = static_cast<float>(center + enhancedVal * range * 0.8);
                    }
                }
            }
            
            // 重新计算增强后的范围
            cv::minMaxLoc(enhancedDepth, &minVal, &maxVal, nullptr, nullptr, mask);
            
            // 归一化到0-255范围
            cv::Mat normalizedDepth;
            enhancedDepth.convertTo(normalizedDepth, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
            
            // 应用颜色映射
            cv::applyColorMap(normalizedDepth, colorDepthImage, depthColorMapType_);
            
            // 在黑色区域（无深度信息的地方）设置为黑色
            cv::Mat zeroMask = depthImage == 0;
            colorDepthImage.setTo(cv::Scalar(0, 0, 0), zeroMask);
            
        } else {
            // 深度值都相同，创建单色图像
            colorDepthImage = cv::Mat::zeros(depthImage.size(), CV_8UC3);
            cv::Mat mask = depthImage != 0;
            colorDepthImage.setTo(cv::Scalar(128, 128, 128), mask);  // 灰色
        }
        
        Logger::debug("Generated enhanced color depth image with color map type: " + std::to_string(depthColorMapType_));
        return colorDepthImage;
        
    } catch (const std::exception& e) {
        Logger::error("Exception in generating color depth image: " + std::string(e.what()));
        return cv::Mat();
    }
}

// 生成增强对比度的灰度深度图（专为分割优化）
cv::Mat TemplatePointCloudProcessor::generateEnhancedGrayscaleDepthImage(const cv::Mat& depthImage) {
    try {
        cv::Mat enhancedGrayImage;
        
        // 找到非零像素的深度范围
        cv::Mat mask = depthImage != 0;
        cv::Mat nonZeroDepths;
        depthImage.copyTo(nonZeroDepths, mask);
        
        double minVal, maxVal;
        cv::minMaxLoc(nonZeroDepths, &minVal, &maxVal, nullptr, nullptr, mask);
        
        Logger::debug("Depth range for enhanced grayscale: " + std::to_string(minVal) + " to " + std::to_string(maxVal));
        
        if (maxVal > minVal) {
            // 初始化输出图像
            enhancedGrayImage = cv::Mat::zeros(depthImage.size(), CV_8U);
            
            double range = maxVal - minVal;
            
            // 计算直方图以进行自适应增强
            const int histSize = 256;
            std::vector<int> histogram(histSize, 0);
            
            // 构建深度直方图
            for (int i = 0; i < depthImage.rows; ++i) {
                for (int j = 0; j < depthImage.cols; ++j) {
                    float val = depthImage.at<float>(i, j);
                    if (val != 0) {
                        int binIndex = static_cast<int>((val - minVal) / range * (histSize - 1));
                        binIndex = std::max(0, std::min(histSize - 1, binIndex));
                        histogram[binIndex]++;
                    }
                }
            }
            
            // 计算累积分布函数用于直方图均衡化
            std::vector<float> cdf(histSize, 0);
            cdf[0] = histogram[0];
            for (int i = 1; i < histSize; ++i) {
                cdf[i] = cdf[i-1] + histogram[i];
            }
            
            // 归一化CDF
            int totalPixels = cv::countNonZero(mask);
            for (int i = 0; i < histSize; ++i) {
                cdf[i] = cdf[i] / totalPixels;
            }
            
            // 分段线性拉伸增强对比度
            for (int i = 0; i < depthImage.rows; ++i) {
                for (int j = 0; j < depthImage.cols; ++j) {
                    float val = depthImage.at<float>(i, j);
                    if (val != 0) {
                        // 归一化到[0,1]
                        double normalizedVal = (val - minVal) / range;
                        
                        // 应用多级对比度增强
                        double enhancedVal;
                        
                        if (normalizedVal < 0.2) {
                            // 近处区域：线性拉伸，增强细节
                            enhancedVal = normalizedVal * 2.5;  // 扩展到0.5
                        } else if (normalizedVal < 0.8) {
                            // 中等距离：S形曲线增强
                            double centered = (normalizedVal - 0.5) * 2.0;  // 映射到[-1,1]
                            double sigmoid = std::tanh(centered * 2.0) * 0.5 + 0.5;
                            enhancedVal = 0.2 + sigmoid * 0.6;  // 映射到[0.2, 0.8]
                        } else {
                            // 远处区域：压缩但保持可区分性
                            double farVal = (normalizedVal - 0.8) / 0.2;  // 归一化到[0,1]
                            enhancedVal = 0.8 + farVal * 0.2;  // 映射到[0.8, 1.0]
                        }
                        
                        // 进一步应用伽马校正增强中等亮度区域
                        double gamma = 0.7;  // 小于1的伽马值增强中等亮度
                        enhancedVal = std::pow(enhancedVal, gamma);
                        
                        // 应用直方图均衡化的影响（混合）
                        int binIndex = static_cast<int>(normalizedVal * (histSize - 1));
                        binIndex = std::max(0, std::min(histSize - 1, binIndex));
                        double histEqualizedVal = cdf[binIndex];
                        
                        // 混合增强值和直方图均衡化值
                        double finalVal = enhancedVal * 0.7 + histEqualizedVal * 0.3;
                        
                        // 转换到0-255范围并应用最终对比度增强
                        int grayVal = static_cast<int>(finalVal * 255);
                        
                        // 应用局部对比度增强（类似于锐化）
                        grayVal = std::max(0, std::min(255, static_cast<int>(grayVal * 1.2 - 25)));
                        
                        enhancedGrayImage.at<uchar>(i, j) = static_cast<uchar>(grayVal);
                    }
                }
            }
            
            // 应用轻微的高斯模糊以减少噪声，然后再次增强对比度
            cv::Mat blurred;
            cv::GaussianBlur(enhancedGrayImage, blurred, cv::Size(3, 3), 0.5);
            
            // 最终对比度增强
            cv::Mat finalEnhanced;
            blurred.convertTo(finalEnhanced, CV_8U, 1.3, -30);  // 增强对比度并减少亮度偏移
            
            // 确保黑色区域保持黑色
            cv::Mat zeroMask = depthImage == 0;
            finalEnhanced.setTo(0, zeroMask);
            
            Logger::debug("Generated enhanced grayscale depth image with advanced contrast enhancement");
            return finalEnhanced;
            
        } else {
            // 深度值都相同，创建单一灰度值图像
            enhancedGrayImage = cv::Mat::zeros(depthImage.size(), CV_8U);
            cv::Mat validMask = depthImage != 0;
            enhancedGrayImage.setTo(128, validMask);  // 中等灰度
        }
        
        return enhancedGrayImage;
        
    } catch (const std::exception& e) {
        Logger::error("Exception in generating enhanced grayscale depth image: " + std::string(e.what()));
        return cv::Mat();
    }
}

// 生成带深度信息的图像
cv::Mat TemplatePointCloudProcessor::generateDepthInfoImage(const cv::Mat& depthImage) {
    try {
        cv::Mat infoImage = generateColorDepthImage(depthImage);
        if (infoImage.empty()) {
            return cv::Mat();
        }
        
        // 绘制ROI
        drawROIs(infoImage);
        
        // 添加深度图例
        double minVal, maxVal;
        cv::minMaxLoc(depthImage, &minVal, &maxVal);
        drawDepthLegend(infoImage, minVal, maxVal);
        
        // 添加统计信息文本
        int nonZeroPixels = cv::countNonZero(depthImage > 0);
        std::string statsText = "Points: " + std::to_string(nonZeroPixels) + 
                               " | Depth: " + std::to_string(minVal) + "-" + std::to_string(maxVal);
        
        cv::putText(infoImage, statsText, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        
        Logger::debug("Generated depth info image with statistics and ROIs");
        return infoImage;
        
    } catch (const std::exception& e) {
        Logger::error("Exception in generating depth info image: " + std::string(e.what()));
        return cv::Mat();
    }
}

// 绘制深度图例
void TemplatePointCloudProcessor::drawDepthLegend(cv::Mat& image, double minVal, double maxVal) {
    try {
        int legendWidth = 20;
        int legendHeight = 200;
        int legendX = image.cols - legendWidth - 20;
        int legendY = 50;
        
        // 创建渐变色带
        cv::Mat legendBar(legendHeight, legendWidth, CV_8U);
        for (int i = 0; i < legendHeight; ++i) {
            int value = static_cast<int>(255.0 * i / legendHeight);
            legendBar.row(legendHeight - 1 - i) = value; // 反向，上方为最大值
        }
        
        // 应用颜色映射
        cv::Mat colorLegend;
        cv::applyColorMap(legendBar, colorLegend, depthColorMapType_);
        
        // 将图例复制到主图像
        cv::Rect legendRect(legendX, legendY, legendWidth, legendHeight);
        colorLegend.copyTo(image(legendRect));
        
        // 添加刻度标签
        cv::putText(image, "+" + std::to_string(maxVal), cv::Point(legendX + legendWidth + 5, legendY + 15),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        cv::putText(image, std::to_string(minVal), cv::Point(legendX + legendWidth + 5, legendY + legendHeight),
                   cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        
        // 添加零平面标记
        if (minVal < 0 && maxVal > 0) {
            int zeroY = legendY + static_cast<int>(legendHeight * (maxVal / (maxVal - minVal)));
            cv::line(image, cv::Point(legendX - 5, zeroY), cv::Point(legendX + legendWidth + 3, zeroY),
                    cv::Scalar(255, 255, 255), 2);
            cv::putText(image, "0 (plane)", cv::Point(legendX + legendWidth + 5, zeroY + 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }
        
        // 添加标题
        cv::putText(image, "Rel.Depth", cv::Point(legendX - 15, legendY - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
    } catch (const std::exception& e) {
        Logger::error("Exception in drawing depth legend: " + std::string(e.what()));
    }
}

// ROI管理功能
int TemplatePointCloudProcessor::addROI(const ImageROI& roi) {
    ImageROI newROI = roi;
    newROI.id = nextROIId_++;
    rois_.push_back(newROI);
    Logger::debug("Added ROI with ID: " + std::to_string(newROI.id) + ", name: " + newROI.name);
    return newROI.id;
}

bool TemplatePointCloudProcessor::removeROI(int roiId) {
    auto it = std::find_if(rois_.begin(), rois_.end(), 
                          [roiId](const ImageROI& roi) { return roi.id == roiId; });
    if (it != rois_.end()) {
        Logger::debug("Removed ROI with ID: " + std::to_string(roiId));
        rois_.erase(it);
        return true;
    }
    Logger::warning("ROI with ID " + std::to_string(roiId) + " not found for removal");
    return false;
}

void TemplatePointCloudProcessor::clearAllROIs() {
    rois_.clear();
    Logger::debug("Cleared all ROIs");
}

std::vector<ImageROI> TemplatePointCloudProcessor::getAllROIs() const {
    return rois_;
}

// 绘制ROI
void TemplatePointCloudProcessor::drawROIs(cv::Mat& image) const {
    for (const auto& roi : rois_) {
        // 绘制边界框
        cv::rectangle(image, roi.boundingBox, roi.color, 2);
        
        // 绘制轮廓（如果有）
        if (!roi.contour.empty()) {
            std::vector<std::vector<cv::Point>> contours = {roi.contour};
            cv::drawContours(image, contours, -1, roi.color, 2);
        }
        
        // 添加ROI标签
        cv::Point labelPos(roi.boundingBox.x, roi.boundingBox.y - 5);
        std::string label = roi.name.empty() ? ("ROI_" + std::to_string(roi.id)) : roi.name;
        cv::putText(image, label, labelPos, cv::FONT_HERSHEY_SIMPLEX, 0.6, roi.color, 2);
    }
}

// 像素到点云映射相关方法
void TemplatePointCloudProcessor::updatePixelToPointMapping(int u, int v, int pointIndex, float depth, const cv::Point3f& worldCoord) {
    if (v >= 0 && v < static_cast<int>(pixelToPointMap_.size()) && 
        u >= 0 && u < static_cast<int>(pixelToPointMap_[v].size())) {
        pixelToPointMap_[v][u].pointIndex = pointIndex;
        pixelToPointMap_[v][u].depth = depth;
        pixelToPointMap_[v][u].worldCoord = worldCoord;
    }
}

void TemplatePointCloudProcessor::clearPixelToPointMapping() {
    for (auto& row : pixelToPointMap_) {
        for (auto& pixel : row) {
            pixel = PixelToPointMapping(); // 重置为默认值
        }
    }
}

// 判断点是否在ROI内
bool TemplatePointCloudProcessor::isPointInROI(const cv::Point& point, const ImageROI& roi) const {
    // 首先检查边界框
    if (!roi.boundingBox.contains(point)) {
        return false;
    }
    
    // 如果有轮廓，检查点是否在轮廓内
    if (!roi.contour.empty()) {
        double result = cv::pointPolygonTest(roi.contour, point, false);
        return result >= 0; // 0表示在边界上，>0表示在内部
    }
    
    return true; // 如果只有边界框，则已经通过检查
}

// 获取ROI内的点索引（未来扩展接口）
std::vector<int> TemplatePointCloudProcessor::getPointIndicesInROI(int roiId) const {
    std::vector<int> pointIndices;
    
    auto it = std::find_if(rois_.begin(), rois_.end(), 
                          [roiId](const ImageROI& roi) { return roi.id == roiId; });
    if (it == rois_.end()) {
        Logger::warning("ROI with ID " + std::to_string(roiId) + " not found");
        return pointIndices;
    }
    
    const ImageROI& roi = *it;
    
    // 遍历ROI区域内的像素
    for (int v = roi.boundingBox.y; v < roi.boundingBox.y + roi.boundingBox.height; ++v) {
        for (int u = roi.boundingBox.x; u < roi.boundingBox.x + roi.boundingBox.width; ++u) {
            if (v >= 0 && v < static_cast<int>(pixelToPointMap_.size()) && 
                u >= 0 && u < static_cast<int>(pixelToPointMap_[v].size())) {
                
                cv::Point pixel(u, v);
                if (isPointInROI(pixel, roi)) {
                    const PixelToPointMapping& mapping = pixelToPointMap_[v][u];
                    if (mapping.pointIndex >= 0) {
                        pointIndices.push_back(mapping.pointIndex);
                    }
                }
            }
        }
    }
    
    Logger::debug("Found " + std::to_string(pointIndices.size()) + " points in ROI " + std::to_string(roiId));
    return pointIndices;
}

// ROI文件保存/加载（简单的JSON格式）
bool TemplatePointCloudProcessor::saveROIs(const std::string& filePath) const {
    try {
        std::ofstream file(filePath);
        if (!file.is_open()) {
            Logger::error("Could not open file for writing ROIs: " + filePath);
            return false;
        }
        
        file << "{\n";
        file << "  \"rois\": [\n";
        
        for (size_t i = 0; i < rois_.size(); ++i) {
            const auto& roi = rois_[i];
            file << "    {\n";
            file << "      \"id\": " << roi.id << ",\n";
            file << "      \"name\": \"" << roi.name << "\",\n";
            file << "      \"boundingBox\": [" << roi.boundingBox.x << ", " << roi.boundingBox.y 
                 << ", " << roi.boundingBox.width << ", " << roi.boundingBox.height << "],\n";
            file << "      \"minDepth\": " << roi.minDepth << ",\n";
            file << "      \"maxDepth\": " << roi.maxDepth << "\n";
            file << "    }";
            if (i < rois_.size() - 1) file << ",";
            file << "\n";
        }
        
        file << "  ]\n";
        file << "}\n";
        
        Logger::debug("Saved " + std::to_string(rois_.size()) + " ROIs to: " + filePath);
        return true;
        
    } catch (const std::exception& e) {
        Logger::error("Exception in saving ROIs: " + std::string(e.what()));
        return false;
    }
}

bool TemplatePointCloudProcessor::loadROIs(const std::string& filePath) {
    // 简化实现，未来可以使用JSON库
    Logger::info("ROI loading from file not yet implemented. Use addROI() method instead.");
    return false;
}

// 点云分割接口（预留实现）
bool TemplatePointCloudProcessor::segmentPointCloudByROI(int roiId, const std::string& outputPath) {
    try {
        std::vector<int> pointIndices = getPointIndicesInROI(roiId);
        if (pointIndices.empty()) {
            Logger::warning("No points found in ROI " + std::to_string(roiId));
            return false;
        }
        
        // 创建分割后的点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr segmentedCloud(new pcl::PointCloud<pcl::PointXYZ>);
        
        for (int index : pointIndices) {
            if (index >= 0 && index < static_cast<int>(originalPointCloud_->size())) {
                segmentedCloud->points.push_back(originalPointCloud_->points[index]);
            }
        }
        
        segmentedCloud->width = segmentedCloud->points.size();
        segmentedCloud->height = 1;
        segmentedCloud->is_dense = false;
        
        // 保存分割后的点云
        if (pcl::io::savePLYFile(outputPath, *segmentedCloud) == 0) {
            Logger::info("Segmented point cloud saved to: " + outputPath);
            Logger::info("Segmented " + std::to_string(segmentedCloud->size()) + " points from ROI " + std::to_string(roiId));
            return true;
        } else {
            Logger::error("Failed to save segmented point cloud to: " + outputPath);
            return false;
        }
        
    } catch (const std::exception& e) {
        Logger::error("Exception in segmenting point cloud by ROI: " + std::string(e.what()));
        return false;
    }
}

// 保存相机配置到文件
bool TemplatePointCloudProcessor::saveCameraConfiguration(const std::string& configPath) const {
    try {
        std::ofstream configFile(configPath);
        if (!configFile.is_open()) {
            Logger::error("Failed to open config file for writing: " + configPath);
            return false;
        }
        
        // 写入配置文件头部信息
        configFile << "# Template Point Cloud Analysis Configuration File\n";
        configFile << "# Generated on: " << std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()) << "\n";
        configFile << "# This file contains template plane fitting results and transformation settings\n";
        configFile << "# Use this configuration for processing similar point clouds with consistent orientation\n\n";
        
        // 基本图像参数
        configFile << "[ImageParameters]\n";
        configFile << "width=" << imageWidth_ << "\n";
        configFile << "height=" << imageHeight_ << "\n";
        configFile << "focal_length=" << focalLength_ << "\n";
        configFile << "principal_point_x=" << principalPointX_ << "\n";
        configFile << "principal_point_y=" << principalPointY_ << "\n\n";
        
        // 深度范围参数
        configFile << "[DepthParameters]\n";
        configFile << "min_depth=" << minDepth_ << "\n";
        configFile << "max_depth=" << maxDepth_ << "\n";
        configFile << "use_plane_relative_depth=" << (usePlaneRelativeDepth_ ? "true" : "false") << "\n";
        configFile << "plane_relative_min_depth=" << planeRelativeMinDepth_ << "\n";
        configFile << "plane_relative_max_depth=" << planeRelativeMaxDepth_ << "\n\n";
        
        // 相机位置和姿态
        configFile << "[CameraTransform]\n";
        configFile << "camera_x=" << cameraX_ << "\n";
        configFile << "camera_y=" << cameraY_ << "\n";
        configFile << "camera_z=" << cameraZ_ << "\n";
        configFile << "camera_pitch=" << cameraPitch_ << "\n";
        configFile << "camera_yaw=" << cameraYaw_ << "\n";
        configFile << "camera_roll=" << cameraRoll_ << "\n\n";
        
        // 平面拟合结果
        configFile << "[PlaneParameters]\n";
        configFile << "plane_normal_x=" << planeNormal_(0) << "\n";
        configFile << "plane_normal_y=" << planeNormal_(1) << "\n";
        configFile << "plane_normal_z=" << planeNormal_(2) << "\n";
        configFile << "plane_center_x=" << planeCenter_(0) << "\n";
        configFile << "plane_center_y=" << planeCenter_(1) << "\n";
        configFile << "plane_center_z=" << planeCenter_(2) << "\n";
        configFile << "plane_distance=" << planeDistance_ << "\n\n";
        
        // 可视化参数
        configFile << "[VisualizationParameters]\n";
        configFile << "enable_depth_visualization=" << (enableDepthVisualization_ ? "true" : "false") << "\n";
        configFile << "depth_color_map_type=" << depthColorMapType_ << "\n\n";
        
        // 其他设置
        configFile << "[OtherSettings]\n";
        configFile << "auto_fit_plane=" << (autoFitPlane_ ? "true" : "false") << "\n";
        
        configFile.close();
        
        Logger::info("Camera configuration saved to: " + configPath);
        Logger::debug("Saved camera position: (" + std::to_string(cameraX_) + ", " + 
                     std::to_string(cameraY_) + ", " + std::to_string(cameraZ_) + ")");
        Logger::debug("Saved plane normal: (" + std::to_string(planeNormal_(0)) + ", " + 
                     std::to_string(planeNormal_(1)) + ", " + std::to_string(planeNormal_(2)) + ")");
        
        return true;
        
    } catch (const std::exception& e) {
        Logger::error("Exception in saving camera configuration: " + std::string(e.what()));
        return false;
    }
}

// 从文件加载相机配置
bool TemplatePointCloudProcessor::loadCameraConfiguration(const std::string& configPath) {
    try {
        std::ifstream configFile(configPath);
        if (!configFile.is_open()) {
            Logger::error("Failed to open config file for reading: " + configPath);
            return false;
        }
        
        std::string line;
        std::string currentSection;
        
        while (std::getline(configFile, line)) {
            // 跳过注释和空行
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            // 检查是否是节标题
            if (line[0] == '[' && line.back() == ']') {
                currentSection = line.substr(1, line.length() - 2);
                continue;
            }
            
            // 解析键值对
            size_t equalPos = line.find('=');
            if (equalPos == std::string::npos) {
                continue;
            }
            
            std::string key = line.substr(0, equalPos);
            std::string value = line.substr(equalPos + 1);
            
            // 根据当前节解析参数
            if (currentSection == "ImageParameters") {
                if (key == "width") imageWidth_ = std::stoi(value);
                else if (key == "height") imageHeight_ = std::stoi(value);
                else if (key == "focal_length") focalLength_ = std::stof(value);
                else if (key == "principal_point_x") principalPointX_ = std::stof(value);
                else if (key == "principal_point_y") principalPointY_ = std::stof(value);
            }
            else if (currentSection == "DepthParameters") {
                if (key == "min_depth") minDepth_ = std::stof(value);
                else if (key == "max_depth") maxDepth_ = std::stof(value);
                else if (key == "use_plane_relative_depth") usePlaneRelativeDepth_ = (value == "true");
                else if (key == "plane_relative_min_depth") planeRelativeMinDepth_ = std::stof(value);
                else if (key == "plane_relative_max_depth") planeRelativeMaxDepth_ = std::stof(value);
            }
            else if (currentSection == "CameraTransform") {
                if (key == "camera_x") cameraX_ = std::stof(value);
                else if (key == "camera_y") cameraY_ = std::stof(value);
                else if (key == "camera_z") cameraZ_ = std::stof(value);
                else if (key == "camera_pitch") cameraPitch_ = std::stof(value);
                else if (key == "camera_yaw") cameraYaw_ = std::stof(value);
                else if (key == "camera_roll") cameraRoll_ = std::stof(value);
            }
            else if (currentSection == "PlaneParameters") {
                if (key == "plane_normal_x") planeNormal_(0) = std::stof(value);
                else if (key == "plane_normal_y") planeNormal_(1) = std::stof(value);
                else if (key == "plane_normal_z") planeNormal_(2) = std::stof(value);
                else if (key == "plane_center_x") planeCenter_(0) = std::stof(value);
                else if (key == "plane_center_y") planeCenter_(1) = std::stof(value);
                else if (key == "plane_center_z") planeCenter_(2) = std::stof(value);
                else if (key == "plane_distance") planeDistance_ = std::stof(value);
            }
            else if (currentSection == "VisualizationParameters") {
                if (key == "enable_depth_visualization") enableDepthVisualization_ = (value == "true");
                else if (key == "depth_color_map_type") depthColorMapType_ = std::stoi(value);
            }
            else if (currentSection == "OtherSettings") {
                if (key == "auto_fit_plane") autoFitPlane_ = (value == "true");
            }
        }
        
        configFile.close();
        
        // 重新初始化像素到点云映射数组
        clearPixelToPointMapping();
        pixelToPointMap_.resize(imageHeight_, std::vector<PixelToPointMapping>(imageWidth_));
        
        Logger::info("Camera configuration loaded from: " + configPath);
        Logger::debug("Loaded camera position: (" + std::to_string(cameraX_) + ", " + 
                     std::to_string(cameraY_) + ", " + std::to_string(cameraZ_) + ")");
        Logger::debug("Loaded plane normal: (" + std::to_string(planeNormal_(0)) + ", " + 
                     std::to_string(planeNormal_(1)) + ", " + std::to_string(planeNormal_(2)) + ")");
        Logger::info("Configuration loaded successfully. Auto plane fitting disabled for this session.");
        
        // 加载配置后禁用自动平面拟合
        autoFitPlane_ = false;
        
        return true;
        
    } catch (const std::exception& e) {
        Logger::error("Exception in loading camera configuration: " + std::string(e.what()));
        return false;
    }
}

// 检查是否有已保存的配置
bool TemplatePointCloudProcessor::hasSavedConfiguration() const {
    // 检查是否已经完成了平面拟合（有效的平面法向量）
    return (planeNormal_.norm() > 0.1f && !autoFitPlane_);
}
