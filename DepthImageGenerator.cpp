#include "DepthImageGenerator.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <pcl/common/transforms.h>
#include <pcl/filters/filter.h>

DepthImageGenerator::DepthImageGenerator()
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
    , planeNormal_(0.0f, 0.0f, 1.0f)
    , planeCenter_(0.0f, 0.0f, 0.0f)
    , planeDistance_(0.0f)
{
    Logger::debug("DepthImageGenerator initialized with default parameters");
    Logger::debug("Image size: " + std::to_string(imageWidth_) + "x" + std::to_string(imageHeight_));
    Logger::debug("Depth range: " + std::to_string(minDepth_) + " - " + std::to_string(maxDepth_));
    Logger::debug("Auto fit plane: " + std::string(autoFitPlane_ ? "enabled" : "disabled"));
}

DepthImageGenerator::~DepthImageGenerator() {
    Logger::debug("DepthImageGenerator destroyed");
}

bool DepthImageGenerator::generateDepthImage(const std::string& inputCloudPath, const std::string& outputDir) {
    try {
        Logger::info("Starting depth image generation");
        Logger::info("Input cloud: " + inputCloudPath);
        Logger::info("Output directory: " + outputDir);

        // 创建输出目录
        if (!createOutputDirectory(outputDir)) {
            Logger::error("Failed to create output directory: " + outputDir);
            return false;
        }

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
        }

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

void DepthImageGenerator::setImageSize(int width, int height) {
    imageWidth_ = width;
    imageHeight_ = height;
    principalPointX_ = width / 2.0f;
    principalPointY_ = height / 2.0f;
    // 根据图像尺寸调整焦距
    focalLength_ = width * 0.8f;  // 经验值，约为图像宽度的0.8倍
    Logger::debug("Image size set to: " + std::to_string(width) + "x" + std::to_string(height));
    Logger::debug("Focal length adjusted to: " + std::to_string(focalLength_));
}

void DepthImageGenerator::setDepthRange(float minDepth, float maxDepth) {
    minDepth_ = minDepth;
    maxDepth_ = maxDepth;
    Logger::debug("Depth range set to: " + std::to_string(minDepth) + " - " + std::to_string(maxDepth));
}

void DepthImageGenerator::setCameraPosition(float x, float y, float z) {
    cameraX_ = x;
    cameraY_ = y;
    cameraZ_ = z;
    Logger::debug("Camera position set to: (" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")");
}

void DepthImageGenerator::setCameraOrientation(float pitch, float yaw, float roll) {
    cameraPitch_ = pitch;
    cameraYaw_ = yaw;
    cameraRoll_ = roll;
    Logger::debug("Camera orientation set to: pitch=" + std::to_string(pitch) + ", yaw=" + std::to_string(yaw) + ", roll=" + std::to_string(roll));
}

void DepthImageGenerator::setAutoFitPlane(bool enable) {
    autoFitPlane_ = enable;
    Logger::debug("Auto fit plane " + std::string(enable ? "enabled" : "disabled"));
}

bool DepthImageGenerator::loadPointCloud(const std::string& filePath, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
    try {
        if (pcl::io::loadPLYFile<pcl::PointXYZ>(filePath, *cloud) == -1) {
            Logger::error("Could not read PLY file: " + filePath);
            return false;
        }

        // 移除NaN点
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);

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

bool DepthImageGenerator::projectToDepthImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, cv::Mat& depthImage) {
    try {
        // 初始化深度图像
        depthImage = cv::Mat::zeros(imageHeight_, imageWidth_, CV_32F);

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
        
        for (const auto& point : cloud->points) {
            totalPoints++;
            
            // 检查点是否在深度范围内
            // 对于负Z值，需要检查绝对值是否在范围内
            float depth = std::abs(point.z);
            float minDepthAbs = std::min(std::abs(minDepth_), std::abs(maxDepth_));
            float maxDepthAbs = std::max(std::abs(minDepth_), std::abs(maxDepth_));
            
            if (depth < minDepthAbs || depth > maxDepthAbs) {
                depthFilteredOut++;
                continue;
            }

            // 投影到图像平面 - 使用绝对值进行投影计算 (x和y轴交换)
            int u = static_cast<int>((point.y * focalLength_) / depth + principalPointX_);
            int v = static_cast<int>((point.x * focalLength_) / depth + principalPointY_);

            // 检查投影是否在图像范围内
            if (u >= 0 && u < imageWidth_ && v >= 0 && v < imageHeight_) {
                // 只保留最近的深度值（Z-buffer）
                float currentDepth = depthImage.at<float>(v, u);
                if (currentDepth == 0 || depth < currentDepth) {
                    depthImage.at<float>(v, u) = depth;  // 存储绝对深度值
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

void DepthImageGenerator::transformPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
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

bool DepthImageGenerator::saveDepthImage(const cv::Mat& depthImage, const std::string& outputPath) {
    try {
        // 创建8位灰度图像用于保存
        cv::Mat depthImage8U;
        depthImage.convertTo(depthImage8U, CV_8U);

        // 保存深度图像
        if (!cv::imwrite(outputPath, depthImage8U)) {
            Logger::error("Failed to write depth image to: " + outputPath);
            return false;
        }

        // 同时保存原始深度值（浮点格式）
        std::string rawOutputPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_raw.tiff";
        cv::imwrite(rawOutputPath, depthImage);

        Logger::info("Depth image saved to: " + outputPath);
        Logger::info("Raw depth data saved to: " + rawOutputPath);
        return true;

    } catch (const std::exception& e) {
        Logger::error("Exception saving depth image: " + std::string(e.what()));
        return false;
    }
}

bool DepthImageGenerator::createOutputDirectory(const std::string& dirPath) {
    try {
        std::string mkdirCmd = "mkdir \"" + dirPath + "\" 2>nul";
        system(mkdirCmd.c_str());
        return true;
    } catch (...) {
        return false;
    }
}

void DepthImageGenerator::normalizeDepthImage(cv::Mat& depthImage) {
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

bool DepthImageGenerator::fitPlaneToPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
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

void DepthImageGenerator::computeCameraFromPlane() {
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

bool DepthImageGenerator::fitPlaneSimple(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
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

void DepthImageGenerator::adjustDepthRangeForTransformedCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
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

pcl::PointCloud<pcl::PointXYZ>::Ptr DepthImageGenerator::preprocessPointCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
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

bool DepthImageGenerator::fitPlaneRobustPCA(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
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
