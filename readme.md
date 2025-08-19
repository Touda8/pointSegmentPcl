# 3D文件转换工具 (STL/STEP to PLY Converter)

这是一个基于PCL和VTK库开发的C++工具，用于将3D模型文件转换为PLY格式。

## 功能特性

- **支持的输入格式**: .stl, .step, .stp
- **输出格式**: .ply
- **智能日志系统**: Debug模式显示详细信息，Release模式只显示关键信息
- **错误处理**: 转换失败时不会终止程序，只显示错误信息

## 编译和运行

### 方法1: 使用BAT文件（推荐）
1. 双击 `build_and_run.bat` 自动编译项目
2. 编译成功后显示使用说明

### 方法2: 手动编译
1. 打开"开发者命令提示符 for VS 2022"
2. 切换到项目目录
3. 运行: `msbuild 0820PointSegment.sln /p:Configuration=Debug /p:Platform=x64`

## 使用方法

### 基本语法
```
0820PointSegment.exe <输入文件路径> <输出文件路径>
```

### 使用示例

#### 转换STL文件
```
x64\Debug\0820PointSegment.exe point\input\target\ball801_1008.STL output\converted_ball.ply
```

#### 尝试转换STEP文件（目前不支持）
```
x64\Debug\0820PointSegment.exe point\input\target\ball801_1008.step output\converted_step.ply
```

### 带参数的BAT文件使用
```
build_and_run.bat "point\input\target\ball801_1008.STL" "output\converted.ply"
```

## 项目结构

```
├── main.cpp              # 主程序入口
├── Logger.h/cpp          # 日志系统
├── FileConverter.h/cpp   # 文件转换核心类
├── build_and_run.bat     # 自动编译运行脚本
└── pcl_vtk_debug.props   # PCL和VTK库配置
```

## 特性说明

### 日志系统
- **Debug模式**: 显示详细的调试信息，包括文件加载过程和点云数据
- **Release模式**: 只显示关键的信息和错误

### 文件格式支持
- **STL文件**: ✅ 完全支持，使用VTK库进行转换
- **STEP文件**: ❌ 目前不支持，需要额外的OpenCASCADE库

### 错误处理
- 输入文件不存在时会显示错误信息
- 不支持的文件格式会给出清晰的提示
- 输出目录不存在时会自动创建

## 技术依赖

- Visual Studio 2022
- PCL 1.12.1
- VTK 9.1
- C++17 标准

## 注意事项

1. 确保所有依赖库路径在 `pcl_vtk_debug.props` 中正确配置
2. 目前只支持STL格式的转换，STEP格式需要额外的库支持
3. 转换后的PLY文件为ASCII格式，便于查看和调试
