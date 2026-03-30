# 字幕生成工具 (GPU加速版)

基于 FunASR 和 LM Studio 的智能字幕生成工具，支持 GPU 加速，可自动将视频转录为带时间戳的 SRT/ASS 字幕文件。

## 功能特性

- **GPU 加速**：支持 NVIDIA CUDA 加速，大幅提升转录速度
- **多种格式输出**：支持 SRT 和 ASS 两种字幕格式
- **智能校对**：集成 LM Studio LLM 模型，自动优化字幕文本
- **批量处理**：支持多线程并行处理多个视频文件
- **多语言支持**：支持中文、英文、粤语、日语、韩语等自动检测

## 系统要求

### 硬件要求
- NVIDIA 显卡（推荐 4GB 以上显存，支持 CUDA）
- 如不使用 GPU，至少需要 8GB 以上内存

### 软件要求
- Python 3.8+
- FFmpeg（已包含在 FunASR 包中）
- Windows 10/11 或 Linux

## 安装步骤

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 下载 FunASR 模型

工具使用 SenseVoiceSmall 模型，首次运行会自动下载。如需手动配置：

```python
# 在 short-mt-gpu.py 中修改模型路径
MODEL_DIR = r"你的模型路径\SenseVoiceSmall"
```

### 3. 安装并启动 LM Studio

1. 下载并安装 [LM Studio](https://lmstudio.ai/)
2. 下载适合的校对模型（如 Qwen、Llama 等中文模型）
3. 启动 LM Studio 并加载模型，确保 API 服务运行在 `http://127.0.0.1:1234`

### 4. 配置 FFmpeg

工具已内置 FFmpeg，如遇问题可在配置区指定路径：

```python
FFMPEG_PATH = r"I:\AI\APP\FunASR\ffmpeg\bin"
```

## 使用方法

### 启动工具

```bash
python short-mt-gpu.py
```

### 操作步骤

1. **GPU 设置**：勾选"启用GPU加速"，选择 GPU 设备号
2. **输入设置**：点击"选择文件"或"选择文件夹"添加视频
3. **输出设置**：
   - 设置保存位置
   - 选择字幕格式（SRT 或 ASS）
   - 文件名格式：`原视频名.srt` 或 `原视频名.ass`
4. **处理设置**：根据硬件配置调整并行线程数和批处理大小
5. **LM Studio 设置**：确保已连接并选择校对模型
6. **校对提示词**：可使用默认提示词或根据需求调整
7. 点击"开始生成字幕"

## 字幕格式说明

### SRT 格式
标准字幕格式，兼容所有播放器：
```
1
00:00:01,000 --> 00:00:03,500
这是第一条字幕
```

### ASS 格式
高级字幕格式，支持样式和特效：
- 可自定义字体、大小、颜色
- 支持文字特效和定位
- 推荐用于有定制需求的用户

## 校对提示词

默认提示词已针对字幕格式优化，包含：
- 删除无意义标记
- 控制单条字幕长度
- 语义断句优化
- 移除冗余表达
- 使用【SPLIT】分隔输出

## 常见问题

### Q: GPU 不可用？
- 检查是否安装 CUDA 驱动
- 确认 PyTorch CUDA 版本匹配
- 可切换到 CPU 模式继续使用

### Q: LM Studio 连接失败？
- 确认 LM Studio 已启动并加载模型
- 检查 API 服务地址是否正确
- 查看防火墙设置

### Q: 生成的字幕时间轴不准？
- 尝试降低批处理大小
- 确保视频音频清晰
- 检查原始音频是否存在

## 依赖库

- funasr >= 1.0（语音识别）
- torch >= 2.0（深度学习框架）
- tkinter（图形界面，Python 内置）
- requests（HTTP 请求）

## 配置文件说明

在脚本顶部的配置区可以调整：

```python
# 模型路径
MODEL_DIR = r"I:\AI\APP\FunASR\models\SenseVoiceSmall"

# FFmpeg 路径
FFMPEG_PATH = r"I:\AI\APP\FunASR\ffmpeg\bin"

# LM Studio 地址
LM_STUDIO_URL = "http://127.0.0.1:1234/v1"

# GPU 设置
USE_GPU = True
GPU_DEVICE = "0"
```

## 许可证

本工具仅供个人学习研究使用。

## 更新日志

### v2.0 (2024)
- 新增 SRT/ASS 字幕格式支持
- 优化校对提示词为字幕格式
- 保留原始时间戳信息
- 界面布局优化
