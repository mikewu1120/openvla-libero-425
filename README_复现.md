# OpenVLA：RTX 4080 Laptop 12GB 本地复现

目标：使用官方 LIBERO-Spatial 微调模型，在本机完成模型推理与机械臂仿真，无需真实机械臂，也无需重新训练。

后续小规模训练的配置、启动命令和当前状态见 [README_训练.md](README_训练.md)。

## 已完成的实测（2026-11-17）

**首个模型控制回合成功**：LIBERO-Spatial 的任务 0，要求把盘子与小碟之间的黑碗放到盘子上。使用官方第 0 个初始状态、随机种子 7，环境在第 **76 步**返回成功。

| 项目 | 实测结果 |
|---|---|
| 模型精度 / 注意力 | NF4 4-bit / SDPA |
| 平均每步模型推理耗时 | 0.4196 秒，约 2.38 次/秒（不含环境步进和预处理） |
| PyTorch 显存分配峰值 | 4.729 GiB |
| PyTorch 显存预留峰值 | 4.857 GiB |
| 运行期间整卡显存占用抽样 | 6864 MiB，含桌面、渲染及其他占用，不是峰值 |
| 结果视频 | `results/libero_spatial_task0/episode_0.mp4`，H.264，256×256，77 帧 |
| 详细结果 | `results/libero_spatial_task0/summary.json` |
| 每步动作记录 | `results/libero_spatial_task0/episode_0_actions.json`，76 条 |
| 控制台日志 | `results/inference_console.log` |

已检查视频可解码、截图方向正确、动作维度为 7 且数值有限、CUDA 和 NF4 运算成功、依赖无冲突。此结果说明本机能够运行该配置的推理和仿真；单次成功不代表完整基准成功率，也不说明本机已具备训练所需显存。

## 运行

在当前文件夹打开 PowerShell：

也可以直接双击 `运行仿真.cmd`。首次加载模型需要等待，命令窗口会打印加载与动作进度。

```powershell
# 一次完整回合：第 0 个任务，第 0 个官方初始状态，最多 220 步
.\.venv\Scripts\python.exe -u run_libero_4080.py

# 仅检查仿真场景渲染，不运行模型
.\.venv\Scripts\python.exe -u run_libero_4080.py --sim-only --steps 10 --output results/simulation_check

# 选择另一个 Spatial 任务；独立输出目录避免覆盖已有结果
.\.venv\Scripts\python.exe -u run_libero_4080.py --task-id 1 --episodes 1 --output results/spatial_task1
```

结果目录包含 MP4 视频、初始/最终截图、模型实际输入图、每步原始动作与执行动作、推理时间和 `summary.json`。视频按仿真控制频率 20fps 编码，不代表模型实际推理速度。`--sim-only` 视频只有空动作，用于验证环境，不能作为模型效果。

## 复现边界

- 原论文为 `2406.09246v3.pdf`，仿真实验见附录 E、表 12。
- 使用官方 `openvla/openvla-7b-finetuned-libero-spatial`，并非用真实机器人基础模型直接测试仿真。
- 为适配 Windows 和 12GB 显存，使用 PyTorch 2.5.1 CUDA 12.1、NF4 量化和 SDPA；论文使用 PyTorch 2.2.0、BF16、Flash Attention 2、A100。这些差异必须在报告中披露。
- 保留官方图像处理：旋转 180 度、JPEG 编解码、Lanczos3 缩放、90% 面积中心裁剪；保留动作反归一化、夹爪符号转换、10 步静置与 Spatial 的 220 步上限。
- 一个回合只是运行链路验证，不能声称复现了论文 84.7% 的 Spatial 成功率。论文每个任务组是 10 个任务 × 50 次回合 × 3 个随机种子。
- 程序只运行模拟器，不连接真实机械臂，不上传结果到 W&B。

## 为什么采用 QLoRA，而不是普通 LoRA

本机显卡是 **RTX 4080 Laptop GPU，12282MiB 显存**，不是桌面版 4080。官方 README 给出的普通 LoRA 微调最低显存约 27GB，超过本机容量。4-bit 推理和 4-bit 训练的显存需求不同：训练还需要激活值、梯度和优化器状态。

论文表 2 的 int4 约 7GB 数字来自较小的早期模型（见第 10 页脚注 4），不能当作当前 DinoSigLIP 模型的实测显存。本项目以实际运行日志为准。后续已实测通过一种受限的 QLoRA 配置：冻结视觉和基础权重，仅训练语言模型 q_proj/v_proj 的 rank=8 适配器，完成 50 次更新，PyTorch 显存分配峰值约 5.51GiB。详见 [训练实测结果](README_训练.md)。这不意味着 12GB 能直接运行论文的全量或普通 LoRA 训练配置。

## 文件与版本

- `openvla/`：官方源码，提交 `c8f03f48af692657d3060c19588038c7220e9af9`。
- `LIBERO/`：官方场景与仿真源码，提交 `8f1084e3132a39270c3a13ebe37270a43ece2a01`。
- `.venv/`：项目独立 Python 3.10 环境。
- `models/openvla-7b-finetuned-libero-spatial/`：官方权重，版本 `962318cec55ac10993ff0f5f43eda9a270b4c873`。
- `download_model.py`：下载器；支持分段续传，按官方 LFS SHA256 校验权重。运行 `python download_model.py` 可继续下载或校验。
- `run_libero_4080.py`：本机评估入口，直接加载官方模型定义，默认离线读取本地权重。
- `requirements-windows.lock.txt`：实际安装环境的完整版本锁定文件，`pip check` 已检查依赖一致性。
- `apply_windows_fixes.py`：针对 robosuite 1.4.1 的 Windows 修正。MuJoCo DLL 从实际安装位置加载；Windows 使用 GLFW，避免被错误强制设置为 EGL。只修改当前 `.venv` 内的依赖文件。

如需重建环境，在 Python 3.10 下创建 `.venv`，再执行：

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe apply_windows_fixes.py
.\.venv\Scripts\python.exe download_model.py
```

## 来源

- [OpenVLA 官方项目](https://github.com/openvla/openvla)
- [官方 Spatial 权重](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)
- [LIBERO 官方项目](https://github.com/Lifelong-Robot-Learning/LIBERO)
