# 12GB 显卡上的小规模 QLoRA 训练

目标：继续微调已经下载的 OpenVLA LIBERO-Spatial 模型，让本机完成真实示范数据上的前向计算、反向传播、梯度更新、适配器保存和重新加载评估。这不是从零训练，也不是论文完整训练配置。

当前状态（2026-11-17）：已完成一次显存探测和第一轮 **50 次优化器更新**，并保存 LoRA 适配器。检查了 50 条训练日志、非零且有限的梯度、适配器参数确实变化，以及保存文件包含全部 128 个张量、4,194,304 个参数。

## 本轮实测

| 指标 | 训练前 | 训练后 |
|---|---:|---:|
| 固定训练样本损失 | 0.11673 | 0.08054 |
| 固定验证样本损失 | 0.10777 | 0.09654 |
| 固定训练样本动作 token 准确率 | 95.83% | 97.62% |
| 固定验证样本动作 token 准确率 | 96.43% | 96.43% |

每个分割只抽取固定的 24 帧用于上述前后对比，并非全部 558/180 帧；准确率按 7 个动作 token 计算。50 次更新、每次累积 4 帧，共采样训练 200 帧，未完整遍历一个 epoch。损失下降不能等同于仿真成功率提高。

训练主体及加载、前后打分和保存共耗时 **257 秒**（不含 Python 启动导入）。PyTorch 显存分配峰值 **5.5135GiB**，预留峰值 **5.7324GiB**；训练期间整卡显存占用一次抽样为 **7311MiB**，包括桌面和其他占用，不是整卡峰值。

- 权重：`training_runs/qlora_task0/adapter/adapter_model.safetensors`，约 16.8MB。
- 详细统计：`training_runs/qlora_task0/summary.json`。
- 训练曲线：`training_runs/qlora_task0/training_metrics.png`。
- 控制台记录：`training_runs/qlora_task0_console.log`。

## 保存后重新加载的仿真验证

已在新的 Python 进程中重新加载磁盘上的适配器，使用相同的 Spatial 任务 0、官方初始状态 0、随机种子 7 完成一个回合。环境返回成功，视频可解码。

| 回合 | 结果 | 动作步数 |
|---|---|---:|
| 原底座（此前实测） | 成功 | 76 |
| 加载本次 LoRA 后 | 成功 | 78 |

此单回合验证没有显示步数改善，也不足以比较成功率。它证明保存的适配器可以被加载并实际控制仿真机械臂。

- 视频：`results/qlora_task0/episode_0.mp4`，256×256，79 帧，20fps，按仿真时间编码。
- 动作和耗时：`results/qlora_task0/episode_0_actions.json`；平均每次推理约 0.485 秒。
- 仿真摘要：`results/qlora_task0/summary.json`。
- 运行入口：双击 `运行训练后的模型.cmd`。

## 数据与配置

- 官方来源：`openvla/modified_libero_rlds`，只下载了 Spatial 的前两个分片，约 236MB；固定版本和 SHA256 记录在 `datasets/training_subset/raw/manifest.json`。
- 任务：将盘子与小碟之间的黑碗放到盘子上。
- 5 条完整轨迹用于训练，共 558 帧；另外 2 条轨迹用于本次验证，共 180 帧。分割清单位于 `datasets/training_subset/manifest.json`。
- 底座已经被作者在 LIBERO-Spatial 上训练过，所以这里的验证数据只是没有参加本次参数更新，**不能称为底座从未见过的数据**。
- 图像采用官方已清理、旋转后的示范图；缩放至 224×224 后做 90% 面积中心裁剪。本次没有随机图像增强。
- 夹爪转换、动作离散化、文本提示和损失掩码遵循官方代码；复用当前底座的动作统计量。
- NF4 量化、SDPA、梯度检查点；冻结视觉编码器及基础权重，仅训练语言模型 `q_proj`、`v_proj` 的 LoRA，rank=8、alpha=16。
- 单步 batch=1，累积 4 次梯度后更新，学习率 0.0001，默认 50 次优化器更新。
- 训练前后在相同的固定样本上记录损失与动作 token 准确率。这些离线指标不是仿真任务成功率。

## 运行

先退出占用显存的游戏或其他 GPU 程序。脚本在可用显存不足 8GiB 时会停止，不会关闭其他程序。

```powershell
# 先验证一次真正的反向传播与参数更新
.\.venv\Scripts\python.exe -u train_qlora_4080.py --steps 1 --accumulation 1 --eval-frames 4 --output training_runs/memory_probe

# 第一轮 50 次更新，输出目录必须不存在
.\.venv\Scripts\python.exe -u train_qlora_4080.py

# 加载生成的适配器运行仿真，底座保持不变
.\.venv\Scripts\python.exe -u run_libero_4080.py --adapter training_runs/qlora_task0/adapter --output results/qlora_task0
```

`training_runs/qlora_task0` 和 `training_runs/memory_probe` 已有实测结果。为防止覆盖，再次训练时请用 `--output training_runs/qlora_task0_run2` 等新目录；默认命令和 `开始小规模训练.cmd` 会拒绝覆盖已有结果。运行已训练模型可双击 `运行训练后的模型.cmd`，原来的 `运行仿真.cmd` 仍使用未加本次适配器的底座。

## 输出与判断

- `training_runs/<运行名称>/adapter/`：训练完成后的 LoRA 权重。它不是完整 7B 模型，必须与原底座一起加载。
- `metrics.jsonl`：每次参数更新的训练损失、梯度范数、显存峰值与耗时。
- `summary.json`：训练前后固定样本的指标、参数变化量、完成步数以及成功或失败状态。

训练脚本会检查动作 token 编码可逆、损失和梯度有限、梯度非零、适配器参数确实发生变化。完成这些检查只证明训练链路有效。模型是否更好，仍需查看验证指标和多个仿真回合；少量示范继续训练也可能造成过拟合或能力退化。

## 依赖与来源

在已有推理环境上新增 `peft==0.11.1`。重新提取数据可运行 `prepare_training_data.py`，该步骤只使用 CPU。

- [OpenVLA 官方微调脚本](https://github.com/openvla/openvla/blob/main/vla-scripts/finetune.py)
- [官方处理后的示范数据](https://huggingface.co/datasets/openvla/modified_libero_rlds)
- [PEFT 量化训练说明](https://huggingface.co/docs/peft/v0.11.0/en/developer_guides/quantization)
