# RTX 3090 显存适配实测

日期：2026-10-22。服务器：222.201.145.7。测试代码提交：`968e028c49e22b0821e325a6117221b5e87f7dc6`，环境为 `.venv-gpu`。测试全部在独立 tmux 会话中运行，未停止或修改已有实验。

## 结论与默认配置

Qwen3-VL-8B 已完成真实六视角推理；两张 RTX 3090 24 GB 已完成 LoRA 短训练、验证与检查点保存。

默认训练改为每图最多 **128 个视觉 token**（131072 pixels），每卡 batch=1、梯度累积=8、BF16、SDPA、梯度检查点、冻结视觉编码器和 aligner、LoRA rank=16/alpha=32、DeepSpeed ZeRO-3。最小图像预算仍为 64 token。最大序列长度仍为 4096。

256 token 配置也通过短训练，但出现过一次 allocator cache flush 警告；128 token 的预留显存峰值更低，适合先作为正式实验起点。降低图像预算可能影响细小目标和定位题效果，本次没有评估这种质量变化。

## 实测数据

下表单位为 GiB，使用 `torch.cuda.max_memory_allocated` 和 `max_memory_reserved`，不是整张卡的 nvidia-smi 总占用。CUDA 上下文、NCCL 等非 PyTorch 分配及其他用户进程不完全包含在这些数字内。

| 场景 | 每图上限 | allocated 峰值 | reserved 峰值 | 结果 |
| --- | ---: | ---: | ---: | --- |
| 单卡六图推理 | 64 token | 16.49 | 16.53 | 两题均成功 |
| 单卡六图推理 | 128 token | 16.60 | 16.63 | 两题均成功 |
| 单卡六图推理 | 256 token | 16.83 | 16.86 | 两题均成功 |
| 双卡 LoRA，rank 0 | 256 token | 17.16 | 17.65 | 2 次更新、2 次验证，无保存 |
| 双卡 LoRA，rank 1 | 256 token | 17.20 | 17.71 | 同上 |
| 双卡 LoRA，rank 0 | 128 token | 15.10 | 16.89 | 累积 8、1 次更新、验证及保存成功 |
| 双卡 LoRA，rank 1 | 128 token | 15.16 | 17.03 | 同上 |

推理使用物理 GPU 1，batch=1，最多生成 128 token；输入是本地验证集第一条有效题和全验证集中问题文本最长的一条题，各包含六路真实图像。没有以输出内容计算问答准确率。

256 token 训练使用物理 GPU 0、1，选择训练集中按消息字符长度排序的前 8 条样本，验证集选前 2 条；梯度累积=1，max_steps=2。128 token 训练使用同样设备，选择前 16 条训练样本及 2 条验证样本；梯度累积=8，max_steps=1，完成一轮 16 样本训练并保存。按字符长度选择不等同于严格按 tokenizer 后的长度选择。

两次训练都使用每卡 batch=1、4096 长度上限、相同的 BF16/LoRA/ZeRO-3 设置。测试样本没有保证填满 4096 token，因此这不是“任意 4096 长输入都不会 OOM”的证明。正式长训练、不同图像宽高比、更大 batch、更长生成和检查点恢复仍需分别验证。

两档训练的梯度范数均非零。128 token 运行 train_loss=1.10957、eval_loss=1.12010，仅说明短测试完成，不代表泛化效果。

## 与其他任务共用 GPU

开始时 GPU 0、1 各约有 22 GB 空闲，仍存在其他进程。测试为每个进程设置 `torch.cuda.set_per_process_memory_fraction(0.78)`，将 PyTorch 分配器限制在约 18.72 GiB；这不是整卡硬隔离，也不能约束其他进程或全部 NCCL 分配。

正式运行前重新执行 `nvidia-smi`，确认所选每张卡至少约 20 GiB 空闲，并继续观察其他任务的动态占用。普通项目 CLI/训练启动器没有自动施加本次探针的分配器上限。显存预留与其他任务变化可能使实际峰值不同。

## 运行方式

先在项目根目录激活 `.venv-gpu`。推理可使用：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 \
drivescene infer --manifest data/processed/val_manifest.jsonl \
  --model /home/yyh/models/Qwen3-VL-8B-Instruct --local-files-only \
  --device-map cuda:0 --min-pixels 65536 --max-pixels 131072 \
  --max-new-tokens 128 --limit 2 --output outputs/infer-smoke.jsonl
```

设备号仅是本次测试示例，运行前需重新选择。训练默认 JSON 已改为 128 token，先检查计划：

```bash
python scripts/training_plan.py --dry-run \
  --model /home/yyh/models/Qwen3-VL-8B-Instruct
```

正式训练在 tmux 内运行，保持干净 Git 工作区，显式指定两个空闲 GPU，再传入 `--execute`。默认将使用完整数据与两轮训练，本次没有启动此长实验。

## 结果与复现材料

- `outputs/vram-fit/inference-results.json`：六次真实推理结果、时间和显存。
- `outputs/vram-fit/train256-rank0.json`、`train256-rank1.json`：256 token 训练进程峰值。
- `outputs/vram-fit/train-rank0.json`、`train-rank1.json`：128 token 训练及保存全过程峰值。
- `outputs/vram-fit/inference.log`、`training.log`、`training128.log`：执行日志。
- `outputs/vram-fit/probe_infer.py`、`probe_train.py`、`run_train.sh`、`run_train128.sh`：本次探针及实际命令；前一次 8 条样本可由当前按长度排序的 16 条训练样本取前 8 条恢复。
- `outputs/vram-fit/train128-run/checkpoint-1`：显存测试产生的 LoRA/训练状态检查点，不能视作有质量保证的训练成品；本次未测试重载推理或恢复训练。
- `docs/benchmarks/vram-fit-20261022.json`：提交到 Git 的精简量化结果，不含原始标注或模型输出。

运行数据、权重和完整日志留在被 Git 忽略的 outputs 目录；只提交配置与说明。
