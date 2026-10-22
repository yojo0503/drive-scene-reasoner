# GPU 环境安装与验证

> 最新进展（2026-10-22）：真实六视角推理、双卡短训练、验证与检查点保存已通过显存适配测试，见 [显存报告](VRAM_FIT.md)。下文保留环境安装/历史部署阶段的验证边界。
更新日期：2026-10-22。项目：`/home/yyh/Github/drive-scene-reasoner`。

## 当前状态

已建立独立的 `.venv-gpu` 环境，安装训练、推理、API 和测试依赖。依赖一致性、CUDA 基础运算、模型类和处理器加载、API 健康检查、CPU 回归测试及训练计划检查已通过。

尚未加载完整 8B 模型执行真实问答，也未启动 LoRA 训练。基础环境检查不能证明六视角推理显存足够、双卡训练能完成，或模型效果达标。执行实际任务前需重新确认 GPU 空闲情况。

## 为什么新建环境

先检查了现有项目环境：本项目 `.venv` 为 Python 3.13 CPU 环境；TSFReviewLab 环境为 Python 3.12、torch 2.4.1；风电项目环境使用不同版本的 torch/Transformers，且服务其他实验。没有一个现有环境直接满足当前配方。

复用 `/home/yyh/wind-power-research/envs/p2-py311/bin/python` 的 Python 3.11.15 解释器创建隔离 venv，不修改该解释器所属环境的包，也不使用 system-site-packages。这个 venv 依赖基础解释器路径继续存在；基础解释器迁移后应重建 venv。

## 使用已安装环境

```bash
cd /home/yyh/Github/drive-scene-reasoner
source .venv-gpu/bin/activate
python --version
python -m pip check
drivescene --help
```

原 `.venv` 保留。新环境不提交 Git，已加入 `.gitignore`。

## 重新安装

通用入口是 `scripts/setup_gpu_env.sh`，默认使用 `python3.11`。可指定解释器和包源：

```bash
cd /home/yyh/Github/drive-scene-reasoner
PYTHON_BIN=/home/yyh/wind-power-research/envs/p2-py311/bin/python \
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
bash scripts/setup_gpu_env.sh
```

脚本将 torch/torchvision CUDA 12.8 wheels 从 PyTorch 官方源安装，其他依赖由所选 Python 包源解析。本次服务器直连 pypi.org 超时，使用清华镜像完成安装。安装过程在专用 tmux 会话中执行，记录在 `outputs/setup/gpu-environment-install.log`。

关键版本：

| 组件 | 版本 |
| --- | --- |
| Python | 3.11.15 |
| torch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| Transformers / PEFT | 4.57.6 / 0.18.1 |
| Accelerate / DeepSpeed | 1.12.0 / 0.18.2 |
| ms-swift / TRL | 4.6.0 / 0.24.0 |
| qwen-vl-utils / Pillow | 0.0.14 / 11.3.0 |
| FastAPI / Uvicorn | 0.159.1 / 0.54.0 |

`requirements/gpu-cu128.txt` 是关键版本配方。`requirements/gpu-cu128.lock.txt` 是本次成功安装后的完整第三方包版本快照，排除了本项目 editable 路径。它限定包版本，不包含 wheel 内容哈希，不承诺跨平台复现。基础 CUDA wheels 装好后可按此快照安装其余依赖，再 `pip install -e '.[dev,vision,lora,api]'`。

服务器系统 nvcc 为 12.6，而 torch 自带 CUDA runtime 为 12.8。已通过的 CUDA 张量运算使用 torch runtime；本次没有验证 DeepSpeed 自定义 CUDA 算子编译或多卡通信。

## 验证证据

- `pip check`：无依赖冲突。
- 新环境 `pytest -q`：26 项通过。
- torch、torchvision、Transformers、PEFT、Accelerate、DeepSpeed、Swift 可导入。
- 在物理 GPU 3 上运行小张量计算并同步成功；进程退出后释放资源。
- 本地模型目录可离线加载 `Qwen3VLProcessor`，`Qwen3VLForConditionalGeneration` 类可导入。
- FastAPI TestClient 的 `/health`、`/v1/samples` 返回成功，识别官方推理清单 15480 个样本，模型未加载。
- 官方 demo 数据的训练计划 dry-run 通过，不启动训练。
- 实际导入 `SftArguments`、`sft_main` 和 `Seq2SeqTrainer` 成功；训练计划中全部参数名均存在于此版本的 SftArguments。此检查不实例化模型或启动训练。
- `swift sft --help` 可启动；其顶层帮助仅显示 tuner_backend，不作为完整参数兼容性证据。

详细记录在 `outputs/setup/gpu-environment-check.log`、`gpu-training-dry-run.txt`、`swift-sft-help.txt` 和 `gpu-pip-freeze.txt`。这些运行日志留在服务器，不提交 Git。

## 后续运行入口

先激活 `.venv-gpu`，用 `nvidia-smi` 确认有可用 GPU，再选择设备。下面的命令仍然只是训练计划检查：

```bash
python scripts/training_plan.py --dry-run \
  --model /home/yyh/models/Qwen3-VL-8B-Instruct
```

真实训练需要 `--execute`、tmux、干净的 Git 工作区和与 gpu_count 一致的 `CUDA_VISIBLE_DEVICES`。推理使用 `drivescene infer`，具体参数见 README；API 使用 `uvicorn drivescene.api:app`。本次环境搭建不代表这些完整模型任务已经完成验证。
