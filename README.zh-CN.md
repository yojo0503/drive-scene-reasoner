# Drive Scene Reasoner — 中文快速开始

**定位**：在 DriveLM-nuScenes 数据基础上，建立独立的自动驾驶**场景理解与推理**研究项目。第一阶段研究多模态问答，不输出可以用于实际驾驶控制的指令。

当前 v0.1 是**独立编写的起步工程**，不是 DriveLM 仓库的复制，也不是已经训练好的模型。已经验证：DriveLM 风格数据解析、按 scene 划分训练/验证集、训练 JSONL 转换、本地简单评测、CPU 测试。尚未验证：真实 Qwen3-VL-8B 模型加载、3090 的显存占用、LoRA 收敛效果。

完整使用说明请看 [README.md](README.md)。

## 目录

```text
src/drivescene/
  data.py      # DriveLM 数据适配、路径处理、scene-level split
  infer.py     # Qwen3-VL 多图推理适配器（GPU 尚未实测）
  eval.py      # 本地 Exact Match / Token-F1（不是官方榜单得分）
  cli.py       # prepare / infer / eval 命令行
  api.py       # 本地 FastAPI 服务
scripts/
  make_toy_data.py  # 生成合成数据进行无 GPU 验证
  train_lora.sh     # ms-swift LoRA 起始脚本（尚未 GPU 实测）
  publish_with_gh.sh  # 用 GitHub CLI 创建私有仓库并推送
```

## 本地启动

建议 Linux + Python 3.11。运行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pip install pillow
python scripts/make_toy_data.py --output data/toy

drivescene prepare \
  --input data/toy/drivelm_train.json \
  --image-root data/toy/nuscenes \
  --output data/processed --cameras front

pytest -q
```

模拟图片专门用于格式测试，不是 DriveLM 数据，更不是有效的智驾评测。

## 真实 DriveLM 数据

从[原项目 challenge 文档](https://github.com/OpenDriveLab/DriveLM/tree/main/challenge)和[数据集页面](https://huggingface.co/datasets/OpenDriveLab/DriveLM)下载授权数据，按 `data/drivelm/v1_1_train_nus.json` 与 `data/nuscenes/samples/CAM_*/...` 整理。

```bash
drivescene prepare \
  --input data/drivelm/v1_1_train_nus.json \
  --image-root data/nuscenes \
  --output data/processed \
  --cameras all
```

**注意**：官方 v1.1 challenge validation 只有问题、没有公开答案。项目默认从含标注的训练集按 **scene** 做本地划分，避免同一场景的问题同时出现在训练和验证中。`--cameras front` 可以降低调试成本，但会缺失侧面和后方交通证据。

## 第一个 GPU 验收目标

先使用少量真实数据（例如 20 条），运行基座 Qwen3-VL-8B 的推理，再逐步尝试 LoRA。示意命令：

```bash
pip install -e '.[vision]'
drivescene infer --manifest data/processed/val_manifest.jsonl --limit 5

drivescene eval \
  --manifest data/processed/val_manifest.jsonl \
  --predictions outputs/predictions.jsonl
```

这里的 `infer` 默认输出 `outputs/predictions.jsonl`，`eval` 只是本地可解释的指标。模型、CUDA 和包版本需要在 GPU 机器上实际验证。

## GitHub 发布

目前提供两种方法：

**方法 A（推荐）**：安装 GitHub CLI（`gh`），`gh auth login` 后运行：

```bash
bash scripts/publish_with_gh.sh
```

脚本默认创建**私有** `drive-scene-reasoner` 仓库、初始化 Git、提交并推送。建议审核版权、数据文件和训练脚本以后再公开。

**方法 B**：打开 https://github.com/new ，创建空的 `drive-scene-reasoner` 仓库（先不勾选自动生成 README），然后：

```bash
git init -b main
git add .
git commit -m 'chore: initialize driving scene reasoning project'
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/drive-scene-reasoner.git
git push -u origin main
```

新仓库的许可证由你决定；这个起步工程没有复制上游源代码或数据。原始 DriveLM **语言数据 CC BY-NC-SA 4.0**，不要直接公开训练数据，具体细节见 [docs/UPSTREAM.md](docs/UPSTREAM.md)。

## 建议的第一期任务

1. 真实数据路径与图像一致性检查；生成100条六视角验证样本。
2. Qwen3-VL-8B 基线的多任务推理，保存每条输出。
3. 先执行前视角与六视角对照，记录显存、耗时、QA 得分。
4. 使用 LoRA 领域微调，比较微调前后各子任务的变化。
5. 第二阶段再加入 Graph VQA 上游依赖、视觉证据和多摄像头动态选择。
