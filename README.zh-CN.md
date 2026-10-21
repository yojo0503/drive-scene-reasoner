# Drive Scene Reasoner 中文说明

当前版本为 **0.2**。本次补齐了五项工程问题，并验证 CPU 数据流程；没有运行真实模型加载、推理、训练或付费评测。

## 已完成的更新

1. **六视角严格检查**：选择 all 时必须有六路摄像头；默认解码每张图片，记录尺寸，损坏图片立即报错。
2. **完整评测**：重复 ID、未知 ID、缺失预测默认报错。只有明确选择 missing-policy wrong 时才把漏答计错，报告同时显示覆盖率。
3. **无答案数据保留**：purpose infer 保留所有问题，不划分训练/验证，不生成训练文件。purpose sft 专用于有答案数据。
4. **保留 DriveLM 原始结构**：原始问答字段、问题依赖、分组、层级、对象框和场景信息都会保留。每个样本同时有内部 ID 和官方 ID；筛选样本不会重新编号。frame_graphs.jsonl 保留被选中帧的完整原始问题。
5. **补齐微调后使用流程**：推理接口支持“基座模型 + LoRA 参数”；训练与推理使用同一用户提示词；训练配置包含独立验证集、图像预算、随机种子、版本快照和断点续训。

保留图结构信息不等于已经实现分阶段 Graph VQA 推理。本次没有根据标注猜测依赖，也没有把标注答案、对象说明或场景描述偷偷加入模型输入。

## 在服务器使用

```bash
cd /home/yyh/Github/drive-scene-reasoner
source .venv/bin/activate
pytest -q
```

现有 .venv 用于 CPU 数据与工程检查。GPU 依赖请将来在独立 Python 3.11 环境中按 requirements/gpu-cu128.txt 安装，不要直接套用现有 CPU 环境。

服务器实际资源位置与检查结果见 [SERVER_SETUP.md](docs/SERVER_SETUP.md)。

## 准备有答案数据

```bash
drivescene prepare --input /home/yyh/datasets/DriveLM/v1.1/v1_1_train_nus.json \
  --image-root /home/yyh/datasets/DriveLM/v1.1/images/nuscenes \
  --output data/processed --purpose sft --cameras all
```

输出训练和验证的 manifest、Swift 训练文件、原始问题结构文件、preparation.json。按原始场景划分，记录文件哈希和划分结果。

## 准备官方无答案问题

```bash
drivescene prepare --input /home/yyh/datasets/DriveLM/v1.1/v1_1_val_nus_q_only.json \
  --image-root /home/yyh/datasets/DriveLM/v1.1/images/val_data \
  --output data/challenge --purpose infer --cameras all
```

这里只处理数据，不会执行推理。输出 infer_manifest.jsonl；不能用这个无答案文件计算本地准确率。

官方 v1.1 无答案文件包含 1 条空问题。项目保留其原始编号并标记 question_valid=false；训练准备会排除空问题。将来批量推理遇到这条记录时会保存空预测及异常状态，不会改写问题或为它调用模型生成。

## 评测与官方结果导出

```bash
drivescene eval --manifest data/processed/val_manifest.jsonl \
  --predictions outputs/predictions.jsonl
drivescene export-official --manifest data/challenge/infer_manifest.jsonl \
  --predictions outputs/challenge_predictions.jsonl --output outputs/official_output.json
```

本地指标仍是答案完全匹配率和词语重合得分。eval-tag0 只计算已有 tag=0 标记题目的严格字符串匹配，不包含官方图依赖筛选和综合分数。完整差异见 [DriveLM 对齐说明](docs/DRIVELM_COMPATIBILITY.md)。

## 训练计划检查

```bash
bash scripts/train_lora.sh --dry-run
```

默认只检查并打印命令，不加载模型。实际配置在 configs/lora_qwen3vl8b.json，支持命令行覆盖。将来执行训练时还需显式传入 --execute、进入 tmux，并指定空闲 GPU。当前没有证明这套 8B 六视角配置能在两张 3090 上稳定训练；包版本也尚未完成 GPU 环境联合验证。

项目沿用 Qwen3-VL，并按 DriveLM 的数据与结果接口组织。它没有完整复制官方旧版 LLaMA-Adapter 工程，也没有宣称达到原模型效果。
