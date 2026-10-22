# 服务器准备记录（2026-10-21）

> 最新进展（2026-10-22）：真实六视角推理、双卡短训练、验证与检查点保存已通过显存适配测试，见 [显存报告](VRAM_FIT.md)。下文保留环境安装/历史部署阶段的验证边界。
本次仅完成代码、下载、文件完整性和 CPU 数据处理；没有真实模型加载、训练、生成或付费评测调用。

## 路径

| 内容 | 服务器位置 |
| --- | --- |
| 项目 | /home/yyh/Github/drive-scene-reasoner |
| 原 DriveLM 参考源码 | 项目内 third_party/DriveLM，固定提交 1de72a74b257e5373400fa68239e99bd5d20580a |
| Qwen3-VL-8B-Instruct | /home/yyh/models/Qwen3-VL-8B-Instruct |
| DriveLM v1.1 原始文件 | /home/yyh/datasets/DriveLM/v1.1 |
| 训练图片根目录 | /home/yyh/datasets/DriveLM/v1.1/images/nuscenes |
| 官方验证图片根目录 | /home/yyh/datasets/DriveLM/v1.1/images/val_data |
| 本地训练/验证数据清单 | 项目内 data/processed |
| 官方无答案问题清单 | 项目内 data/challenge |
| 下载与检查报告 | 项目内 outputs/setup |

## 下载校验

- 模型：16 个文件，共 17,545,915,883 字节；固定版本 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b。四个权重分片按上游 SHA256 校验，配套文件按 Git blob 哈希校验。配置、分片索引与 750 个张量头部信息对应；没有读取张量执行模型。
- 数据：4 个文件，共 4,390,892,186 字节；固定版本 ae973fbd4e8d4684af4ab234d504bd6c5e946868。训练/验证 ZIP 和两份 JSON 均通过上游哈希校验。
- 解压：训练图片 24,432 张，验证图片 4,794 张；ZIP 条目通过 CRC 校验。所有图片均经 CPU 实际解码，尺寸均为 1600×900。
- 每个资源目录保留 asset_inventory.json，包含来源、固定版本、文件大小和 SHA256；图片目录另有 extraction_inventory.json。
- 下载凭据只用于官方 Hugging Face 主机，临时鉴权文件在完成后已移除。资源保留在服务器，不进入项目 Git。

## 实际数据规模

| 集合 | 场景 | 帧 | 问答 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 官方训练文件 | 696 | 4,072 | 377,956 | 377,955 条有答案，另有 1 条空答案 |
| 官方验证问题文件 | 149 | 799 | 15,480 | 全部无公开答案，其中 1 条问题为空 |

本地训练/验证来自有答案训练场景的固定随机划分。官方无答案文件完整保留原始 ID，空问题显式标记。不要把官方无答案问题集当成本地有标签验证集。

使用 seed=42、按场景保留约 10% 作为本地验证，实际输出 339,357 条训练样本和 38,598 条本地验证样本。官方问题集单独输出 15,480 条推理数据清单。

## CPU 验证

- 26 项单元/回归测试通过；源码编译和训练 shell 语法检查通过。
- 官方公开示例的 790 条问答、54 张真实图片通过数据准备检查。
- 全部真实数据的逐帧图像检查报告：outputs/setup/train-data-audit.json、val-data-audit.json。
- 模型文件头部检查报告：outputs/setup/model-file-audit.json。
- 全量数据准备日志：outputs/setup/data-prepare.log。
- 训练计划只使用 --dry-run 检查，不执行训练。

## 后续 GPU 环境

现有 .venv 是 Python 3.13 CPU 工具环境。后续 GPU 工作建议使用独立 Linux Python 3.11 环境，安装 requirements/gpu-cu128.txt 的关键版本。服务器 NVIDIA 驱动为 595.71.05；本次未安装或实际验证完整 GPU 软件组合。

训练入口默认只打印计划。真正执行需要 --execute、tmux、干净代码提交以及明确指定 CUDA_VISIBLE_DEVICES。现有其他实验没有被接管或改动。
