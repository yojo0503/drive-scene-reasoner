# DriveLM 对齐边界

参考上游：OpenDriveLab/DriveLM，固定提交 1de72a74b257e5373400fa68239e99bd5d20580a。

## 数据与结果

保留原始四类 QA，以及 C、con_up、con_down、cluster、layer、tag 等原始字段和未知扩展字段。对象框来自 key_object_infos，按原始像素坐标保存，不声称是 Qwen 生成或验证的定位结果。完整原始字段留在旁路文件，不加入模型输入。

官方 challenge/evaluation.py 使用 scene_id + '_' + frame_id + '_' + 帧内题号。帧内题号按 perception、prediction、planning、behavior 拼接顺序生成。本项目按源文件生成 official_id，因此筛选任务、限量或删掉无答案记录不会改变其余记录的官方 ID。

export-official 生成与原 output.json 一致的列表：每条包含 id 和 answer。它不自动生成含队伍信息的 submission.json，也不向外部提交结果。若只筛选了部分样本，导出只保证该 manifest 内完整；官方提交仍须覆盖完整官方问题集。

实际完整 v1.1 数据还存在 1 条无答案训练记录及 1 条空问题的验证记录。空答案归一化为 None，但原始 QA 字典不变；空问题保留官方 ID 并显式标记。不会静默改变后续帧内题号。

官方两个图像包的目录也不同：训练包解压为 nuscenes/samples/CAM_*，验证包解压为 val_data/CAM_*。准备训练集时传入 images/nuscenes，准备官方验证问题时传入 images/val_data。应明确指定对应图片根目录，避免自动混用两个集合。

## 评测

本地 eval 是检查完整性与基本问答结果的工具。它不代表官方综合分数。

eval-tag0 复用 tag=0 题目上的严格字符串比较规则；不会自动从自然语言猜 tag。此命令不应用官方 set_graph/eval_graph 的前序对象匹配筛选，因此明确命名为 without_graph_gating。它不能替代完整官方 Accuracy。

完整官方流程还包含：首题建立对象坐标匹配结果、后续问题图筛选、语言指标、ChatGPT 判断、匹配分数和加权综合。项目保留原始输入和官方 ID，使结果可以交给固定版本的原评测程序；没有将本地词语重合分数冒充这些指标。

上游源码保存在服务器 third_party/DriveLM（Git 忽略目录）。完整评测的旧依赖和 API key 配置需单独准备，本次未运行，也没有付费 API 调用。

## 模型与图结构

本项目继续使用 Qwen3-VL，不更换成官方早期 LLaMA-Adapter 基线。保存原始问题依赖是框架准备工作，不表示独立回答已变成图推理。后续实现应区分标注上游答案与模型实际预测，避免在测试时引入真实答案。

本次核对的 v1.1 完整训练文件中，con_up、con_down、cluster、layer 均无非空值，tag 也未赋值。项目保留这些原始字段，但不能因此声称已有可直接监督的依赖图或官方题型标签。上游 challenge 另有问题抽取、选择题转换和 tag 分配步骤；完整训练 QA 与该 challenge 子集不能混为同一评测协议。

上游文件未将语言数据授权给本项目再分发；下载的数据、图像、权重和原始示例都不会提交进本项目 Git。
