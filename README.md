# Qwen3.5 Receipt OCR

基于 **Qwen3.5-VL** 的票据结构化抽取工具链：从拍照小票/多版式票据图像，输出可校验的结构化 JSON，并覆盖数据治理、SFT、DPO、GRPO 训练、批推理、评估与人工复核 API。

> 独立项目，不依赖任何上游仓库的运行时代码、数据目录或 editable 安装。许可证为 MIT（上游 Apache-2.0 部分见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)）。

## 它解决什么

OCR 本身不是终点。业务真正需要的是「上传一张票据 → 得到可归档的结构化字段 + 商品明细 + 校验结果 + 复核状态」。因此本项目把重心放在三件事上：

1. **严格 schema**：11 个业务字段 + 商品明细，输出必须通过 JSON Schema 校验，不合格即判失败，不用「合法 JSON」冒充正确结果。
2. **可信评估**：全分母生成式指标、一对一商品匹配、分组结果，避免用覆盖率不同的分子分母美化分数。
3. **可靠交付边界**：低置信度进入人工复核队列，不追求「全自动」，而是让错误可控。

## 能力范围

| 模块 | 已实现 | 验证边界 |
|---|---|---|
| 数据 | CORD 固定 revision 下载校验、原图/原 GT、受限字段 manifest、SFT/DPO/GRPO 导出与审计 | 真实 CORD 已处理；FATURA 仅有合成格式回归，全量下载仍受阻 |
| 业务核心 | 11 字段 schema、严格 JSON、保守规范化、一对一商品匹配、适用规则与 composite reward | CPU 回归通过；**无模型准确率声明** |
| 训练 | SFT / DPO / GRPO，显式 reward 接线、阶段与 resume 合同、DoRA 接线、安全合并 | **未安装 GPU 训练环境**，未实测 Qwen3.5 权重加载与合并数值等价性 |
| 推理与服务 | Transformers / OpenAI 兼容客户端、批推理、全分母评估、Flask 上传 API | mock 回归；未启动真实推理后端 |
| 复核闭环 | API 返回复核状态，候选偏好导入要求审定与实际 prompt | 尚无复核 UI、持久化队列、生产身份鉴权 |

## 目录结构

```
configs/            业务配置：receipt/{sft,dpo,grpo,vision_lora}.json、data_sources.json
data/fixtures/      纯合成 CPU 夹具（不含真实票据）
docs/plan.md        完整设计方案、实验口径与实施手册
licenses/           上游 Apache-2.0 许可证原文
scripts/            上游训练 shell 入口（finetune_*.sh、merge_lora.sh、DeepSpeed 配置）
src/model/          Qwen3.5 模型加载与建模（衍生自上游）
src/dataset/        SFT/DPO/GRPO/分类数据集
src/train/          训练入口与 monkey patch
src/trainer/        SFT/DPO/GRPO/分类 trainer
src/receipt/        票据业务层：schema、规范化、匹配、指标、奖励、推理、服务、数据治理
src/serve/          通用演示服务
tests/              CPU 回归测试
tools/receipt/      数据处理、导出、评估与验收脚本
```

## 快速开始

要求 Python 3.11 – 3.13。

```bash
git clone git@github.com:onedayhappen/Qwen3.5-receipt-ocr.git
cd Qwen3.5-receipt-ocr

python -m venv .venv
source .venv/Scripts/activate      # Windows
# source .venv/bin/activate        # Linux / macOS

pip install -e ".[dev]"
pytest tests -q
```

跑通后你会看到 188 个测试通过——全部为 CPU 测试，不加载模型、不需要 GPU。

## 常用命令

安装后可用以下入口（均可加 `--help` 查看参数）：

```bash
receipt-data          # 数据治理：下载校验、manifest、导出
receipt-batch-infer   # 批推理
receipt-evaluate      # 全分母评估
receipt-serve         # 上传与复核 API
receipt-train         # 训练启动器（SFT / DPO / GRPO）
```

数据与验收脚本：

```bash
python tools/receipt/make_fixtures.py        # 生成合成 CPU 夹具
python tools/receipt/prepare_cord.py         # 处理 CORD
python tools/receipt/build_dataset.py        # 构建 manifest
python tools/receipt/export_training_data.py # 导出训练数据
python tools/receipt/verify_project.py --wheel-target <dir>  # 全量验收，产出 reports/acceptance.json
```

## 验证边界（请先读这一段）

本项目**只完成了 CPU 层面的代码与数据管线验收**，以下结论**不成立**，也不应据此引用：

- 没有 GPU 训练结果，没有模型准确率、F1 或业务正确交付率数字
- 未实测 Qwen3.5 权重加载、LoRA 合并的数值等价性
- 未启动真实推理后端，服务层仅有 mock 回归
- FATURA 未完成全量下载，相关处理只有合成格式回归
- 偏好数据为合成错误，不是人工审定的模型生成

这些边界同时记录在 `reports/acceptance.json` 的 `limitations` 字段中。设计与实验口径见 [docs/plan.md](docs/plan.md)。

## 许可证

- 本项目原创部分：**MIT**，见 [LICENSE](LICENSE)
- `src/model/`、`src/dataset/`、`src/train/`、`src/trainer/` 中衍生自 [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) 的文件：仍为 **Apache-2.0**，署名与改动清单见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，许可原文见 [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)

MIT 许可**不覆盖**上述衍生文件。数据许可与代码许可相互独立，见 `configs/data_sources.json`；原始票据图像不随仓库分发。
