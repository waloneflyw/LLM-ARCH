# LLM-ARCH · 大模型微调学习仓库

基于 **RTX 5060（8GB 显存）** 的 LLM 架构与微调学习项目：从 Transformer 原理出发，用 **LoRA 微调 DeepSeek-R1-Distill-Qwen-1.5B**，并沉淀了 Transformer / BERT / CLIP / Unsloth / LLaMA-Factory 等系列教学文档，以及一个**从零实现的迷你 GPT**。

> 主线：理解 Transformer 结构 → 手写 LoRA 微调 → 用 Unsloth / LLaMA-Factory 加速与工程化 → 从零复现迷你 GPT 打通全流程。

---

## 环境要求

| 项 | 要求 |
|---|---|
| 系统 | Windows 11 |
| 显卡 | RTX 5060（Blackwell，`sm_120`，8GB 显存） |
| Python | 3.11，conda 环境 `lora` |
| PyTorch | 需 `cu128+`（支持 Blackwell `sm_120`） |
| 主要依赖 | `transformers` `peft` `datasets` `accelerate` `bitsandbytes` |

环境自检（会实际在 GPU 上跑一次矩阵乘法，验证 `sm_120` 内核可用）：

```bash
python verify_env.py
```

---

## 目录结构

```
LLM-ARCH/
├── train_lora.py               # LoRA 微调脚本（含详细注释）
├── llm.ipynb                   # LoRA 微调 Notebook（与脚本等价，含运行输出）
├── test_inference.py           # 加载微调后模型做推理测试
├── verify_env.py               # 环境自检
├── 教学文档_*.md               # 系列教学文档（见下）
├── transformer_scratch/        # 从零实现的迷你 GPT（不依赖 transformers）
├── lora_model/                 # 微调出的 LoRA 权重（不提交，见「模型权重说明」）
├── merged_model/               # 合并后的完整模型（不提交）
└── results/                    # 训练 checkpoint（不提交）
```

---

## 快速开始

### 1. 训练 LoRA

```bash
python train_lora.py
```

脚本会：加载基础模型 → 预处理 `MBZUAI/LaMini-instruction` 数据集 → 训练 → 把 LoRA 权重保存到 `./lora_model`。

核心配置：

| 项 | 值 |
|---|---|
| 基础模型 | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| 数据集 | `MBZUAI/LaMini-instruction`（默认取 200 条跑通流程） |
| LoRA | `r=8`, `alpha=32`, `dropout=0.1`，目标模块 `q/k/v/o_proj` |
| 训练 | 3 epochs，`lr=2e-4`，batch=2 × 梯度累积 4（等效 8），`fp16` |
| 数据 | `max_length=512`，指令部分与 padding 的 label 置 `-100` |

> 8GB 显存友好：`fp16` + `batch_size=2` + 梯度累积，跑 1.5B 模型无压力。

### 2. 推理测试

```bash
python test_inference.py
```

会加载基础模型 + LoRA adapter，对几个问题做自回归生成。

### 3. 合并导出完整模型

参照 `llm.ipynb` 最后一个 cell：用 `PeftModel.merge_and_unload()` 把 LoRA 增量合并回基础权重，导出为普通完整模型。

---

## 教学文档

按推荐阅读顺序：

| 文档 | 内容 |
|---|---|
| `教学文档_Transformer.md` | decoder-only Transformer 结构全解（对照 Qwen2 / DeepSeek-1.5B） |
| `教学文档_BERT.md` | encoder-only BERT：双向表示与预训练任务 |
| `教学文档_CLIP.md` | 双塔 + 对比学习的图文对齐 |
| `教学文档_Unsloth.md` | Unsloth 如何把 LoRA/QLoRA 微调提速 2–5 倍、省 60–80% 显存 |
| `教学文档_LLaMA-Factory.md` | 全流程可视化微调框架 |
| `教学文档_发展脉络学习路线.md` | 大模型发展脉络与学习路线图 |

---

## 从零实现 Transformer

[`transformer_scratch/`](transformer_scratch/) 是一个**只用 PyTorch、不依赖 transformers** 的迷你 GPT（0.81M 参数、字符级、4 层），几分钟即可在本地跑完整个训练，直观看到 loss 下降与过拟合现象。

```bash
cd transformer_scratch
python train.py
```

它把注意力、多头、因果掩码、残差、MLP、自回归生成等概念全部落成「能跑、能改」的代码——吃透它就真正理解了 Transformer，而不只是「会用 transformers 库」。详见 [transformer_scratch/README.md](transformer_scratch/README.md)。

---

## 模型权重说明

仓库**不提交**任何模型权重（`lora_model/`、`merged_model/`、`results/` 已加入 `.gitignore`）：

- `merged_model/` 约 3.4GB，超过 GitHub 单文件限制；
- `lora_model/` 与 `results/` 是训练产物，运行 `train_lora.py` 即可重新生成。

基础模型由 Hugging Face 托管，运行时按需自动下载，无需手动管理。
