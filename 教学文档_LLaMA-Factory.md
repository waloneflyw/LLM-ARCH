# LLaMA-Factory 教学文档：原理、架构与使用场景

> 面向对象：已经用 `transformers + peft` 手写过 LoRA 微调、想要一个**"全流程 + 可视化"**框架的同学。
> 配套环境：Windows 11 + RTX 5060（8GB）+ conda 环境 `lora`。

---

## 一、LLaMA-Factory 是什么

一句话：**LLaMA-Factory（现已更名 LlamaFactory）是一个"大模型微调全家桶"**——把从数据准备、预训练、SFT、RLHF、量化、推理到可视化监控的**整条流水线**打包进一个框架，提供 **Web 界面 / 命令行 / API** 三种用法。

对比你之前的做法：

| 维度 | 你之前的写法 | LLaMA-Factory |
|------|-------------|---------------|
| 写代码 | 自己写预处理、LoRA、Trainer | 改 YAML 配置 / 网页点按钮 |
| 训练方法 | 只有 LoRA/QLoRA | SFT、DPO、PPO、KTO、ORPO、预训练、奖励模型…… |
| 可视化 | 无 | LLaMA Board 网页实时看 loss |
| 灵活性 | 高（自己控制一切） | 中（配置驱动，深度定制仍需写代码） |

关键理解：**你之前手写的那套（加载模型 → LoRA 配置 → Trainer → 保存）在 LLaMA-Factory 里被抽象成"一个配置文件"**。它底层用的仍然是 transformers、peft、trl 这些你熟悉的库，只是帮你在上层做了统一封装。

---

## 二、原理：它把哪些东西"统一"了起来

### 1. 统一了"训练方法"

LLaMA-Factory 覆盖了 LLM 微调的**完整技术栈**，而不是只有 SFT：

- **（增量）预训练** Pre-training：继续预训练，让模型补学领域知识
- **监督微调 SFT**：你正在做的，指令-回应配对（含多模态 SFT）
- **奖励模型 RM 训练**：训练一个打分器，为 RLHF 服务
- **偏好对齐 RLHF**：PPO、DPO、KTO、ORPO、SimPO

### 2. 统一了"高效微调方法"（PEFT）

| 方法 | 说明 |
|------|------|
| **LoRA** | 你已学的低秩适配 |
| **QLoRA** | 4-bit 量化 + LoRA（显存更省） |
| **DoRA** | LoRA 的改进：把增量分解为"方向+幅度" |
| **OFT** | 正交微调，另一类参数高效方法 |

还内置了一批**省显存的优化器**：GaLore、BAdam、APOLLO、Muon。

### 3. 统一了"量化方案"

支持从 16-bit 全参数微调，一路到 **2/3/4/5/6/8-bit QLoRA**，量化后端可选 **AQLM / AWQ / GPTQ / LLM.int8 / HQQ / EETQ**。

### 4. 统一了"推理加速后端"

训练或推理时可以切换底层引擎：**FlashAttention-2、Unsloth、Transformers、vLLM、SGLang**。

> 注意：LLaMA-Factory 的"加速"里，**Unsloth 是它的一个可选项**——你可以理解为"LLaMA-Factory 是全家桶，Unsloth 是它引擎里的一档"。

### 小结

LLaMA-Factory 的原理本质是：**把大模型微调涉及到的"模型、数据、训练方法、量化、加速、监控"六大块，抽象成统一的配置项和模块**，让你通过"选配置"而不是"写代码"来完成一条完整的微调流水线。

---

## 三、架构：模块化设计

LLaMA-Factory 是典型的**模块化/分层架构**，核心子系统如下：

### 1. 配置系统（Configuration）
所有超参数集中在 `yaml` 配置里（`examples/train_lora/...` 下有大量现成模板），框架负责**校验 + 路由**：你填了什么，就调用对应的模块。

```yaml
# 一个 LoRA SFT 配置的精简示例
model_name_or_path: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
stage: sft                # 训练阶段：sft / dpo / ppo / rm / pt
finetuning_type: lora     # 微调类型：lora / qlora / full / dora
lora_rank: 8
lora_alpha: 32
dataset: alpaca_zh        # 数据集名（注册在 data/ 里）
template: qwen            # 对话模板
output_dir: saves/deepseek-lora
```

### 2. 数据系统（Data）
负责**数据集加载、格式化、批处理**。你之前手写的 `preprocess_funcation`，在这里由框架根据 `template`（对话模板）**自动完成**——它知道如何把指令、输入、输出拼成训练所需的 `input_ids` 和 `labels`。

### 3. 模型系统（Model）
负责**模型加载、打补丁（patching）、量化、挂载 adapter**。你之前的 `AutoModelForCausalLM + get_peft_model` 都封装在这一层。

### 4. 训练系统（Train）
按 `stage` 分发到不同的 trainer：SFT、DPO、PPO 各有专门的 trainer 类（底层来自 transformers / trl）。

### 5. 推理系统（Inference）
推理/对话时的后端切换：Transformers、vLLM、SGLang。

> 早期版本把它概括为**三层**：`Model Loader`（模型加载器）+ `Data Worker`（数据工人）+ `Trainer`（训练器），现在则细分为上述五系统。当前代码还并存 **V0（稳定版）** 和 **V1（实验版）** 两套实现，靠 `USE_V1` 环境变量切换。

### 三种使用入口

```bash
# 1. 命令行训练（最常用）
llamafactory-cli train examples/train_lora/qwen_lora_sft.yaml

# 2. 网页界面（可视化，零代码）
llamafactory-cli webui

# 3. 其他
llamafactory-cli chat      # 交互对话
llamafactory-cli export    # 合并/导出模型
llamafactory-cli api       # OpenAI 风格 API 服务
```

**LLaMA Board** 是基于 Gradio 的 Web UI：可视化调超参数、实时画 loss 曲线、训练监控、交互式对话评测，是它区别于纯代码库的最大卖点。

---

## 四、使用场景：什么时候用 LLaMA-Factory

### ✅ 强烈推荐用

1. **想要零代码 / 低代码微调**：不写 Python，网页点点就能训练、能看 loss、能对话验证。
2. **要跑"完整流水线"**：从 SFT → RM → PPO/DPO 一条龙，或做预训练、多模态 SFT。
3. **批量 / 工程化实验**：一个 YAML 对应一次实验，方便复现、对比、记录，配合 Wandb/TensorBoard/MLflow/SwanLab。
4. **想对比多种方法**：LoRA vs DoRA vs QLoRA、不同量化后端，改一个字段就能切换。

### ⚠️ 需要斟酌的场景

1. **深度自定义**：如果数据处理逻辑很特殊、训练目标很定制，配置驱动的框架反而受限，这时手写（你现在的 `train_lora.py`）更灵活。
2. **Windows 安装**：LLaMA-Factory 依赖较多（尤其 bitsandbytes、部分量化后端在 Windows 上需要特殊处理），安装可能比直接 `pip install transformers peft` 折腾。
3. **只是想理解原理**：框架把细节都藏起来了，学习阶段"手写一遍"（你刚做的）比"直接用框架"更能吃透原理。

### 支持的模型（100+ 家族）

LLaMA、LLaVA、**Qwen**（含 Qwen3.5/3.6）、Mistral、Mixtral-MoE、Gemma、Baichuan、ChatGLM、Phi、Yi、**DeepSeek**、InternLM 等。你用的 DeepSeek-R1-Distill-Qwen-1.5B 属于受支持范围。

---

## 五、和你现有流程的对应关系

| 你之前的代码 | LLaMA-Factory 里 |
|--------------|------------------|
| `AutoModelForCausalLM.from_pretrained(...)` | 配置项 `model_name_or_path`（模型系统自动加载） |
| 手写 `preprocess_funcation` | 配置项 `dataset` + `template`（数据系统自动拼） |
| `LoraConfig(r=8, alpha=32, target_modules=[...])` | 配置项 `finetuning_type: lora` + `lora_rank/lora_alpha` |
| `TrainingArguments(...)` | 配置文件里的训练参数区 |
| `Trainer(...)` | `llamafactory-cli train`（训练系统） |
| `merge_and_unload()` + `save_pretrained` | `llamafactory-cli export` |

你学到的**分词器原理、LoRA 原理、`r/alpha`、标签 `-100` 掩码、merge 合并**这些底层知识，恰恰是**看懂 LLaMA-Factory 那些 YAML 配置项到底在控制什么**的基础——有了手写经验，再上框架就不会"只会点按钮不懂原理"。

---

## 参考来源

- [LLaMA-Factory 官方文档 (LlamaFactory)](https://llamafactory.readthedocs.io/en/latest/index.html)
- [hiyouga/LlamaFactory | DeepWiki — Overview](https://deepwiki.com/hiyouga/LlamaFactory/1-llamafactory-overview)
- [LLaMA-Factory | AI Native Landscape](https://landscape.jimmysong.io/projects/llama-factory/)
- [LLaMA-Factory | AI Wiki](https://aiwiki.ai/wiki/llama_factory)
- [APPO/LLaMA-Factory README](https://github.com/xuc865/APPO/blob/main/LLaMA-Factory/README.md)
