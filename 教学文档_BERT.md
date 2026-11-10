 # BERT 教学文档：模型结构 · 输入输出 · 训练与微调

> 上一篇 `教学文档_Transformer.md` 讲的是 decoder-only（你的 DeepSeek-R1-Distill-Qwen-1.5B）。这篇讲它的「镜像」——**BERT（encoder-only）**。
> 核心一句话：**BERT 砍掉解码器、去掉因果掩码，让每个词都能看前后文，再靠「完形填空」做自监督预训练，学出强大的双向表示。**
> 对照模型：`bert-base-uncased`（12 层、768 维、110M 参数，440MB，你的机器随便跑）。

---

## 〇、一页看懂全局

BERT 从一段文字到「能用于下游任务」，完整数据流是：

```
输入文本（一句，或两句 [A][SEP][B]）
  │
  ▼
① 分词器（WordPiece）──→ token id + 特殊标记 [CLS] [SEP] [MASK] [PAD]
  │
  ▼
② 三个 Embedding 相加 ──→ 每个 token 变成一个 768 维向量
     · token embedding   （词本身）
     · segment embedding （属于 A 句还是 B 句）
     · position embedding（绝对位置，可学习，加在输入处）
  │
  ▼
③ 重复 12 层 Encoder Block，每层做两件事：
     · 多头自注意力（双向，无因果掩码，每个词能看到所有词）
     · 前馈 MLP
  │
  ▼
④ 输出：每个 token 一个 768 维隐藏向量
     · [CLS] 位置的向量 = 整句的表示（用于分类）
     · 其它位置向量      = 该 token 的上下文表示（用于标注/问答）
  │
  ▼
⑤ 下游任务：接一个「任务头」（task-specific head）做分类/标注/问答
```

---

## 一、BERT 的定位：它和你的 decoder-only 模型正好相反

在 `教学文档_发展脉络学习路线.md` 里，BERT 属于 **Encoder 系**。把上一篇的知识翻过来对比，就是理解 BERT 的钥匙：

| 维度 | 你的 DeepSeek-Qwen-1.5B（decoder-only） | BERT（encoder-only） |
|---|---|---|
| 用 Transformer 哪部分 | 只用 Decoder | 只用 **Encoder** |
| 注意力方向 | 单向（因果掩码，只能看左边）| **双向（无掩码，前后都能看）** |
| 预训练目标 | 预测下一个 token（自回归生成）| **完形填空 MLM + 下一句判断 NSP** |
| 能不能生成文本 | ✅ 逐词生成 | ❌ 不能自回归生成（只能「填空」）|
| 擅长 | 生成、对话 | **理解、分类、检索、标注** |
| 位置编码 | RoPE（注意力内旋转）| **绝对位置 embedding（输入处相加，可学习）** |
| 归一化 | Pre-Norm + RMSNorm | 原始版 Post-Norm + LayerNorm |
| 模型类型 | 生成式 | **判别式** |

**记死这一条**：decoder-only 是「从左往右讲故事」，BERT 是「读完一整句再理解」。后者因为能同时看到左右上下文，对**理解类任务**天然更强，代价是**放弃生成能力**。

---

## 二、模型结构：Encoder Block 长什么样

BERT 由 12 个**完全相同的 Encoder Block** 堆叠而成。每个 Block 内部：

```
        ┌────────── 输入 x (768维) ───────────┐
        │                                      │
        ▼                                      │
  ① 多头自注意力（双向，无掩码）                  │
        │                                      │
        ├── + 残差连接 ──────────────────────────┤
        │                                      │
        ▼                                      │
  ② LayerNorm                                │
        │                                      │
        ▼                                      │
  ③ 前馈 MLP（768 → 3072 → 768，GELU 激活）      │
        │                                      │
        └── + 残差连接 ──────────────────────────┘
        │
        ▼
  ④ LayerNorm
        │
        ▼
   输出到下一层（形状不变，仍 768 维）
```

和你的模型对照三个**关键差异**：

1. **没有因果掩码**：BERT 的 attention 是 `(Q·Kᵀ/√d) · V`，但 softmax 前**不把未来位置置 -∞**。所以第 t 个 token 的注意力权重会同时分布在它前面**和后面**的词上。这就是「双向」的全部含义。
2. **位置编码在输入处相加，是「可学习的绝对位置」**：一个 `nn.Embedding(512, 768)`，第 0 个位置是第 0 行的向量，直接加到 token embedding 上。和你的 RoPE（在注意力内部旋转 Q/K、编码相对位置）是两条完全不同的路线。
3. **Post-Norm（LayerNorm 在残差之后）+ 普通 LayerNorm + GELU**：原始 BERT 用的是老式布局。现代 LLM 改成了 Pre-Norm + RMSNorm + SiLU/SwiGLU（你上一篇学过），这些是后人「改良」的点，读论文时心里有数即可。

> 补充：BERT 多头注意力的 head 划分和你的模型一样——768 = 12 头 × 64 维，只是**没有 GQA**（原始 BERT 每头都有独立 K/V）。

---

## 三、输入：三个 Embedding 相加 + 特殊标记

这是 BERT 最「花哨」、也最需要单独讲的一节，因为它和你的模型差别最大。

### 1. 三个 Embedding 逐元素相加

BERT 的输入向量 = **token + segment + position 三者相加**（都是 768 维）：

```
输入向量 E = E_token(tok) + E_segment(seg) + E_position(pos)
```

| Embedding | 作用 | 具体形状 |
|---|---|---|
| Token Embedding | 词本身 | `nn.Embedding(30522, 768)`，WordPiece 子词表 |
| Segment Embedding | 区分 A 句 / B 句 | `nn.Embedding(2, 768)`，只有 0 和 1 两种 |
| Position Embedding | 绝对位置 | `nn.Embedding(512, 768)`，位置 0~511，可学习 |

### 2. 四个特殊 token（BERT 的灵魂）

| 特殊 token | id | 含义 |
|---|---|---|
| `[CLS]` | 101 | 每句开头必加。它最后一层的向量 = **整句表示**，喂给分类器 |
| `[SEP]` | 102 | 分隔符。两句之间、句尾加 |
| `[MASK]` | 103 | 完形填空时，被遮住的词 |
| `[PAD]` | 0 | 补齐到固定长度 |

### 3. 一个具体例子（必看）

输入两个句子：`"I love Paris"` 和 `"It is the capital"`：

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
tok("I love Paris", "It is the capital", return_tensors="pt")
```

tokenizer 会把它编码成（这是理解 BERT 输入的最直观方式）：

```
tokens:         [CLS] I   love Paris [SEP] It   is   the  capital [SEP]
input_ids:      [101] 1045 2293 3000  [102] 2009 2003 1996 3007    [102]
token_type_ids: [ 0 ]  0    0    0    [ 0 ]  1    1    1    1      [ 1 ]
attention_mask: [ 1 ]  1    1    1    [ 1 ]  1    1    1    1      [ 1 ]
```

逐行看懂：

- **`input_ids`**：每个 token 的词表编号。`[CLS]`=101 在开头，`[SEP]`=102 分隔两句、结尾再一个。
- **`token_type_ids`（= segment embedding 的输入）**：第一句全 0，第二句全 1。这就是「segment embedding 只有 0/1 两种」的直接体现——模型靠它知道「这个词属于 A 句还是 B 句」。
- **`attention_mask`**：1 表示有效 token，0 表示 `[PAD]`（这里没 pad 所以全 1）。

然后这三样分别查三个 embedding 表，**逐元素相加**，得到一个 `(11, 768)` 的矩阵，进入 12 层 Encoder。

> 记一句话：**`token_type_ids` 是 BERT 独有的**，你的 decoder-only 模型没有——因为它不需要「区分句子」这个能力。

---

## 四、输出：每个 token 一个向量，[CLS] 是整句代表

12 层跑完后，输出形状和输入一样：`(batch, seq_len, 768)`。**每个位置都有一个 768 维向量**，但用法不同：

1. **`[CLS]` 位置的向量** → 经过一个 `pooler`（Linear + tanh）→ 代表「整句话/整段输入的语义」。**句子级任务**（分类、相似度、NSP）都用它。
2. **其它每个 token 的向量** → 代表「这个词在**完整上下文**里的含义」。**词级任务**（命名实体识别 NER、问答定位答案起止位置）用它。

```
输出矩阵 (seq_len, 768)
  [CLS] 行 ──→ pooler ──→ 768维句向量 ──→ 分类头 ──→ 类别
  token1 行 ──→ 直接接 softmax ──→ 该 token 的标签（NER）
  token2 行 ──→ 同上
  ...
```

**关键点**：BERT 的输出**没有 LM head 预测下一个词**（它不做自回归生成）。它输出的是一堆「表示向量」，具体拿来做什么，由下游任务头决定——这就是「预训练 + 微调」范式。

---

## 五、训练（预训练）：两个自监督任务

BERT 之所以牛，是因为它在**海量无标签文本**上，用两个**自监督**任务预训练（回顾：自监督 = 数据自己给自己出标签）。

### 任务 1：Masked Language Model（MLM，完形填空）

**目标**：随机遮住 15% 的 token，让模型预测被遮住的词。

```
原文:  my dog is hairy
遮罩:  my dog is [MASK]     → 模型要预测 [MASK] = hairy
```

**为什么要「随机替换」而不是全用 [MASK]**：因为微调时输入里不会有 `[MASK]`，会造成训练/推理不一致。所以对那 15% 被选中的 token：

- **80%** → 换成 `[MASK]`（正常考完形填空）
- **10%** → 换成**随机词**（防止模型「看到 [MASK] 就瞎猜」）
- **10%** → **原样不动**（逼模型对每个位置都保持警惕）

损失函数：**只对被遮住的那 15% 位置算交叉熵**，其余位置忽略（类比你上一篇学的 `-100` 掩码思想）。

### 任务 2：Next Sentence Prediction（NSP，判断下一句）

**目标**：给两句 A、B，判断 B 是不是 A 的下一句（二分类，用 `[CLS]` 向量）。

- 50% 时间 B 是真下一句 → 标签 `IsNext`
- 50% 时间 B 是随机句子 → 标签 `NotNext`

设计动机：让模型理解**句子之间的关系**（问答、推理需要）。但后来 RoBERTa 证明这个任务**帮助不大**（动态掩码 + 更大数据 + 更长训练更重要），所以现代模型大多**只保留 MLM**。

### 预训练超参（感受一下规模）

- 数据：BooksCorpus（8 亿词）+ 英文维基（25 亿词）
- BERT-base：12 层 / 768 维 / 12 头 / 110M 参数，4 张 TPU 训 4 天
- BERT-large：24 层 / 1024 维 / 16 头 / 340M 参数
- 优化器 Adam，学习率先 warmup 再线性衰减，dropout 0.1，GELU 激活

---

## 六、微调（Fine-tuning）：换一个「任务头」即可

预训练完的 BERT 是个「通用语言理解器」。**微调 = 在它上面接一个小的任务头，用少量有标签数据，整个模型（含 BERT 本体）一起小幅更新。**

这是「**生成式/自监督预训练 → 判别式微调**」范式的完美体现（前面讲判别式 vs 生成式时的关键联系）。

### 四类典型下游任务

| 任务类型 | 输入 | 用什么输出 | 任务头 | 例子 |
|---|---|---|---|---|
| 单句分类 | 一句话 | `[CLS]` 向量 | Linear(768→类别数) | 情感分析 SST-2 |
| 句对分类 | 两句 `[SEP]` 分隔 | `[CLS]` 向量 | Linear(768→类别数) | 自然语言推理 MNLI |
| 词级标注 | 一句话 | **每个 token** 向量 | 每个 token 接 softmax | 命名实体识别 NER |
| 问答 | `[CLS]问题[SEP]文章[SEP]` | **每个 token** 向量 | 两个头分别预测答案**起/止**位置 | SQuAD |

### 动手：微调一个分类器（你的 8GB 机器秒跑）

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# model 内部结构：12层 BERT encoder → pooler → classifier(Linear 768→2)
# 预训练的 110M 参数 + 一个新增的分类头，全部一起微调
```

**两个必须理解的微调要点**：

1. **微调时预训练权重是「起点」而非「固定」**：BERT 全部 110M 参数都会继续更新（只是用很小的学习率，如 2e-5~5e-5）。这和你做 LoRA「冻结主干、只训小矩阵」是**两种思路**——LoRA 是后人为省显存发明的替代方案。
2. **只需在目标数据上训几个 epoch**：预训练已经给了强大的表示，微调是「小步微调」，数据少、收敛快。

---

## 七、和你的 LoRA 知识衔接

你之前用 LoRA 微调 decoder-only 模型。BERT 也能 LoRA 微调，挂载点一模一样：

```
BERT Encoder Block
  ├─ 注意力：query/key/value（注意：BERT 里这些叫 self.query/self.key/self.value）
  └─ 中间层：dense（MLP，768→3072→768）
```

- 挂 `query` / `value`（BERT 的 attention 线性层）就能省参数微调。
- 分类任务用 `AutoModelForSequenceClassification` + PEFT 的 `TaskType.SEQ_CLS`（注意：**不是** `CAUSAL_LM`，因为 BERT 不做生成）。
- 完形填空演示用 `AutoModelForMaskedLM` + `TaskType.MASKED_LM`。

---

## 八、一句话串起全文

> **BERT 只用 Encoder、去掉因果掩码**，让每个词能看**前后文**；输入是 `[CLS]` 开头、`[SEP]` 分隔的一句或两句，由 **token + segment + position 三个 embedding 相加**得到；它靠 **MLM 完形填空 + NSP 判断下一句**两个自监督任务在海量文本上预训练；输出是每个 token 的 768 维向量，其中 **`[CLS]` 向量代表整句**；微调时**接一个任务头**，用少量标签数据把整模型小幅更新，就能做分类、标注、问答。它放弃了生成，换来了最强的「理解」能力。

---

## 附 A：动手实验清单（8GB 无障碍）

1. **观察输入**：`tokenizer("I love Paris", "It is the capital", return_tensors="pt")`，逐字段看 `input_ids` / `token_type_ids` / `attention_mask`。
2. **完形填空**：加载 `AutoModelForMaskedLM`，输入 `"The capital of France is [MASK]."`，看预测结果是不是 `paris`。
3. **抽取 [CLS] 向量**：加载 `AutoModel`，取 `outputs.last_hidden_state[:, 0, :]`，这就是句向量，可以算两句话的余弦相似度。
4. **微调分类**：用 `AutoModelForSequenceClassification` 在 IMDb 情感数据上训 1~2 epoch，看 loss 下降。

## 附 B：对照你代码/知识里的名词

| 你已经会的（decoder-only） | BERT 对应 | 关键差异 |
|---|---|---|
| `TaskType.CAUSAL_LM` | `TaskType.SEQ_CLS` / `MASKED_LM` | BERT 不做生成 |
| 因果掩码（未来置 -∞） | 无掩码（双向）| BERT 能看前后文 |
| RoPE（注意力内旋转）| 绝对位置 embedding（输入处相加）| 两条路线 |
| `token_type_ids` 不存在 | `token_type_ids` 区分 A/B 句 | BERT 独有 |
| LM head 预测下一个词 | `[CLS]` + pooler 输出句表示 | 输出用途不同 |
| RMSNorm / Pre-Norm | LayerNorm / Post-Norm | 原始 vs 现代改良 |

---

笔记：
