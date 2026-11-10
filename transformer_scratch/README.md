# 从零实现 Transformer（迷你 GPT）

这是一个**只用 PyTorch、不依赖 transformers** 的 Transformer 完整实现，用于把你前面学的所有概念（注意力、多头、因果掩码、位置编码、残差、MLP、LM head、自回归）变成**能跑、能改、能看 loss 下降**的代码。

对照你之前的真实模型：

| | 真实模型 DeepSeek-1.5B | 这个迷你 GPT |
|---|---|---|
| 隐藏维度 d_model | 1536 | 128 |
| 层数 | 28 | 4 |
| 注意力头 | 12（GQA，2 个 KV 头） | 4（标准 MHA） |
| 位置编码 | RoPE（相对） | 可学习位置嵌入（绝对） |
| 词表 | 15.2 万（BPE 子词） | 33（字符级） |
| 结构 | decoder-only | decoder-only（**完全一样**） |

**关键：两者的"骨架"是同一套**。只是这个迷你版把每个维度都缩到最小，让你几分钟内在自己的显卡上跑完整个训练。

---

## 文件结构

```
transformer_scratch/
├── model.py            # 模型：注意力 / MLP / Block / GPT
├── train.py            # 训练：数据 / batch / 训练循环 / 评估 / 生成
├── data/input.txt      # 训练语料（一篇小故事）
└── README.md           # 本文件
```

---

## 怎么跑

在 `lora` 环境里，进入本目录后：

```bash
python train.py
```

如果显卡可用会自动用 GPU，否则退回 CPU（这个迷你模型 CPU 也能跑）。

你会看到类似输出（这是真实运行结果）：

```
词表大小 vocab_size = 33，文本长度 = 2637 字符
设备 = cuda
模型参数量 = 0.81M
step     0 | train loss 3.2451 | val loss 3.2657
step   200 | train loss 1.3147 | val loss 1.9437
step   400 | train loss 0.4145 | val loss 1.8051
...
step  2999 | train loss 0.0917 | val loss 2.4106
===== 模型生成（字符级小模型，效果有限属正常）=====
The little robot liked the warm sun. The little robot liked the quiet street...
```

**train loss 从一开始的 ~3.2 一路降到 ~0.09**，生成文本越来越像原文——这就是"模型在学"的直接证据。

> ⚠️ **注意观察一个现象：train loss 一直降，但 val loss 从 1.9 一路涨到 2.4。** 这叫**过拟合（overfitting）**——语料只有 2637 个字符，模型（81 万参数）很快把训练集"背"下来了（所以 train loss 极低、生成几乎复述原文），但在没见过的验证集上反而变差。这正是"为什么真实训练需要海量数据"的直观答案。

---

## 代码结构讲解（对应训练全流程）

### `model.py` —— 模型长什么样

四个类，从内到外一层层包起来：

```
GPT (整个模型)
 └─ blocks: 4 个 TransformerBlock 叠起来
     └─ 每个 Block 里：
         ├─ MultiHeadSelfAttention（注意力）
         │    ├─ q_proj / k_proj / v_proj → 切多头
         │    ├─ Q·Kᵀ / √d → 因果掩码 → softmax → 加权求和 V
         │    └─ o_proj 合并
         └─ FeedForward（MLP：128 → 512 → 128，GELU 激活）
```

- **`MultiHeadSelfAttention`**：你之前问的 q/k/v/o_proj 全部在这里。`causal_mask` 就是那个"只能看前面"的下三角矩阵。
- **`FeedForward`**：先扩张（128→512）再收缩（512→128），参数大头。
- **`TransformerBlock`**：`x = x + 子层(x)` 这两行就是**残差连接**（防止浅层信息丢失，你问过的那题）。
- **`GPT.forward`**：token 嵌入 + 位置嵌入相加 → N 层 block → LM head。
- **`GPT.generate`**：自回归——只取最后一个位置的输出预测下一个 token，拼回再循环（你问过的"上一个输出怎么变成下一个输入"）。

### `train.py` —— 训练过程长什么样

编号 `0~7` 七个步骤，就是一个标准训练流程：

1. **超参数**：全部集中在文件顶部，改一个就能看到效果（试试把 `n_layers` 从 4 改成 2 或 6）。
2. **字符级分词**：`stoi`/`itos` 两张表做"字符↔整数"映射，就是简化版的分词器。
3. **构造 batch**：注意 `y = data[i+1 : i+block_size+1]`——**这就是"右移一位"**，x 是"前面"，y 是"下一个字符"（你之前问的自回归/右移）。
4. **初始化模型**：`GPT(...)` + `AdamW`。
5. **训练循环**：`前向 → 算 loss → 反向 → 更新`，四行代码。
6. **评估**：`estimate_loss` 用 `@torch.no_grad()` 关掉梯度，`model.eval()` 关掉 dropout，分别在训练/验证集上取平均。
7. **生成**：`model.generate` 从起始 token 开始自回归生成 300 个字符。

---

## 建议你动手改的点（加深理解）

1. **把 `n_layers` 改成 1**：看模型还能不能学到东西（对比 4 层的 loss）。
2. **把 `block_size` 改成 16**：上下文变短，看生成质量怎么变。
3. **把 `temperature` 调大（如 1.5）**：生成更"随机"；调小（0.5）更"保守"。
4. **去掉 `y = data[i+1...]` 的右移**（改成 y=x）：loss 会变得极低但毫无意义——因为模型学会了"照抄"，这就是"右移"存在的意义。
5. **把 `self.causal_mask` 那行注释掉**：模型变成"双向"（能偷看未来），训练 loss 会低到不真实，但生成会崩——感受因果掩码的作用。

---

## 从迷你模型回到真实模型

这个迷你 GPT 和你的 DeepSeek-1.5B 之间，差距只在**规模和几个工程细节**：

- **规模**：维度 128→1536，层数 4→28，头数 4→12。
- **位置编码**：这里用可学习位置嵌入，真实模型用 RoPE（教学文档第三章）。
- **分词**：这里字符级（~60 词表），真实模型 BPE 子词（15.2 万词表）。
- **归一化**：这里 LayerNorm，真实模型 RMSNorm。
- **注意力**：这里标准 MHA，真实模型 GQA（2 个 KV 头共享）。

**但主干完全一致**：都是"嵌入 + 位置 → N 层（注意力 + MLP + 残差）→ LM head → 自回归生成"。把这个迷你版吃透，你就真正理解了 Transformer，而不只是"会用 transformers 库"。
