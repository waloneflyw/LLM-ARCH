# -*- coding: utf-8 -*-
"""
训练脚本：从零训练一个字符级迷你 GPT
=====================================
展示一个完整的训练流程结构：
  1. 加载数据 + 字符级分词
  2. 划分训练集/验证集
  3. 构造 batch
  4. 初始化模型
  5. 训练循环（前向 → 算 loss → 反向 → 更新）
  6. 定期评估验证集
  7. 自回归生成文本
"""

import os
import torch
from model import GPT

# ================= 0. 超参数（都集中在这里，方便你调） =================
block_size = 64        # 上下文长度：一次看多少个字符
batch_size = 32        # 每个 batch 多少条样本
d_model = 128          # 隐藏维度（真实模型是 1536）
n_heads = 4            # 注意力头数
n_layers = 4           # Transformer Block 层数（真实模型 28 层）
d_ff = 4 * d_model     # MLP 中间维度（通常是 4 倍）
dropout = 0.1

lr = 3e-4              # 学习率
max_iters = 3000       # 训练总步数
eval_interval = 200    # 每隔多少步评估一次验证集
eval_iters = 100       # 评估时取多少批做平均

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)  # 固定随机种子，保证可复现

# ================= 1. 加载数据 + 字符级分词 =================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(SCRIPT_DIR, "data", "input.txt")
text = open(data_path, encoding="utf-8").read()

# 字符级分词：每个字符是一个 token，建立 字符<->整数 的映射
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}          # 字符 -> id
itos = {i: ch for i, ch in enumerate(chars)}          # id -> 字符
encode = lambda s: [stoi[c] for c in s]               # 文本 -> id 列表
decode = lambda ids: "".join(itos[i] for i in ids)    # id 列表 -> 文本

print(f"词表大小 vocab_size = {vocab_size}，文本长度 = {len(text)} 字符")
print(f"设备 = {device}")

data = torch.tensor(encode(text), dtype=torch.long)

# ================= 2. 划分训练集 / 验证集 =================
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# ================= 3. 构造 batch =================
def get_batch(split):
    """随机采样一个 batch：x 是输入，y 是"右移一位"的目标（预测下一个字符）"""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# ================= 4. 初始化模型 =================
model = GPT(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    d_ff=d_ff,
    block_size=block_size,
    dropout=dropout,
).to(device)

# 数一下参数
num_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量 = {num_params/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# ================= 5 & 6. 训练循环 + 定期评估 =================
@torch.no_grad()
def estimate_loss():
    """在训练集和验证集上各取 eval_iters 批，估算平均 loss"""
    out = {}
    model.eval()   # 切换到评估模式（关掉 dropout）
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()  # 切回训练模式
    return out

for step in range(max_iters):
    # ---- 前向：得到 logits 和 loss ----
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    # ---- 反向 + 更新参数 ----
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # ---- 定期打印训练/验证 loss ----
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

# ================= 7. 生成文本 =================
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # 起始 token
print("\n===== 模型生成（字符级小模型，效果有限属正常）=====")
print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
