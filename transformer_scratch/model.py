# -*- coding: utf-8 -*-
"""
从零实现 Transformer（decoder-only，即 GPT 结构）
=================================================
这个文件不依赖 transformers，只用 PyTorch 手写每一个模块，
对应教学文档《教学文档_Transformer.md》里的结构：

    分词器 → Embedding → 位置编码 → N 层 Block（注意力 + MLP）→ LM head

对照你之前学过的名词：
  q_proj/k_proj/v_proj/o_proj  = MultiHeadSelfAttention 里的四个 Linear
  多头注意力 / 因果掩码         = 本文件的 MultiHeadSelfAttention
  MLP / 前馈网络               = FeedForward
  残差连接 + LayerNorm          = TransformerBlock
  LM head + softmax            = GPT.forward 末尾 + generate
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力（含因果掩码）"""

    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被头数整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads   # 每个头的维度

        # 三个投影：把输入投影成 Q / K / V（就是 LLM 里的 q_proj/k_proj/v_proj）
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # 输出投影：把多头的输出拼回 d_model 维（就是 o_proj）
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # 因果掩码：一个 T×T 的下三角矩阵，1 表示"可以看"，0 表示"未来，屏蔽"
        # register_buffer 表示它不是可训练参数，但会跟着模型一起 .to(device)
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape   # batch, 序列长度, 隐藏维度

        # ---- 1. 投影得到 Q/K/V，然后切分成多头 ----
        # view 把 (B, T, C) 拆成 (B, T, n_heads, head_dim)，transpose 交换维度
        # 得到 (B, n_heads, T, head_dim)：每个头独立处理
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # ---- 2. 注意力分数 = Q·Kᵀ / √d ----
        # (B, n_heads, T, T)：每个查询对每个 key 的相似度
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ---- 3. 因果掩码：把"未来位置"的分数置 -inf，softmax 后变 0 ----
        # causal_mask[:T, :T] 取当前实际长度对应的掩码
        att = att.masked_fill(self.causal_mask[:T, :T] == 0, float("-inf"))

        # ---- 4. softmax 归一化成权重，再 dropout ----
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        # ---- 5. 加权求和 V ----
        out = att @ v   # (B, n_heads, T, head_dim)

        # ---- 6. 合并多头 + o_proj ----
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)
        return out


class FeedForward(nn.Module):
    """前馈网络 MLP：先把维度放大 d_ff，再压回 d_model"""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)     # 扩张
        self.fc2 = nn.Linear(d_ff, d_model)     # 收缩
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)          # 激活函数（GPT 用 GELU，Qwen/Llama 用 SiLU）
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """一个 Transformer Block = 注意力 + MLP，各自带残差和 LayerNorm"""

    def __init__(self, d_model, n_heads, d_ff, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)        # 注意力前的归一化
        self.attn = MultiHeadSelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)        # MLP 前的归一化
        self.mlp = FeedForward(d_model, d_ff, dropout)

    def forward(self, x):
        # 残差连接：x = x + 子层(x)，让浅层信息能"抄近路"直达深层
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """decoder-only Transformer（迷你 GPT）"""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, block_size, dropout):
        super().__init__()
        self.block_size = block_size

        # 词嵌入：把 token id 查表成 d_model 维向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 位置嵌入：每个位置一个可学习向量（老式做法；现代模型用 RoPE，见教学文档第三章）
        self.position_embedding = nn.Embedding(block_size, d_model)

        # 堆叠 N 层 Block
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, d_ff, block_size, dropout)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)              # 最后的归一化
        self.lm_head = nn.Linear(d_model, vocab_size)  # 输出层：映射到词表

        # 权重共享（weight tying）：lm_head 与词嵌入共用同一套权重，省参数
        self.token_embedding.weight = self.lm_head.weight

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # token 嵌入 + 位置嵌入（相加）
        pos = torch.arange(0, T, device=idx.device)   # 位置 0,1,...,T-1
        tok_emb = self.token_embedding(idx)            # (B, T, d_model)
        pos_emb = self.position_embedding(pos)         # (T, d_model)
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)

        logits = self.lm_head(x)   # (B, T, vocab_size)

        # 如果给了 targets，就计算交叉熵 loss（训练时用）
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """自回归生成：每次预测一个 token，拼回输入，循环"""
        for _ in range(max_new_tokens):
            # 只保留最后 block_size 个 token（防止超出位置嵌入范围）
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # 只取"最后一个位置"的输出，预测下一个词
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # 按概率采样（而不是取最大，这样输出更多样）
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
