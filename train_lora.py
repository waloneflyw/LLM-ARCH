# -*- coding: utf-8 -*-
"""LoRA 微调脚本（修正版）：DeepSeek-R1-Distill-Qwen-1.5B on RTX 5060 (8GB)"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# ---------- 1. 加载模型和分词器 ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", dtype=torch.float16)

# ---------- 2. 加载数据集（先取 200 条跑通流程，正式训练可去掉 [:200]） ----------
dataset = load_dataset("MBZUAI/LaMini-instruction", split="train[:200]")
split = dataset.train_test_split(test_size=0.1, seed=42)
train_raw, eval_raw = split["train"], split["test"]

# ---------- 3. 数据预处理（修正版：拼接指令+回应，指令部分和 padding 的标签置 -100） ----------
def preprocess(examples):
    # 1) 指令 + 回应 拼成完整一段文本
    texts = [f"指令:{ins}\n回应:{res}" for ins, res in zip(examples["instruction"], examples["response"])]
    # 2) 一次性编码完整文本
    tokenized = tokenizer(texts, max_length=512, truncation=True, padding="max_length")
    # 3) 标签 = input_ids 副本，把「指令部分」和「padding」置 -100（不参与 loss）
    labels = []
    for i, ins in enumerate(examples["instruction"]):
        prompt_len = len(tokenizer(f"指令:{ins}\n回应:")["input_ids"])
        row = tokenized["input_ids"][i][:]
        for j in range(len(row)):
            if j < prompt_len or tokenized["attention_mask"][i][j] == 0:
                row[j] = -100
        labels.append(row)
    tokenized["labels"] = labels
    return tokenized

train_dataset = train_raw.map(preprocess, batched=True, remove_columns=train_raw.column_names)
eval_dataset = eval_raw.map(preprocess, batched=True, remove_columns=eval_raw.column_names)

# ---------- 4. LoRA 配置 ----------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

# ---------- 5. 训练参数 + 训练 ----------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,      # 等效 batch = 2 * 4 = 8
    weight_decay=0.01,
    fp16=True,                          # 混合精度，省显存提速
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    save_total_limit=2,
    push_to_hub=False,
    report_to="none",
    logging_steps=10,
    label_names=["labels"],
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

trainer.train()

# ---------- 6. 保存 LoRA 权重 ----------
peft_model.save_pretrained("./lora_model")
tokenizer.save_pretrained("./lora_model")
print("训练完成，LoRA 权重已保存到 ./lora_model")
