# -*- coding: utf-8 -*-
"""加载微调后的 LoRA 模型，测试推理效果。"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
adapter_path = "./lora_model"

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", dtype=torch.float16)
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

def generate(text):
    prompt = f"指令:{text}\n回应:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).split("回应:")[-1].strip()

for q in ["List 3 benefits of regular exercise", "What is machine learning?"]:
    print("问题:", q)
    print("回答:", generate(q))
    print("-" * 50)
