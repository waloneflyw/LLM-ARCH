"""环境自检：确认 torch / CUDA / bitsandbytes 在 RTX 5060 上可用。"""
import torch
import bitsandbytes as bnb
import transformers, peft, datasets, accelerate

print("=" * 50)
print("Python 3.11 环境：lora")
print("=" * 50)

print(f"torch           : {torch.__version__}")
print(f"torch CUDA      : {torch.version.cuda}")
print(f"cuda available  : {torch.cuda.is_available()}")
print(f"device count    : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"device name     : {torch.cuda.get_device_name(0)}")
    print(f"capability      : {torch.cuda.get_device_capability(0)}")  # 期望 (12, 0) = sm_120

# 真正在 GPU 上跑一次矩阵运算，验证 sm_120 内核可用
x = torch.randn(1024, 1024, device="cuda")
y = torch.randn(1024, 1024, device="cuda")
z = x @ y
torch.cuda.synchronize()
print(f"GPU matmul OK   : {z.shape}, sum={z.sum().item():.4f}")

print("-" * 50)
print(f"transformers    : {transformers.__version__}")
print(f"peft            : {peft.__version__}")
print(f"datasets        : {datasets.__version__}")
print(f"accelerate      : {accelerate.__version__}")
print(f"bitsandbytes    : {bnb.__version__}")
print("=" * 50)
print("环境就绪 ✓")
