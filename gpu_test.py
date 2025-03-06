# 用于测试pytorch gpu是否可用
import torch
if torch.cuda.is_available():
    print("GPU is available")
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), "GB")
else:
    print("GPU is not available")