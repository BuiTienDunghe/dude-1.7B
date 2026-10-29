import torch, transformers, datasets

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "datasets", datasets.__version__)
print("available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("arch_list", torch.cuda.get_arch_list())
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
torch.cuda.synchronize()
print("matmul bf16 ok", y)
