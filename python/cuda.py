import torch
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Arch list: {torch.cuda.get_arch_list()}")
# Test bojowy - to musi przejść bez błędu:
try:
    torch.randn(10, 10).cuda()
    print("✅ GPU działa i alokuje pamięć!")
except Exception as e:
    print(f"❌ Nadal błąd: {e}")