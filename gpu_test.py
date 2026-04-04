import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version used by PyTorch:", torch.version.cuda)
print("GPU name:", torch.cuda.get_device_name(0))
print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")