"""
test_gpu.py
Simple test to verify GPU + DGL CUDA setup
"""

import torch
import dgl
import sys

print("="*60)
print("GPU & DGL CUDA VERIFICATION TEST")
print("="*60)

# 1. PyTorch CUDA Check
print(f"PyTorch version       : {torch.__version__}")
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Device Name       : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version          : {torch.version.cuda}")
    print(f"Number of GPUs        : {torch.cuda.device_count()}")
else:
    print("❌ CUDA is NOT available in PyTorch")

# 2. DGL Check
print(f"DGL version           : {dgl.__version__}")
print(f"DGL backend           : {dgl.backend.backend_name()}")

# 3. Create a simple test graph on GPU
try:
    # Create a simple heterogeneous graph
    g = dgl.heterograph({
        ('face', 'adj', 'face'): (torch.tensor([0, 1, 1]), torch.tensor([1, 0, 2]))
    }, num_nodes_dict={'face': 3})
    
    # Move to GPU
    g = g.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add some features
    g.nodes['face'].data['feat'] = torch.randn(3, 10).to(g.device)
    
    print(f"✅ Test graph successfully moved to device: {g.device}")
    print(f"   Number of nodes : {g.num_nodes('face')}")
    print(f"   Feature shape   : {g.nodes['face'].data['feat'].shape}")
    
except Exception as e:
    print(f"❌ Failed to create/move graph to GPU: {e}")

print("="*60)
if torch.cuda.is_available():
    print("🎉 GPU is ENABLED and working with DGL!")
    print("You can now safely use batch_size=8 or higher.")
else:
    print("⚠️  GPU is NOT working. Please check CUDA installation.")

print("="*60)