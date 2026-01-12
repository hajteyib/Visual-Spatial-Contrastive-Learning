"""
Test simple pour vérifier si Colab fonctionne avec l'extension VSCode.
"""

import torch
import sys

def test_environment():
    """Test basique de l'environnement."""
    print("=" * 60)
    print("TEST COLAB ENVIRONMENT")
    print("=" * 60)
    
    # 1. Version Python
    print(f"\n✓ Python version: {sys.version}")
    
    # 2. PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")
    
    # 3. Device disponible
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✓ MPS available (Mac)")
    else:
        device = torch.device("cpu")
        print(f"✗ Only CPU available")
    
    # 4. Test simple de calcul
    print(f"\n✓ Testing computation on {device}...")
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    
    import time
    start = time.time()
    z = torch.matmul(x, y)
    elapsed = time.time() - start
    
    print(f"  Matrix multiplication (1000x1000): {elapsed*1000:.2f} ms")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED ✓")
    print("=" * 60)
    
    return device


if __name__ == "__main__":
    device = test_environment()
    print(f"\nDevice to use: {device}")
