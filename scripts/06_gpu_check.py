"""Step 6: Verify GPU is available for training."""

import torch

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    props = torch.cuda.get_device_properties(0)
    mem = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    print(f"Memory: {mem / 1e9:.1f} GB")
    print("PASS")
else:
    print("WARNING: No CUDA GPU detected — training will use CPU")
    print("PASS (cpu-only)")
