"""Quick check that the training stack can see and use the GPU."""

import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul ok", float((x @ x).float().mean()) == float((x @ x).float().mean()))

import bitsandbytes, trl, unsloth  # noqa: E402

print("unsloth", unsloth.__version__, "trl", trl.__version__, "bitsandbytes", bitsandbytes.__version__)
