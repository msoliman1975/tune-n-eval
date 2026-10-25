"""Compiles and runs one tiny Triton kernel; prints the full compiler error if it fails."""

import subprocess

import torch
import triton
import triton.language as tl


@triton.jit
def add(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out_ptr + i, tl.load(x_ptr + i, mask=m) + tl.load(y_ptr + i, mask=m), mask=m)


x = torch.randn(1000, device="cuda")
y = torch.randn(1000, device="cuda")
out = torch.empty_like(x)
try:
    add[(4,)](x, y, out, 1000, BLOCK=256)
    print("triton", triton.__version__, "ok:", torch.allclose(out, x + y))
except subprocess.CalledProcessError as e:
    print("CMD:", " ".join(e.cmd))
    print("STDERR:", (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else e.stderr)
    raise
