# M0 runbook: machine setup

Everything runs inside WSL2 (Ubuntu 24.04). The repo lives at `~/agritune` on the WSL disk;
from Windows it is at `\\wsl.localhost\Ubuntu-24.04\home\msoli\agritune`.

## What was installed

| Piece | Version | Notes |
|---|---|---|
| WSL distro | Ubuntu 24.04.5 | user `msoli`, passwordless sudo, systemd on |
| WSL memory | 16 GB + 8 GB swap | `%USERPROFILE%\.wslconfig` |
| GPU driver (Windows) | 621.01, CUDA 13.5 | RTX 5070 Ti Laptop, 12 GB, compute 12.0 |
| PyTorch / Triton | 2.14.1+cu130 / 3.8.0 | default PyPI build, no index pin |
| Unsloth / TRL / bitsandbytes | 2026.10.8 / 1.15.0 / 0.50.2 | |
| uv | 0.13.5 | |
| Docker Engine / Compose | 29.9.2 / v5.6.1 | installed inside WSL, not Rancher Desktop |

## Daily commands

```bash
cd ~/agritune
bash infra/up.sh                                   # start postgres, redis, mlflow, qdrant
bash services/train/run.sh gpu_check
bash services/train/run.sh bench --preset qwen-1.7b-lora
```

MLflow UI: http://localhost:5000 (opens from a Windows browser too).

## Status (2026-10-24)

- M0 done: WSL, uv, training stack, Docker, infra stack, driver update, `gpu_check` and
  `triton_check` pass, benchmarks below.
- Decision: gemma-12b-qlora is dropped from the presets; qwen-8b-qlora is the largest model.

## Benchmarks (2026-10-24)

LoRA r=16 on all linear layers, batch 1, 25 steps (5 warm-up), bf16, adamw_8bit, gradient
checkpointing. Run one preset per process; results append to `services/train/bench_results.jsonl`.

| Preset | Seq | Peak VRAM (of 11.9 GB) | Step | Tokens/s | Peak WSL RAM |
|---|---|---|---|---|---|
| qwen-1.7b-lora (16-bit) | 4096 | 4.08 GB | 1.54 s | 2,665 | 7.1 GB |
| gemma-4b-qlora | 2048 | 5.02 GB | 1.61 s | 1,273 | 8.1 GB |
| qwen-8b-qlora | 2048 | 7.88 GB | 2.85 s | 719 | 7.3 GB |
| gemma-12b-qlora | 1024 | did not load | — | — | 5.4 GB |

First loads include the model download (gemma-4b 382 s, qwen-8b 655 s); later loads are cached.

## Known issues

- **gemma-12b-qlora does not fit** (dropped). With 10.69 GB free on the GPU, transformers puts
  part of the model on the CPU, and bitsandbytes cannot quantize those layers. Unsloth suggests
  `offload_layers="auto"`, which keeps the extra layers in host RAM and will be much slower.
- **xformers mismatch.** The installed xformers was built for torch 2.10; Unsloth falls back to
  PyTorch attention. `unsloth install-kernels xformers` rebuilds it if speed matters.
- **Windows PATH inside WSL.** Rancher Desktop's Windows `docker-credential-secretservice` leaked
  in and broke image pulls. Now disabled with `[interop] appendWindowsPath=false` in
  `/etc/wsl.conf`; `infra/up.sh` also strips `/mnt/*` from PATH.
- **Triton needs Python headers.** Ubuntu's Python lacks them by default: `python3.12-dev`.
- **Resolved: driver too old for CUDA 13.** Driver 577.13 (CUDA 12.9) made Triton 3.7.1
  segfault. Fixed by updating to 621.01 and dropping the cu129 torch pin.
