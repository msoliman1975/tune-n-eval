# M0 runbook: machine setup

Everything runs inside WSL2 (Ubuntu 24.04). The repo lives at `~/agritune` on the WSL disk;
from Windows it is at `\\wsl.localhost\Ubuntu-24.04\home\msoli\agritune`.

## What was installed

| Piece | Version | Notes |
|---|---|---|
| WSL distro | Ubuntu 24.04.5 | user `msoli`, passwordless sudo, systemd on |
| WSL memory | 24 GB + 8 GB swap | `%USERPROFILE%\.wslconfig` |
| GPU driver (Windows) | 577.13, CUDA 12.9 | RTX 5070 Ti Laptop, 12 GB, compute 12.0 |
| PyTorch | 2.13.0+cu129 | pinned to the cu129 index in `services/train/pyproject.toml` |
| Unsloth / TRL / bitsandbytes | 2026.10.8 / 1.15.0 / 0.50.2 | |
| uv | 0.13.5 | |
| Docker Engine / Compose | 29.9.2 / v5.6.1 | installed inside WSL, not Rancher Desktop |

## Daily commands

```bash
cd ~/agritune
bash infra/up.sh                                   # start postgres, redis, mlflow, qdrant
cd services/train && uv run python -m agritune_train.gpu_check
uv run python -m agritune_train.bench --preset qwen-1.7b-lora
```

MLflow UI: http://localhost:5000 (opens from a Windows browser too).

## Status (2026-10-25)

- Done: WSL, uv, training stack, Docker, infra stack healthy, `python3.12-dev` installed,
  WSL memory lowered to 16 GB, Windows PATH disabled in `/etc/wsl.conf`.
- Blocked: Triton 3.7.1 (required by torch 2.13) segfaults when loading a kernel. Its bundled
  `cuda.h` is CUDA 13.1 and driver 577.13 only supports 12.9. The user updated the NVIDIA
  driver and is rebooting.

## Resume after the driver update

1. `nvidia-smi` inside WSL should report CUDA Version 13.x.
2. Remove the `[tool.uv.sources]` and `[[tool.uv.index]]` cu129 blocks and the `torch==2.13.0`
   pin from `services/train/pyproject.toml`, then `uv sync`.
3. `bash services/train/run.sh gpu_check` and `bash services/train/run.sh triton_check`.
4. Benchmark one preset at a time: `bash services/train/run.sh bench --preset qwen-1.7b-lora`,
   then gemma-4b-qlora, qwen-8b-qlora, gemma-12b-qlora. Watch RAM: an earlier `--all` run was
   killed when the machine ran low on memory (WSL then had 24 GB of 32 GB).
5. `bash infra/up.sh` if the containers are not already up (they restart with Docker).

## Known issues

- **Driver too old for CUDA 13** (577.13). PyPI's default torch and Triton 3.7 need CUDA 13.
  Fixed by the driver update; drop the cu129 pin afterwards.
- **Windows PATH inside WSL.** Rancher Desktop's Windows `docker-credential-secretservice` leaked
  in and broke image pulls. Now disabled with `[interop] appendWindowsPath=false` in
  `/etc/wsl.conf`; `infra/up.sh` also strips `/mnt/*` from PATH.
- **Triton needs Python headers.** Ubuntu's Python lacks them by default: `python3.12-dev`.
