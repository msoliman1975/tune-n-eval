#!/usr/bin/env bash
# Lists the newest PyTorch wheels for Python 3.12 built against CUDA 12.8 and 12.9.
cd ~/agritune/services/train
.venv/bin/python -c 'import torch; print("installed:", torch.__version__, "cuda", torch.version.cuda)'
for cu in cu128 cu129; do
  echo "== $cu"
  curl -s "https://download.pytorch.org/whl/$cu/torch/" \
    | grep -oE "torch-2\.[0-9]+\.[0-9]+(%2B|\+)$cu-cp312-cp312-manylinux[^\"#]*x86_64\.whl" \
    | sed 's/%2B/+/' | sort -uV | tail -3
done
