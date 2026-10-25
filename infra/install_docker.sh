#!/usr/bin/env bash
# Installs Docker Engine + Compose plugin inside WSL Ubuntu from Docker's official apt repo.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
. /etc/os-release
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update -qq
sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin >/dev/null
sudo usermod -aG docker "$USER"
sudo systemctl enable --now docker >/dev/null 2>&1 || true
sudo docker version --format 'docker {{.Server.Version}}'
docker compose version
