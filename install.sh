#!/bin/bash
# Tantalum Installer
# Copyright 2026 Skystarry-AI
# SPDX-License-Identifier: Apache-2.0

set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
DIM='\033[2m'
RESET='\033[0m'

VERSION="0.1.0"

info()    { echo -e "${CYAN}●${RESET} $1"; }
success() { echo -e "${GREEN}✓${RESET} $1"; }
error()   { echo -e "${RED}✗${RESET} $1"; exit 1; }
dim()     { echo -e "${DIM}  $1${RESET}"; }

echo
echo -e "${CYAN}Tantalum${RESET} ${DIM}zero-trust AI agent installer${RESET}"
echo

# ---------------------------------------------------------------------------
# OS Check
# ---------------------------------------------------------------------------
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    error "Linux only. (detected: $OSTYPE)"
fi

ARCH=$(uname -m)
if [[ "$ARCH" != "x86_64" ]]; then
    error "Unsupported architecture: $ARCH (Only x86_64 is supported in v${VERSION})"
fi

# ---------------------------------------------------------------------------
# Dependency Helper
# ---------------------------------------------------------------------------
require() {
    command -v "$1" &>/dev/null || error "$1 is required but not installed."
}

# ---------------------------------------------------------------------------
# Container Runtime Detection
# ---------------------------------------------------------------------------
info "Checking container runtime..."

RUNTIME=""

if command -v podman &>/dev/null; then
    RUNTIME="podman"
    success "Podman already installed ($(podman --version | awk '{print $3}'))"
elif command -v docker &>/dev/null; then
    RUNTIME="docker"
    success "Docker already installed ($(docker --version | cut -d' ' -f3 | tr -d ','))"
else
    info "No container runtime found. Installing Podman..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq podman
    RUNTIME="podman"
    success "Podman installed"
fi

# Docker-specific: ensure daemon is running and accessible
if [[ "$RUNTIME" == "docker" ]]; then
    if ! docker info &>/dev/null; then
        info "Configuring Docker socket permissions..."
        sudo systemctl start docker
        sudo usermod -aG docker "$USER"
        sudo mkdir -p /etc/systemd/system/docker.socket.d
        printf '[Socket]\nSocketMode=0666\n' | sudo tee /etc/systemd/system/docker.socket.d/override.conf > /dev/null
        sudo systemctl daemon-reload
        sudo systemctl restart docker.socket
        success "Docker socket configured (no re-login required)"
    fi
fi

# ---------------------------------------------------------------------------
# Python Dependencies
# ---------------------------------------------------------------------------
info "Checking Python..."
require python3
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)"; then
    success "Python ${PYTHON_VERSION} OK"
else
    error "Python 3.11+ required (found ${PYTHON_VERSION})"
fi

info "Checking Pipx..."
if command -v pipx &>/dev/null; then
    success "Pipx already installed"
else
    info "Pipx is not installed. Installing pipx..."
    sudo apt update -qq && sudo apt install -y -qq pipx
    export PATH="$PATH:$HOME/.local/bin"
    pipx ensurepath
    success "Pipx installed and PATH updated for this session"
fi

# ---------------------------------------------------------------------------
# Download Gateway Binary
# ---------------------------------------------------------------------------
info "Downloading Tantalum Gateway v${VERSION}..."

GATEWAY_DIR="service"
GATEWAY_BIN="${GATEWAY_DIR}/tantalum-gateway"
DOWNLOAD_URL="https://github.com/skystarry-ai/tantalum/releases/download/${VERSION}/tantalum-gateway"

mkdir -p "$GATEWAY_DIR"

if curl -L -o "$GATEWAY_BIN" "$DOWNLOAD_URL"; then
    chmod +x "$GATEWAY_BIN"
    success "Gateway binary downloaded for ${ARCH}"
else
    echo
    error "Failed to download gateway binary from GitHub Releases.\n  URL: $DOWNLOAD_URL\n  Please check your internet connection or if the release exists."
fi

# ---------------------------------------------------------------------------
# Docker Configuration
# ---------------------------------------------------------------------------
info "Setting up container volume..."
$RUNTIME volume create tantalum-storage &>/dev/null
success "Volume 'tantalum-storage' ready"

info "Pulling container image..."
$RUNTIME pull --quiet python:3.12-slim
success "Image 'python:3.12-slim' ready"

# ---------------------------------------------------------------------------
# Configuration Template
# ---------------------------------------------------------------------------
CONFIG_DIR="$HOME/.config/tantalum"
CONFIG_FILE="$CONFIG_DIR/config.env"

info "Setting up configuration directory..."
mkdir -p "$CONFIG_DIR"

if [[ ! -f "$CONFIG_FILE" ]]; then
    cat > "$CONFIG_FILE" << 'ENVEOF'
# Tantalum Environment Variables
# Edit this file to configure your AI agent

# Container runtime (auto-detected: podman preferred over docker)
# TANTALUM_RUNTIME=podman

# LLM (choose one)
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# Model selection (default: gemini/gemini-2.5-flash)
# TANTALUM_MODEL=gemini/gemini-2.5-flash
# TANTALUM_MODEL=claude-haiku-4-5
# TANTALUM_MODEL=ollama/llama3

# System prompt (optional)
# TANTALUM_SYSTEM_PROMPT=You are a helpful assistant.

# Docker settings (optional)
# TANTALUM_VOLUME=tantalum-storage
# TANTALUM_DOCKER_IMAGE=python:3.12-slim
# TANTALUM_DOCKER_MEMORY=256m
# TANTALUM_DOCKER_CPUS=0.5
# TANTALUM_DOCKER_TIMEOUT=30
ENVEOF
    success "Configuration created at $CONFIG_FILE"
    dim "Please edit $CONFIG_FILE and add your API key."
else
    success "Configuration already exists at $CONFIG_FILE"
fi

# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------
echo
echo -e "${GREEN}Tantalum installed successfully.${RESET}"
echo
dim "Start the CLI (it will launch gateway and brain automatically):"
dim "Next Step: Run the command \"pipx install --include-deps .\""
dim "  And run command tantalum"
echo