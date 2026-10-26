#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
# systemd 不读取 ~/.bashrc；MVS Python SDK 加载动态库时需要这些环境变量。
export MVCAM_SDK_PATH="${MVCAM_SDK_PATH:-/opt/MVS}"
export MVCAM_COMMON_RUNENV="${MVCAM_COMMON_RUNENV:-$MVCAM_SDK_PATH/lib}"
export MVCAM_GENICAM_CLPROTOCOL="${MVCAM_GENICAM_CLPROTOCOL:-$MVCAM_COMMON_RUNENV/CLProtocol}"
case "$(uname -m)" in
    aarch64|arm64) MVS_LIB_ARCH=aarch64 ;;
    armv7*|armv6*) MVS_LIB_ARCH=armhf ;;
    x86_64) MVS_LIB_ARCH=64 ;;
    *) MVS_LIB_ARCH=32 ;;
esac
export LD_LIBRARY_PATH="$MVCAM_COMMON_RUNENV/$MVS_LIB_ARCH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Set VISION_PYTHON to the interpreter containing PyQt5/OpenCV, if necessary.
exec "${VISION_PYTHON:-python3}" -m core.main --fullscreen "$@"
