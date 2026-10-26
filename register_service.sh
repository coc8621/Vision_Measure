#!/bin/bash

# 获取当前用户、家目录和项目目录
USER_NAME=${SUDO_USER:-$(whoami)}
HOME_DIR=$(getent passwd "$USER_NAME" | cut -d: -f6)
USER_ID=$(id -u "$USER_NAME")
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="vision-measurement.service"
SERVICE_PATH="/etc/systemd/system/$SERVICE_NAME"

# 检查是否为 root 用户，因为修改 systemd 配置需要 root 权限
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root."
    exit 1
fi

chmod +x "$PROJECT_DIR/AutoStart.sh"

# 创建 systemd service 文件内容，直接启动项目
read -r -d '' SERVICE_CONTENT << EOF
[Unit]
Description=Vision Measurement UI
After=network.target display-manager.service

[Service]
User=$USER_NAME
Type=simple
WorkingDirectory=$PROJECT_DIR
Environment="HOME=$HOME_DIR"
Environment="DISPLAY=${DISPLAY:-:0}"
Environment="XAUTHORITY=${XAUTHORITY:-/run/user/$USER_ID/gdm/Xauthority}"
Environment="XDG_RUNTIME_DIR=/run/user/$USER_ID"
ExecStart=/bin/bash "$PROJECT_DIR/AutoStart.sh"
Restart=always
TimeoutStopSec=10
RestartSec=10s

[Install]
WantedBy=graphical.target
EOF

# 创建/覆写 systemd service 文件
echo "Creating systemd service file at $SERVICE_PATH..."
echo "$SERVICE_CONTENT" > "$SERVICE_PATH"

# 重新加载 systemd 配置
echo "Reloading systemd daemon..."
systemctl daemon-reload

# 启用服务
echo "Enabling service $SERVICE_NAME..."
systemctl enable "$SERVICE_NAME"

# 启动服务
echo "Starting service $SERVICE_NAME..."
systemctl start "$SERVICE_NAME"

echo "Service $SERVICE_NAME has been registered and started."
