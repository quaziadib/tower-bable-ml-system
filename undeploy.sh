#!/bin/bash

# Translation API Undeployment Script
# This script removes the Translation API systemd service

set -e  # Exit on error

echo "======================================"
echo "Translation API Undeployment Script"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="translation-api"
SYSTEMD_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
LOG_DIR="/var/log/translation-api"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Please do not run this script as root. It will request sudo when needed.${NC}"
    exit 1
fi

# Step 1: Stop the service
echo -e "${YELLOW}[1/4] Stopping service...${NC}"
if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
    sudo systemctl stop "$SERVICE_NAME"
    echo -e "${GREEN}✓ Service stopped${NC}"
else
    echo -e "${GREEN}✓ Service not running${NC}"
fi
echo ""

# Step 2: Disable the service
echo -e "${YELLOW}[2/4] Disabling service...${NC}"
if sudo systemctl is-enabled --quiet "$SERVICE_NAME" 2>/dev/null; then
    sudo systemctl disable "$SERVICE_NAME"
    echo -e "${GREEN}✓ Service disabled${NC}"
else
    echo -e "${GREEN}✓ Service not enabled${NC}"
fi
echo ""

# Step 3: Remove systemd service file
echo -e "${YELLOW}[3/4] Removing systemd service file...${NC}"
if [ -f "$SYSTEMD_PATH" ]; then
    sudo rm "$SYSTEMD_PATH"
    echo -e "${GREEN}✓ Service file removed${NC}"
else
    echo -e "${GREEN}✓ Service file not found${NC}"
fi
echo ""

# Step 4: Reload systemd daemon
echo -e "${YELLOW}[4/4] Reloading systemd daemon...${NC}"
sudo systemctl daemon-reload
sudo systemctl reset-failed
echo -e "${GREEN}✓ Systemd daemon reloaded${NC}"
echo ""

# Ask about log directory
echo -e "${YELLOW}Do you want to remove log directory ($LOG_DIR)? [y/N]${NC}"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    if [ -d "$LOG_DIR" ]; then
        sudo rm -rf "$LOG_DIR"
        echo -e "${GREEN}✓ Log directory removed${NC}"
    fi
else
    echo -e "${GREEN}✓ Log directory preserved${NC}"
fi
echo ""

echo -e "${GREEN}======================================"
echo "Undeployment Complete!"
echo "======================================${NC}"
echo ""
echo "The Translation API service has been removed."
echo "The application files remain in the project directory."
echo ""
