#!/bin/bash

# Translation API Deployment Script
# This script sets up the Translation API as a systemd service

set -e  # Exit on error

echo "======================================"
echo "Translation API Deployment Script"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="translation-api"
SERVICE_FILE="translation-api.service"
SYSTEMD_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
LOG_DIR="/var/log/translation-api"
WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Working directory: $WORKING_DIR"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Please do not run this script as root. It will request sudo when needed.${NC}"
    exit 1
fi

# Step 1: Create log directory
echo -e "${YELLOW}[1/6] Creating log directory...${NC}"
if [ ! -d "$LOG_DIR" ]; then
    sudo mkdir -p "$LOG_DIR"
    sudo chown $USER:$USER "$LOG_DIR"
    echo -e "${GREEN}✓ Log directory created: $LOG_DIR${NC}"
else
    echo -e "${GREEN}✓ Log directory already exists: $LOG_DIR${NC}"
fi
echo ""

# Step 2: Stop existing service if running
echo -e "${YELLOW}[2/6] Stopping existing service (if running)...${NC}"
if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
    sudo systemctl stop "$SERVICE_NAME"
    echo -e "${GREEN}✓ Service stopped${NC}"
else
    echo -e "${GREEN}✓ Service not running${NC}"
fi
echo ""

# Step 3: Copy systemd service file
echo -e "${YELLOW}[3/6] Installing systemd service file...${NC}"
if [ ! -f "$WORKING_DIR/$SERVICE_FILE" ]; then
    echo -e "${RED}Error: Service file not found: $WORKING_DIR/$SERVICE_FILE${NC}"
    exit 1
fi

sudo cp "$WORKING_DIR/$SERVICE_FILE" "$SYSTEMD_PATH"
echo -e "${GREEN}✓ Service file installed to: $SYSTEMD_PATH${NC}"
echo ""

# Step 4: Reload systemd daemon
echo -e "${YELLOW}[4/6] Reloading systemd daemon...${NC}"
sudo systemctl daemon-reload
echo -e "${GREEN}✓ Systemd daemon reloaded${NC}"
echo ""

# Step 5: Enable service to start on boot
echo -e "${YELLOW}[5/6] Enabling service to start on boot...${NC}"
sudo systemctl enable "$SERVICE_NAME"
echo -e "${GREEN}✓ Service enabled${NC}"
echo ""

# Step 6: Start the service
echo -e "${YELLOW}[6/6] Starting the service...${NC}"
sudo systemctl start "$SERVICE_NAME"

# Wait a moment for the service to start
sleep 3

# Check if service started successfully
if sudo systemctl is-active --quiet "$SERVICE_NAME"; then
    echo -e "${GREEN}✓ Service started successfully${NC}"
else
    echo -e "${RED}✗ Service failed to start${NC}"
    echo "Check logs with: sudo journalctl -u $SERVICE_NAME -n 50"
    exit 1
fi
echo ""

# Display service status
echo "======================================"
echo "Service Status:"
echo "======================================"
sudo systemctl status "$SERVICE_NAME" --no-pager -l
echo ""

# Display helpful commands
echo "======================================"
echo "Useful Commands:"
echo "======================================"
echo "View logs:        sudo journalctl -u $SERVICE_NAME -f"
echo "View recent logs: sudo journalctl -u $SERVICE_NAME -n 100"
echo "Stop service:     sudo systemctl stop $SERVICE_NAME"
echo "Start service:    sudo systemctl start $SERVICE_NAME"
echo "Restart service:  sudo systemctl restart $SERVICE_NAME"
echo "Service status:   sudo systemctl status $SERVICE_NAME"
echo "Disable service:  sudo systemctl disable $SERVICE_NAME"
echo ""
echo "Log files:"
echo "  Output: $LOG_DIR/output.log"
echo "  Errors: $LOG_DIR/error.log"
echo ""

# Test the API
echo "======================================"
echo "Testing API..."
echo "======================================"
sleep 2

if command -v curl &> /dev/null; then
    echo "Health check:"
    curl -s http://localhost:9000/health | python3 -m json.tool || echo "API not responding yet (may still be loading model)"
    echo ""
    echo ""
    echo "API Info:"
    curl -s http://localhost:9000/ | python3 -m json.tool || echo "API not responding yet"
else
    echo "curl not found. Install it to test the API."
fi
echo ""

echo -e "${GREEN}======================================"
echo "Deployment Complete!"
echo "======================================${NC}"
echo ""
echo "The Translation API is now running as a systemd service."
echo "It will automatically start on system boot."
echo ""
echo "API endpoint: http://$(hostname -I | awk '{print $1}'):9000"
echo ""
