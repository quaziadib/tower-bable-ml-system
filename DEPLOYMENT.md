# Translation API Deployment Guide

This guide explains how to deploy the Translation API as a systemd service for production use.

## Prerequisites

- Ubuntu/Debian-based system with systemd
- Python 3.8+ installed
- Virtual environment set up with dependencies installed
- NVIDIA GPU with CUDA support (for model inference)
- Minimum 16GB GPU VRAM

## Quick Deployment

### 1. Deploy the Service

Run the deployment script:

```bash
./deploy.sh
```

The script will:
- Create log directory at `/var/log/translation-api/`
- Install systemd service file
- Enable service to start on boot
- Start the service
- Display service status

### 2. Verify Deployment

Check service status:
```bash
sudo systemctl status translation-api
```

Test the API:
```bash
# Health check
curl http://localhost:8000/health

# Translation test
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "source_language": "English", "target_language": "Bangla"}'
```

## Managing the Service

### Start/Stop/Restart

```bash
# Start
sudo systemctl start translation-api

# Stop
sudo systemctl stop translation-api

# Restart
sudo systemctl restart translation-api

# Status
sudo systemctl status translation-api
```

### View Logs

```bash
# Follow live logs
sudo journalctl -u translation-api -f

# View recent logs
sudo journalctl -u translation-api -n 100

# View log files directly
tail -f /var/log/translation-api/output.log
tail -f /var/log/translation-api/error.log
```

### Enable/Disable Auto-start

```bash
# Enable auto-start on boot
sudo systemctl enable translation-api

# Disable auto-start
sudo systemctl disable translation-api
```

## Undeployment

To remove the service:

```bash
./undeploy.sh
```

This will:
- Stop the service
- Disable auto-start
- Remove systemd service file
- Optionally remove log directory

## Public Access

### Firewall Configuration

If you have a firewall enabled (ufw), allow port 8000:

```bash
sudo ufw allow 8000/tcp
sudo ufw status
```

### Access the API

The API will be accessible at:
- Local: `http://localhost:8000`
- Network: `http://<your-server-ip>:8000`

Example:
```bash
# From another machine on the network
curl http://192.168.1.100:8000/health
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /translate` - Translate text

### Translation Request Format

```json
{
  "text": "Text to translate",
  "source_language": "English",
  "target_language": "Bangla"
}
```

### Translation Response Format

```json
{
  "original_text": "Text to translate",
  "translated_text": "অনুবাদিত পাঠ্য",
  "source_language": "English",
  "target_language": "Bangla"
}
```

## Supported Languages

- English
- Bangla
- Hindi
- Spanish
- French
- German
- Chinese
- Japanese
- Arabic
- Portuguese

## Troubleshooting

### Service Won't Start

Check logs for errors:
```bash
sudo journalctl -u translation-api -n 50
```

Common issues:
- **Model not downloaded**: First run takes ~5 minutes to download the model
- **GPU not available**: Ensure CUDA is properly installed
- **Port 8000 in use**: Another process is using the port
- **Permission issues**: Check file permissions in the working directory

### Model Loading Takes Long

The first time the service starts, it needs to:
1. Download the model (~18GB) - takes 5-30 minutes depending on internet speed
2. Load model into GPU memory - takes ~30 seconds

Check download progress:
```bash
sudo journalctl -u translation-api -f
```

### Out of Memory

If you get CUDA out of memory errors:
- Ensure no other GPU processes are running
- The model requires ~18GB of GPU VRAM
- Check GPU usage: `nvidia-smi`

## Performance Notes

- **First request**: May take 10-15 seconds (model warmup)
- **Subsequent requests**: Typically 1-3 seconds per translation
- **Concurrent requests**: The API handles one request at a time due to GPU constraints

## Security Considerations

- The API currently allows all CORS origins (`allow_origins=["*"]`)
- For production, consider:
  - Restricting CORS to specific domains
  - Adding authentication/API keys
  - Using reverse proxy (nginx/apache) with SSL
  - Rate limiting

## Advanced Configuration

### Change Port

Edit `api.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)  # Change port here
```

Then restart:
```bash
sudo systemctl restart translation-api
```

### Resource Limits

Edit the systemd service file to add resource limits:
```bash
sudo nano /etc/systemd/system/translation-api.service
```

Add under `[Service]`:
```ini
MemoryLimit=20G
CPUQuota=200%
```

Reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart translation-api
```

## Monitoring

### Check if service is running:
```bash
sudo systemctl is-active translation-api
```

### Check if enabled on boot:
```bash
sudo systemctl is-enabled translation-api
```

### Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

## Backup and Recovery

### Backup the service configuration:
```bash
cp translation-api.service translation-api.service.backup
```

### Restore from backup:
```bash
cp translation-api.service.backup translation-api.service
./deploy.sh
```
