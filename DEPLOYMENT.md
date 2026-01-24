# AI Voice Platform v2 - Deployment Guide

## Ubuntu Server Requirements

### Minimum Specifications

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4 GB | 8 GB |
| Disk | 20 GB SSD | 50 GB SSD |
| OS | Ubuntu 22.04 LTS | Ubuntu 24.04 LTS |

### Why These Specs?

- **CPU**: Real-time audio processing (STT/TTS) requires CPU cycles
- **RAM**: PostgreSQL + Redis + Python workers need memory
- **SSD**: Faster I/O for database operations

---

## Initial Server Setup

### 1. Update System

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (optional)
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Verify installation
docker --version
docker compose version
```

### 3. Configure Firewall (UFW)

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Enable firewall
sudo ufw enable
```

---

## Sophos Firewall Rules

### Ports to Forward

If your server is behind a Sophos firewall, configure DNAT (port forwarding) rules:

| External Port | Internal Port | Protocol | Destination | Purpose |
|---------------|---------------|----------|-------------|---------|
| 80 | 80 | TCP | Server IP | HTTP (Let's Encrypt) |
| 443 | 443 | TCP | Server IP | HTTPS (API) |
| 8000 | 8000 | TCP | Server IP | API (optional, dev only) |
| 8001 | 8001 | TCP | Server IP | WebSocket (optional, dev only) |

### Sophos Configuration Steps

1. **Create DNAT Rules**:
   - Go to: **Rules and policies > NAT > DNAT**
   - Add new rule for each port

2. **Create Firewall Rules**:
   - Go to: **Rules and policies > Firewall rules**
   - Allow traffic from WAN to server on ports 80, 443

3. **Dynamic DNS (Optional)**:
   - If you don't have static IP, set up DDNS
   - Go to: **Routing > Dynamic DNS**

### Example NAT Rule (HTTPS)

```
Name: AI Voice Platform - HTTPS
Source: Any
Inbound Interface: WAN
Service: TCP/443
Translation Host: <your server internal IP>
Translation Port: 443
```

---

## Deployment Steps

### 1. Copy Files to Server

```bash
# From your local machine
scp -r ai-voice-platform-v2/ user@your-server:/home/user/
```

### 2. Configure Environment Variables

```bash
cd ai-voice-platform-v2
cp .env.example .env
nano .env  # Edit with your API keys
```

### 3. Start Services

```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f app
```

### 4. Configure SSL (Let's Encrypt)

For production, use Nginx reverse proxy with SSL:

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx -y

# Get certificate
sudo certbot --nginx -d your-domain.com
```

---

## Testing the Deployment

### 1. Health Check

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "ai-voice-platform",
  "version": "2.0.0",
  "environment": "production"
}
```

### 2. Twilio Webhook Test

Twilio will call: `https://your-domain.com/api/incoming-call`

This should return TwiML XML that connects to your WebSocket.

### 3. WebSocket Connection

Twilio will connect to: `wss://your-domain.com/ws/calls`

Make sure your firewall/proxy supports WebSocket upgrade.

---

## Production Checklist

- [ ] Ubuntu 22.04+ installed
- [ ] Docker and Docker Compose installed
- [ ] Firewall configured (UFW + Sophos)
- [ ] SSL certificate installed (Let's Encrypt)
- [ ] All API keys configured in `.env`
- [ ] Twilio number pointing to your server
- [ ] MS Bookings configured (if using)
- [ ] PostgreSQL persistent volume configured
- [ ] Log rotation configured
- [ ] Monitoring/alerting set up

---

## Troubleshooting

### Issue: "Connection refused" on Twilio webhook

**Check**:
- Firewall allows port 443
- Docker containers are running: `docker compose ps`
- Nginx is configured correctly (if using)

### Issue: WebSocket connection drops

**Check**:
- Your reverse proxy supports WebSocket upgrade
- Sophos firewall has long connection timeout (300s+)
- Nginx configuration includes:
  ```
  proxy_set_header Upgrade $http_upgrade;
  proxy_set_header Connection "upgrade";
  ```

### Issue: High CPU usage

**Solution**:
- Reduce worker count in `.env`: `WEB_CONCURRENCY=2`
- Check if semantic FAQ matcher is loading (preload=false)

---

## Monitoring Commands

```bash
# View all logs
docker compose logs -f

# View only app logs
docker compose logs -f app

# Check resource usage
docker stats

# Restart a service
docker compose restart app

# Stop all services
docker compose down
```

---

## Security Notes

1. **Never commit `.env` file** to version control
2. **Use strong passwords** for PostgreSQL
3. **Enable SSL** in production
4. **Keep Docker images updated**
5. **Restrict API access** by IP if possible
6. **Monitor logs** for suspicious activity
