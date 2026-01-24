# Sophos Firewall Configuration for AI Voice Platform

## Overview

This document provides step-by-step instructions for configuring your Sophos firewall to allow HTTPS access to the AI Voice Platform.

### Network Details

- **Domain:** agent.ghaben.ca
- **Public IP:** 76.71.165.113
- **Internal Server IP:** 192.168.0.177
- **Application Port:** 8000
- **nginx Proxy Port:** 443 (HTTPS) and 80 (HTTP)

### Architecture

```
Internet → Sophos Firewall (76.71.165.113) → nginx (192.168.0.177:443) → App (192.168.0.177:8000)
```

---

## Step 1: Create Web Server Host

1. Go to **Servers > Web Servers**
2. Click **Add** to create a new web server
3. Configure as follows:

| Field | Value |
|-------|-------|
| **Name** | AI-Voice-App-Server |
| **Host Type** | Web server |
| **IP Address** | 192.168.0.177 |
| **Host** | agent.ghaben.ca |
| **Port** | 443 (HTTPS) |

4. Click **Save**

---

## Step 2: Create DNAT Rule (Port Forwarding)

This rule forwards external traffic to your internal server.

1. Go to **Rules and policies > NAT > DNAT**
2. Click **Add** to create new DNAT rule
3. Configure as follows:

### Basic Settings

| Field | Value |
|-------|-------|
| **Name** | DNAT-AI-Voice-HTTPS |
| **Rule Position** | Top (move to top after creation) |
| **Status** | Enable |

### Networks

| Field | Value |
|-------|-------|
| **Source Networks** | Any |
| **Destination Networks** | 76.71.165.113 (your public IP) |
| **Services** | HTTPS (443) |
| **Translated Services** | HTTPS (443) |

### Destination (Translation)

| Field | Value |
|-------|-------|
| **Type** | Map to internal host |
| **Internal Host** | AI-Voice-App-Server (created in Step 1) |

4. Click **Save**

---

## Step 3: Create HTTP to HTTPS Redirect DNAT Rule (Optional but Recommended)

This ensures all HTTP traffic is redirected to HTTPS.

1. Create another DNAT rule with these settings:

| Field | Value |
|-------|-------|
| **Name** | DNAT-AI-Voice-HTTP-Redirect |
| **Rule Position** | Below the HTTPS rule |
| **Source Networks** | Any |
| **Destination Networks** | 76.71.165.113 |
| **Services** | HTTP (80) |
| **Translated Services** | HTTP (80) |
| **Internal Host** | AI-Voice-App-Server |

2. Click **Save**

---

## Step 4: Create Firewall Rules (Allow Traffic)

1. Go to **Rules and policies > Firewall rules**
2. Click **Add** to create new firewall rule
3. Configure as follows:

### HTTPS Rule

| Field | Value |
|-------|-------|
| **Name** | Allow-AI-Voice-HTTPS |
| **Action** | Allow |
| **Rule Position** | Top |
| **Source Zones** | WAN |
| **Destination Zones** | LAN (or DMZ if server is in DMZ) |
| **Source Networks** | Any |
| **Destination Networks** | 192.168.0.177 (AI Voice server) |
| **Services** | HTTPS (443) |
| **NAT** | Enable (use existing DNAT rule) |

4. Click **Save**

### HTTP Rule (for Let's Encrypt and redirect)

Create another rule with same settings but:
- **Name:** Allow-AI-Voice-HTTP
- **Services:** HTTP (80)

---

## Step 5: Configure Application-Level Gateway (ALG) if Needed

If you experience WebSocket issues (Twilio calls disconnecting), you may need to disable HTTP ALG inspection:

1. Go to **Network > DNS > Configuration**
2. Scroll to **ALG (Application Layer Gateways)**
3. Disable **HTTP ALG** if enabled
4. Click **Apply**

---

## Step 6: Verify Configuration

### Test from External Network

From a device outside your network (e.g., your phone on mobile data):

```bash
# Test HTTPS
curl https://agent.ghaben.ca/health

# Expected response:
# {"status":"healthy","service":"ai-voice-platform","version":"2.0.0","environment":"development"}
```

### Check Sophos Logs

1. Go to **Log viewer > Live logs**
2. Filter by:
   - **Traffic Log** for connection attempts
   - **NAT Log** for DNAT translations

You should see:
- External IPs connecting to 76.71.165.113:443
- Successful DNAT to 192.168.0.177:443

---

## Step 7: Configure Twilio Webhook

Once firewall is configured:

1. Log in to your [Twilio Console](https://console.twilio.com)
2. Go to **Phone Numbers > Manage > Active numbers**
3. Click on your Twilio phone number
4. Scroll to **Voice & Fax** section
5. Configure:

| Field | Value |
|-------|-------|
| **A call comes in** | Webhook |
| **Webhook URL** | https://agent.ghaben.ca/api/incoming-call |
| **HTTP Method** | GET |

6. Click **Save**

---

## Troubleshooting

### Issue: "Connection Refused"

**Check:**
- Server is running: `docker compose ps` on the Linux server
- nginx is listening on port 443: `docker compose logs nginx`
- Firewall rules are at the TOP of the list
- DNAT rules are enabled and at the TOP

### Issue: "SSL Certificate Error"

**Check:**
- Certificates exist: `ls -la nginx/ssl/`
- nginx configuration references correct certificate paths
- Certificates are not expired: `openssl x509 -in nginx/ssl/fullchain.pem -noout -dates`

### Issue: "WebSocket Connection Fails"

**Check:**
- HTTP ALG is disabled in Sophos
- nginx proxy timeouts are sufficient (3600s)
- WebSocket upgrade headers are passed correctly

### Issue: "Let's Encrypt Fails"

**Common causes:**
- Port 80 not open from internet
- DNS not propagated (check: `dig agent.ghaben.ca`)
- Firewall blocking inbound port 80

---

## Security Best Practices

1. **HTTPS Only:** The nginx configuration redirects all HTTP to HTTPS
2. **HSTS Header:** Adds Strict-Transport-Security header
3. **Modern SSL:** Uses TLS 1.2 and 1.3 only
4. **Firewall Rules:** Only allow necessary ports (80, 443)
5. **Rate Limiting:** Consider adding rate limiting for API endpoints (future enhancement)
6. **Fail2Ban:** Consider adding fail2ban for brute force protection (future enhancement)

---

## Next Steps After Firewall Configuration

1. ✅ Test HTTPS access from external network
2. ✅ Verify SSL certificate is valid
3. ✅ Configure Twilio webhook URL
4. ✅ Make a test phone call
5. ✅ Verify call recording and transcription
6. ✅ Monitor logs: `docker compose logs -f app`

---

## Contact

For issues or questions, check the logs:
- Application: `docker compose logs -f app`
- nginx: `docker compose logs -f nginx`
- Firewall: Sophos Log Viewer
