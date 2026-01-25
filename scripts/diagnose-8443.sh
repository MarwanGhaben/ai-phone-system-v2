#!/bin/bash
# Diagnostic script for port 8443 connectivity issue

echo "=================================="
echo "Port 8443 Diagnostic Script"
echo "=================================="
echo ""

echo "1. Docker Container Status:"
docker compose ps
echo ""

echo "2. What's listening on port 8443:"
ss -tlnp | grep 8443 || echo "Nothing found on 8443!"
echo ""

echo "3. Test localhost:8443 (should work):"
curl -v -k https://localhost:8443/health 2>&1 | head -20
echo ""

echo "4. Test 127.0.0.1:8443:"
curl -v -k https://127.0.0.1:8443/health 2>&1 | head -20
echo ""

echo "5. Test internal IP 192.168.0.177:8443:"
curl -v -k https://192.168.0.177:8443/health 2>&1 | head -20
echo ""

echo "6. Docker network inspection:"
docker network inspect ai-voice-platform-v2_ai-voice-network 2>/dev/null | grep -A 5 "IPv4Address" || echo "Network not found"
echo ""

echo "7. nginx container test:"
docker compose exec nginx nginx -t
echo ""

echo "8. nginx processes in container:"
docker compose exec nginx ps aux | grep nginx
echo ""

echo "9. nginx error logs (last 20 lines):"
docker compose logs nginx --tail 20 2>&1 | grep -i error || echo "No errors found"
echo ""

echo "10. Firewall rules (iptables):"
iptables -t nat -L -n -v | grep 8443 || echo "No NAT rules for 8443"
echo ""
iptables -L INPUT -n -v | grep 8443 || echo "No INPUT rules for 8443"
echo ""

echo "=================================="
echo "Diagnostic complete!"
echo "=================================="
