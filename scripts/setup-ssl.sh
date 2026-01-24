#!/bin/bash
# =====================================================
# AI Voice Platform - SSL Certificate Setup
# =====================================================
# Simplified script to generate Let's Encrypt SSL certificates

set -e

DOMAIN="agent.ghaben.ca"
EMAIL="admin@iflextax.ca"  # Change to your email

echo "=========================================="
echo "SSL Certificate Setup for $DOMAIN"
echo "=========================================="
echo ""

# Create necessary directories
mkdir -p nginx/ssl nginx/www nginx/logs certbot/conf

echo "Step 1: Checking DNS resolution..."
if host "$DOMAIN" > /dev/null 2>&1; then
    echo "✓ DNS resolves for $DOMAIN"
else
    echo "✗ ERROR: DNS does not resolve for $DOMAIN"
    echo "  Please ensure agent.ghaben.ca points to 76.71.165.113"
    exit 1
fi

echo ""
echo "Step 2: Starting temporary nginx on port 80..."
# Create a simple test nginx config for HTTP-only (for certificate generation)
cat > nginx/nginx-temp.conf <<'EOF'
events {
    worker_connections 1024;
}

http {
    server {
        listen 80;
        server_name agent.ghaben.ca;

        # Serve certbot challenges
        location /.well-known/acme-challenge/ {
            root /var/www/certbot;
        }

        # Temporarily serve health check
        location /health {
            return 200 '{"status":"temporary"}';
            add_header Content-Type application/json;
        }
    }
}
EOF

# Start temporary nginx container for HTTP-only
docker run -d --name nginx-temp \
    -v $(pwd)/nginx/nginx-temp.conf:/etc/nginx/nginx.conf:ro \
    -v $(pwd)/nginx/www:/var/www/certbot:ro \
    -p 80:80 \
    nginx:alpine

sleep 3

echo ""
echo "Step 3: Testing HTTP access from internet..."
# Test if we can reach the server
curl -s http://localhost/health > /dev/null
if [ $? -eq 0 ]; then
    echo "✓ Server is accessible on port 80"
else
    echo "✗ ERROR: Server not accessible on port 80"
    docker stop nginx-temp && docker rm nginx-temp
    exit 1
fi

echo ""
echo "Step 4: Generating SSL certificate with Let's Encrypt..."
echo "  This requires port 80 to be open from the internet!"
echo ""

# Generate certificate
docker run --rm \
    -v $(pwd)/certbot/conf:/etc/letsencrypt \
    -v $(pwd)/nginx/www:/var/www/certbot \
    certbot/certbot:latest \
    certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN \
    --force-renewal

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Certificate generated successfully!"

    # Copy certificates to nginx ssl directory
    cp certbot/conf/live/$DOMAIN/fullchain.pem nginx/ssl/fullchain.pem
    cp certbot/conf/live/$DOMAIN/privkey.pem nginx/ssl/privkey.pem

    chmod 644 nginx/ssl/fullchain.pem
    chmod 600 nginx/ssl/privkey.pem

    echo "✓ Certificates copied to nginx/ssl/"
else
    echo ""
    echo "✗ Certificate generation failed"
    echo ""
    echo "Common issues:"
    echo "  1. Port 80 not open from internet"
    echo "  2. DNS not pointing to correct IP"
    echo "  3. Firewall blocking inbound port 80"
    docker stop nginx-temp && docker rm nginx-temp
    exit 1
fi

# Clean up temporary nginx
echo ""
echo "Step 5: Cleaning up..."
docker stop nginx-temp
docker rm nginx-temp
rm nginx/nginx-temp.conf

echo "✓ Temporary nginx removed"
echo ""
echo "=========================================="
echo "✓ SSL Setup Complete!"
echo "=========================================="
echo ""
echo "Certificates saved to:"
echo "  - nginx/ssl/fullchain.pem"
echo "  - nginx/ssl/privkey.pem"
echo ""
echo "Next steps:"
echo "  1. Configure Sophos firewall (see docs/SOPHOS-FIREWALL-SETUP.md)"
echo "  2. Start the full stack: docker compose up -d"
echo "  3. Test: curl https://$DOMAIN/health"
echo ""
