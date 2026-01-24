#!/bin/bash
# =====================================================
# AI Voice Platform - SSL Certificate Setup
# =====================================================
# This script generates Let's Encrypt SSL certificates

set -e

DOMAIN="agent.ghaben.ca"
EMAIL="admin@iflextax.ca"  # Change to your email
STAGING="${STAGING:-false}"  # Set STAGING=true for testing

echo "=========================================="
echo "SSL Certificate Setup for $DOMAIN"
echo "=========================================="

# Create necessary directories
mkdir -p nginx/ssl
mkdir -p nginx/www

# Check if certificate already exists
if [ -f "nginx/ssl/fullchain.pem" ] && [ -f "nginx/ssl/privkey.pem" ]; then
    echo "✓ SSL certificates already exist"
    echo "  To renew: docker compose run --rm certbot renew"
    exit 0
fi

# Use staging for testing (avoid rate limits)
if [ "$STAGING" = "true" ]; then
    echo ""
    echo "⚠️  USING STAGING ENVIRONMENT (for testing)"
    echo "    Set STAGING=false in environment to use production"
    STAGING_FLAG="--staging"
else
    echo ""
    echo "✓ Using PRODUCTION Let's Encrypt"
    STAGING_FLAG=""
fi

echo ""
echo "Generating certificate for: $DOMAIN"
echo "Email: $EMAIL"
echo ""

# Generate certificate using certbot
docker compose run --rm \
    -e ACME_AGREE=true \
    certbot \
    certonly \
    --webroot \
    --webroot-path=/var/www/certbot \
    --email $EMAIL \
    --agree-tos \
    --no-eff-email \
    -d $DOMAIN \
    $STAGING_FLAG

# Copy certificates to nginx ssl directory
docker compose exec -T app bash -c "cat /etc/letsencrypt/live/$DOMAIN/fullchain.pem" > nginx/ssl/fullchain.pem
docker compose exec -T app bash -c "cat /etc/letsencrypt/live/$DOMAIN/privkey.pem" > nginx/ssl/privkey.pem

# Set proper permissions
chmod 644 nginx/ssl/fullchain.pem
chmod 600 nginx/ssl/privkey.pem

echo ""
echo "✓ SSL certificates generated successfully!"
echo "  Location: nginx/ssl/"
echo ""
echo "Next steps:"
echo "  1. Update docker-compose.yml to enable nginx (remove 'production' profile)"
echo "  2. Restart: docker compose up -d"
echo ""
