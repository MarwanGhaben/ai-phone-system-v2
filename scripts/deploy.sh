#!/bin/bash
# =====================================================
# AI Voice Platform v2 - Deployment Script
# =====================================================
# Deploy the platform to any server with one command

set -e

echo "======================================================"
echo "  AI Voice Platform v2 - Deployment"
echo "======================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${RED}ERROR: .env file not found!${NC}"
    echo "Please copy .env.example to .env and fill in your API keys:"
    echo ""
    echo "  cp .env.example .env"
    echo "  nano .env  # or your favorite editor"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} Configuration file found (.env)"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: Docker is not installed!${NC}"
    echo "Install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker is installed"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}ERROR: Docker Compose is not installed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker Compose is installed"

echo ""
echo "======================================================"
echo "  Starting deployment..."
echo "======================================================"
echo ""

# Stop any existing containers
echo -e "${YELLOW}→${NC} Stopping any existing containers..."
docker-compose down 2>/dev/null || true

# Build and start services
echo -e "${YELLOW}→${NC} Building Docker images..."
docker-compose build

echo ""
echo -e "${YELLOW}→${NC} Starting services..."
docker-compose up -d

echo ""
echo "======================================================"
echo -e "${GREEN}  ✓ Deployment complete!${NC}"
echo "======================================================"
echo ""
echo "The platform is now running!"
echo ""
echo "Services:"
echo "  - Main API:     http://localhost:8000"
echo "  - WebSocket:    ws://localhost:8001"
echo "  - Health check: http://localhost:8000/health"
echo "  - Database:     localhost:5432"
echo "  - Redis:        localhost:6379"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f app"
echo ""
echo "To stop:"
echo "  docker-compose down"
echo ""
echo "To restart:"
echo "  docker-compose restart app"
echo ""
