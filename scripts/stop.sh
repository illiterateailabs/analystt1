#!/bin/bash

# Analyst's Augmentation Agent - Stop Script
# This script stops all running services

set -e

echo "🛑 Stopping Analyst's Augmentation Agent..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Stop frontend
if [ -f logs/frontend.pid ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    print_status "Stopping frontend (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID 2>/dev/null || true
    rm -f logs/frontend.pid
    print_success "Frontend stopped"
else
    print_warning "Frontend PID file not found"
fi

# Stop backend
if [ -f logs/backend.pid ]; then
    BACKEND_PID=$(cat logs/backend.pid)
    print_status "Stopping backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID 2>/dev/null || true
    rm -f logs/backend.pid
    print_success "Backend stopped"
else
    print_warning "Backend PID file not found"
fi

# Stop Docker services
print_status "Stopping Docker services..."
docker-compose down
print_success "Docker services stopped"

# Kill any remaining processes
print_status "Cleaning up remaining processes..."
pkill -f "uvicorn backend.main:app" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true

print_success "🏁 All services stopped successfully!"
