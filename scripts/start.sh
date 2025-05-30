#!/bin/bash

# Analyst's Augmentation Agent - Startup Script
# This script starts all services in the correct order

set -e

echo "🚀 Starting Analyst's Augmentation Agent..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if .env file exists
if [ ! -f .env ]; then
    print_warning ".env file not found. Creating from .env.example..."
    cp .env.example .env
    print_warning "Please edit .env file with your API keys before continuing."
    exit 1
fi

# Check if required environment variables are set
source .env

if [ -z "$GOOGLE_API_KEY" ]; then
    print_error "GOOGLE_API_KEY not set in .env file"
    exit 1
fi

if [ -z "$E2B_API_KEY" ]; then
    print_error "E2B_API_KEY not set in .env file"
    exit 1
fi

# Create logs directory
mkdir -p logs

print_status "Starting infrastructure services..."

# Start Docker services
docker-compose up -d neo4j postgres redis

print_status "Waiting for services to be ready..."

# Wait for Neo4j to be ready
print_status "Waiting for Neo4j..."
until docker-compose exec neo4j cypher-shell -u neo4j -p analyst123 "RETURN 1" > /dev/null 2>&1; do
    sleep 2
    echo -n "."
done
print_success "Neo4j is ready"

# Wait for PostgreSQL to be ready
print_status "Waiting for PostgreSQL..."
until docker-compose exec postgres pg_isready -U analyst > /dev/null 2>&1; do
    sleep 2
    echo -n "."
done
print_success "PostgreSQL is ready"

# Initialize Neo4j schema if needed
print_status "Initializing Neo4j schema..."
docker-compose exec neo4j cypher-shell -u neo4j -p analyst123 -f /docker-entrypoint-initdb.d/001-schema.cypher || true
print_success "Neo4j schema initialized"

# Install Python dependencies
print_status "Installing Python dependencies..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt
print_success "Python dependencies installed"

# Install frontend dependencies
print_status "Installing frontend dependencies..."
cd frontend
npm install
cd ..
print_success "Frontend dependencies installed"

# Start backend
print_status "Starting backend server..."
source venv/bin/activate
nohup python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > logs/backend.pid

# Wait for backend to be ready
print_status "Waiting for backend to be ready..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
    echo -n "."
done
print_success "Backend is ready"

# Start frontend
print_status "Starting frontend server..."
cd frontend
nohup npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../logs/frontend.pid
cd ..

# Wait for frontend to be ready
print_status "Waiting for frontend to be ready..."
until curl -s http://localhost:3000 > /dev/null 2>&1; do
    sleep 2
    echo -n "."
done
print_success "Frontend is ready"

print_success "🎉 All services are running!"
echo ""
echo "📊 Service URLs:"
echo "  • Frontend:  http://localhost:3000"
echo "  • Backend:   http://localhost:8000"
echo "  • API Docs:  http://localhost:8000/docs"
echo "  • Neo4j:     http://localhost:7474 (neo4j/analyst123)"
echo ""
echo "📝 Logs:"
echo "  • Backend:   tail -f logs/backend.log"
echo "  • Frontend:  tail -f logs/frontend.log"
echo ""
echo "🛑 To stop all services: ./scripts/stop.sh"
echo ""

# Keep script running and show logs
print_status "Showing live logs (Ctrl+C to exit)..."
tail -f logs/backend.log logs/frontend.log
