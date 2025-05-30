#!/bin/bash

# Analyst's Augmentation Agent - Setup Script
# This script sets up the development environment

set -e

echo "🔧 Setting up Analyst's Augmentation Agent development environment..."

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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
print_status "Checking system requirements..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ $(echo "$PYTHON_VERSION < 3.9" | bc -l) -eq 1 ]]; then
    print_error "Python 3.9+ is required (found $PYTHON_VERSION)"
    exit 1
fi
print_success "Python $PYTHON_VERSION found"

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is required but not installed"
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [[ $NODE_VERSION -lt 18 ]]; then
    print_error "Node.js 18+ is required (found v$NODE_VERSION)"
    exit 1
fi
print_success "Node.js v$(node --version) found"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is required but not installed"
    exit 1
fi
print_success "Docker found"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is required but not installed"
    exit 1
fi
print_success "Docker Compose found"

# Create environment file
print_status "Setting up environment configuration..."
if [ ! -f .env ]; then
    cp .env.example .env
    print_success "Created .env file from template"
    print_warning "Please edit .env file with your API keys:"
    print_warning "  - GOOGLE_API_KEY (required for Gemini)"
    print_warning "  - E2B_API_KEY (required for code execution)"
    print_warning "  - NEO4J_PASSWORD (set to 'analyst123' or change in docker-compose.yml)"
else
    print_warning ".env file already exists"
fi

# Create directories
print_status "Creating required directories..."
mkdir -p logs
mkdir -p data
mkdir -p uploads
mkdir -p exports
print_success "Directories created"

# Make scripts executable
print_status "Making scripts executable..."
chmod +x scripts/*.sh
print_success "Scripts are now executable"

# Setup Python virtual environment
print_status "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment and install dependencies
print_status "Installing Python dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
print_success "Python dependencies installed"

# Install frontend dependencies
print_status "Installing frontend dependencies..."
cd frontend
npm install
cd ..
print_success "Frontend dependencies installed"

# Setup pre-commit hooks (optional)
print_status "Setting up development tools..."
source venv/bin/activate
pip install pre-commit black isort flake8 mypy
print_success "Development tools installed"

# Pull Docker images
print_status "Pulling Docker images..."
docker-compose pull
print_success "Docker images pulled"

print_success "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "  1. Edit .env file with your API keys"
echo "  2. Run: ./scripts/start.sh"
echo "  3. Open http://localhost:3000 in your browser"
echo ""
echo "📚 Useful commands:"
echo "  • Start services:  ./scripts/start.sh"
echo "  • Stop services:   ./scripts/stop.sh"
echo "  • View logs:       tail -f logs/backend.log"
echo "  • Run tests:       pytest"
echo ""
echo "🔑 Required API Keys:"
echo "  • Google Gemini:   https://makersuite.google.com/app/apikey"
echo "  • e2b.dev:         https://e2b.dev/docs"
echo ""
