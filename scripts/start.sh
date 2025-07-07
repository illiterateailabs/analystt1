#!/bin/bash
# =============================================================================
# Analyst Droid Development Environment Starter
# =============================================================================
# This script starts both the backend and frontend development servers
# concurrently with hot reload enabled. It handles proper cleanup on exit
# and provides status information.
#
# Usage: ./scripts/start.sh
# =============================================================================

# Exit on error
set -e

# Colors for pretty output
BLUE="\033[1;34m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
RED="\033[1;31m"
RESET="\033[0m"

# Configuration
BACKEND_PORT=8000
FRONTEND_PORT=3000
BACKEND_URL="http://localhost:${BACKEND_PORT}"
FRONTEND_URL="http://localhost:${FRONTEND_PORT}"
API_DOCS_URL="${BACKEND_URL}/api/docs"
LOGS_DIR="logs"
BACKEND_LOG="${LOGS_DIR}/backend.log"
FRONTEND_LOG="${LOGS_DIR}/frontend.log"

# Create logs directory if it doesn't exist
mkdir -p "${LOGS_DIR}"

# Store PIDs for cleanup
BACKEND_PID=""
FRONTEND_PID=""

# Function to cleanup processes on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down development servers...${RESET}"
    
    # Kill backend process if running
    if [ -n "${BACKEND_PID}" ] && ps -p "${BACKEND_PID}" > /dev/null; then
        echo -e "${BLUE}Stopping backend server (PID: ${BACKEND_PID})...${RESET}"
        kill -TERM "${BACKEND_PID}" 2>/dev/null || true
    fi
    
    # Kill frontend process if running
    if [ -n "${FRONTEND_PID}" ] && ps -p "${FRONTEND_PID}" > /dev/null; then
        echo -e "${BLUE}Stopping frontend server (PID: ${FRONTEND_PID})...${RESET}"
        kill -TERM "${FRONTEND_PID}" 2>/dev/null || true
    fi
    
    echo -e "${GREEN}Development environment shutdown complete.${RESET}"
    exit 0
}

# Register cleanup function for various signals
trap cleanup EXIT INT TERM

# Function to check if a service is ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1
    
    echo -e "${BLUE}Waiting for ${name} to be ready...${RESET}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f -o /dev/null "$url"; then
            echo -e "${GREEN}✓ ${name} is ready!${RESET}"
            return 0
        fi
        
        echo -n "."
        sleep 1
        attempt=$((attempt + 1))
    done
    
    echo -e "\n${RED}✗ ${name} failed to start within the expected time.${RESET}"
    return 1
}

# Print banner
echo -e "${BLUE}=============================================================${RESET}"
echo -e "${BLUE}           Analyst Droid Development Environment            ${RESET}"
echo -e "${BLUE}=============================================================${RESET}"
echo -e "${YELLOW}Starting backend and frontend with hot reload...${RESET}\n"

# Start backend server
echo -e "${BLUE}Starting backend server...${RESET}"
uvicorn backend.main:app --reload --host 0.0.0.0 --port "${BACKEND_PORT}" > "${BACKEND_LOG}" 2>&1 &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend process started with PID: ${BACKEND_PID}${RESET}"

# Start frontend server
echo -e "${BLUE}Starting frontend server...${RESET}"
(cd frontend && npm run dev) > "${FRONTEND_LOG}" 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend process started with PID: ${FRONTEND_PID}${RESET}"

# Wait for services to be ready
wait_for_service "${BACKEND_URL}/health" "Backend" || cleanup
wait_for_service "${FRONTEND_URL}" "Frontend" || cleanup

# Print access information
echo -e "\n${GREEN}Development environment is ready!${RESET}"
echo -e "${YELLOW}Backend:  ${BACKEND_URL}${RESET}"
echo -e "${YELLOW}Frontend: ${FRONTEND_URL}${RESET}"
echo -e "${YELLOW}API Docs: ${API_DOCS_URL}${RESET}"
echo -e "\n${BLUE}Logs:${RESET}"
echo -e "${YELLOW}Backend:  tail -f ${BACKEND_LOG}${RESET}"
echo -e "${YELLOW}Frontend: tail -f ${FRONTEND_LOG}${RESET}"
echo -e "\n${BLUE}Press Ctrl+C to stop all services${RESET}"

# Keep script running until user interrupts
while true; do
    # Check if processes are still running
    if ! ps -p "${BACKEND_PID}" > /dev/null; then
        echo -e "${RED}Backend process exited unexpectedly. Check ${BACKEND_LOG} for details.${RESET}"
        exit 1
    fi
    
    if ! ps -p "${FRONTEND_PID}" > /dev/null; then
        echo -e "${RED}Frontend process exited unexpectedly. Check ${FRONTEND_LOG} for details.${RESET}"
        exit 1
    fi
    
    sleep 2
done
