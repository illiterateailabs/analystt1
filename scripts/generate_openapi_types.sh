#!/bin/bash
# ============================================================================
# OpenAPI TypeScript Types Generator
# ============================================================================
# This script generates TypeScript types from the OpenAPI specification
# for use in the frontend. It works in both local development and CI environments.
#
# Usage:
#   ./scripts/generate_openapi_types.sh [--ci] [--url URL]
#
# Options:
#   --ci        Run in CI mode (no interactive prompts)
#   --url URL   Specify the OpenAPI spec URL (default: http://localhost:8000/api/v1/openapi.json)
#   --help      Show this help message
#
# Examples:
#   ./scripts/generate_openapi_types.sh
#   ./scripts/generate_openapi_types.sh --ci --url http://backend:8000/api/v1/openapi.json
# ============================================================================

set -e

# Default values
CI_MODE=false
OPENAPI_URL="http://localhost:8000/api/v1/openapi.json"
OUTPUT_FILE="frontend/src/lib/types/api.ts"
MAX_RETRIES=5
RETRY_DELAY=2

# Text formatting
BOLD="\033[1m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RESET="\033[0m"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --ci) CI_MODE=true ;;
    --url) OPENAPI_URL="$2"; shift ;;
    --help)
      echo -e "${BOLD}OpenAPI TypeScript Types Generator${RESET}"
      echo "Usage: $0 [--ci] [--url URL]"
      echo ""
      echo "Options:"
      echo "  --ci        Run in CI mode (no interactive prompts)"
      echo "  --url URL   Specify the OpenAPI spec URL (default: http://localhost:8000/api/v1/openapi.json)"
      echo "  --help      Show this help message"
      exit 0
      ;;
    *) echo "Unknown parameter: $1"; exit 1 ;;
  esac
  shift
done

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Check if npx is available
if ! command -v npx &> /dev/null; then
  echo -e "${RED}Error: npx is not installed. Please install Node.js and npm.${RESET}"
  exit 1
fi

# Check if curl is available
if ! command -v curl &> /dev/null; then
  echo -e "${RED}Error: curl is not installed. Please install curl.${RESET}"
  exit 1
fi

echo -e "${BLUE}${BOLD}Generating TypeScript types from OpenAPI spec${RESET}"
echo -e "OpenAPI URL: ${YELLOW}$OPENAPI_URL${RESET}"
echo -e "Output file: ${YELLOW}$OUTPUT_FILE${RESET}"

# Check if backend is accessible
echo -e "\n${BLUE}Checking if backend is accessible...${RESET}"
for i in $(seq 1 $MAX_RETRIES); do
  if curl --silent --fail --output /dev/null "$OPENAPI_URL"; then
    echo -e "${GREEN}✓ Backend is accessible${RESET}"
    break
  else
    if [ $i -eq $MAX_RETRIES ]; then
      echo -e "${RED}✗ Backend is not accessible after $MAX_RETRIES attempts${RESET}"
      if [ "$CI_MODE" = false ]; then
        read -p "Do you want to continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
          echo -e "${RED}Aborting.${RESET}"
          exit 1
        fi
      else
        echo -e "${RED}Aborting in CI mode.${RESET}"
        exit 1
      fi
    else
      echo -e "${YELLOW}⚠ Backend not accessible, retrying in ${RETRY_DELAY}s (attempt $i/$MAX_RETRIES)...${RESET}"
      sleep $RETRY_DELAY
    fi
  fi
done

# Generate TypeScript types
echo -e "\n${BLUE}Generating TypeScript types...${RESET}"
if npx openapi-typescript "$OPENAPI_URL" --output "$OUTPUT_FILE"; then
  echo -e "${GREEN}✓ TypeScript types generated successfully${RESET}"
else
  echo -e "${RED}✗ Failed to generate TypeScript types${RESET}"
  exit 1
fi

# Add header to the generated file
TEMP_FILE=$(mktemp)
cat > "$TEMP_FILE" << EOL
/**
 * Auto-generated API types from OpenAPI specification
 * Generated on: $(date)
 * Do not modify this file directly, run 'npm run generate-api-types' instead
 */

EOL
cat "$OUTPUT_FILE" >> "$TEMP_FILE"
mv "$TEMP_FILE" "$OUTPUT_FILE"

# Validate the generated file
echo -e "\n${BLUE}Validating generated types...${RESET}"
if [ ! -s "$OUTPUT_FILE" ]; then
  echo -e "${RED}✗ Generated file is empty${RESET}"
  exit 1
fi

if ! grep -q "export interface" "$OUTPUT_FILE"; then
  echo -e "${RED}✗ Generated file does not contain TypeScript interfaces${RESET}"
  exit 1
fi

echo -e "${GREEN}✓ Validation successful${RESET}"

# Add package.json script if not in CI mode
if [ "$CI_MODE" = false ]; then
  if [ -f "frontend/package.json" ]; then
    echo -e "\n${BLUE}Checking if package.json script exists...${RESET}"
    if ! grep -q '"generate-api-types"' "frontend/package.json"; then
      echo -e "${YELLOW}⚠ 'generate-api-types' script not found in package.json${RESET}"
      echo -e "${YELLOW}Add this to your package.json scripts:${RESET}"
      echo -e "${YELLOW}  \"generate-api-types\": \"../scripts/generate_openapi_types.sh\"${RESET}"
    else
      echo -e "${GREEN}✓ 'generate-api-types' script already exists in package.json${RESET}"
    fi
  fi
fi

echo -e "\n${GREEN}${BOLD}TypeScript types generated successfully!${RESET}"
echo -e "You can now import types from ${YELLOW}'@/lib/types/api'${RESET} in your frontend code."
echo -e "Example usage:"
echo -e "${YELLOW}import type { paths } from '@/lib/types/api';${RESET}"
echo -e "${YELLOW}// Use with React Query${RESET}"
echo -e "${YELLOW}const { data } = useQuery<paths['/api/v1/health']['get']['responses']['200']['content']['application/json']>('/api/v1/health');${RESET}"

exit 0
