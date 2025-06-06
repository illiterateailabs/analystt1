# Analyst's Augmentation Agent - Backend Dockerfile
# Multi-stage build for a smaller final image

# Build stage
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Add build-time ARG for pip options to allow debugging
ARG PIP_EXTRA_OPTS=""

# Configure pip to increase resolver backtracking limit and use new resolver
RUN pip config set global.use-feature 2020-resolver && \
    pip config set global.no-cache-dir true && \
    pip config set global.timeout 180 && \
    pip config set global.extra-index-url https://pypi.org/simple && \
    pip config set global.retries 10 && \
    echo "[global]\nbacktrack-limit = 5000" > /etc/pip.conf

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and constraints files
COPY requirements.txt constraints.txt ./

# Install Python dependencies with improved strategy
# First upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# First install packages without dependencies to avoid conflicts
RUN pip install --no-cache-dir --no-deps --constraint constraints.txt ${PIP_EXTRA_OPTS} -r requirements.txt

# Then install all dependencies
RUN pip install --no-cache-dir --constraint constraints.txt ${PIP_EXTRA_OPTS} -r requirements.txt

# Final stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN useradd -m -r -u 1000 appuser

# Create logs directory and set permissions
RUN mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser backend /app/backend

# Expose port
EXPOSE 8000

# Switch to non-root user
USER appuser

# Set Python path
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
