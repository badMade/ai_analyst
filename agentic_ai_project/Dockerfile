# Agentic AI Project Dockerfile
# Multi-stage build for optimized image size

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim as runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY config/ ./config/
COPY src/ ./src/
COPY examples/ ./examples/
COPY data/ ./data/

# Create directories for runtime data
RUN mkdir -p data/logs data/checkpoints data/memory data/knowledge_base data/training

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "examples.single_agent"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.agents import BaseAgent; print('OK')" || exit 1

# Labels
LABEL maintainer="Brij Kishore Pandey"
LABEL version="0.1.0"
LABEL description="Agentic AI Project - Intelligent Autonomous Agents Framework"
