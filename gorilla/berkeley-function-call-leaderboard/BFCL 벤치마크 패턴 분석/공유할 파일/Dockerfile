# Enhanced BFCL Analysis Pipeline Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p /app/output

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash bfcl_user && \
    chown -R bfcl_user:bfcl_user /app
USER bfcl_user

# Default command
ENTRYPOINT ["python", "bfcl_analysis_enhanced.py"]
CMD ["--help"]

# Metadata
LABEL org.opencontainers.image.title="Enhanced BFCL Analysis Pipeline"
LABEL org.opencontainers.image.description="Comprehensive statistical analysis of BFCL benchmark with accessibility features"
LABEL org.opencontainers.image.version="2.0"
LABEL org.opencontainers.image.authors="BFCL Analysis Team"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; print('Container is healthy'); sys.exit(0)"