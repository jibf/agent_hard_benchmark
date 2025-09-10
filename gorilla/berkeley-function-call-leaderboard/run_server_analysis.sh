#!/bin/bash

# Enhanced BFCL Functionality Analysis - Server Deployment Script
# Usage: ./run_server_analysis.sh [workers] [resume_task]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default configuration
WORKERS=${1:-32}
RESUME_TASK=${2:-""}
CONFIG_FILE="deployment_config.json"

echo "=================================================="
echo "Enhanced BFCL Functionality Analysis - Server Mode"
echo "=================================================="
echo "Start time: $(date)"
echo "Script directory: $SCRIPT_DIR"
echo "Workers: $WORKERS"
echo "Resume task: ${RESUME_TASK:-"(none - full analysis)"}"
echo "Config: $CONFIG_FILE"
echo ""

# Check if required files exist
echo "Checking required files..."
required_files=(".env" "enhanced_functionality_analyzer.py" "server_deployment.py" "$CONFIG_FILE")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "ERROR: Required file not found: $file"
        exit 1
    fi
    echo "✓ Found: $file"
done

# Check Python installation
echo ""
echo "Checking Python environment..."
python3 --version || { echo "ERROR: Python 3 not found"; exit 1; }

# Install requirements if needed
if [[ -f "requirements.txt" ]]; then
    echo "Installing/checking Python requirements..."
    python3 -m pip install -r requirements.txt --quiet
    echo "✓ Requirements satisfied"
fi

# Check API configuration
echo ""
echo "Checking API configuration..."
if [[ ! -f ".env" ]]; then
    echo "ERROR: .env file not found"
    exit 1
fi

# Check if API_KEY and BASE_URL are set
if ! grep -q "API_KEY=" .env || ! grep -q "BASE_URL=" .env; then
    echo "ERROR: API_KEY or BASE_URL not found in .env file"
    exit 1
fi
echo "✓ API configuration found"

# Check system resources
echo ""
echo "System Resources:"
echo "CPU cores: $(nproc)"
echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Disk space: $(df -h . | awk 'NR==2 {print $4}')"

# Create output directory
mkdir -p score/backups

# Check for previous runs
if [[ -f "bfcl_analysis.pid" ]]; then
    echo ""
    echo "WARNING: Found existing PID file. Checking if previous run is still active..."
    if kill -0 "$(cat bfcl_analysis.pid)" 2>/dev/null; then
        echo "ERROR: Another analysis is already running (PID: $(cat bfcl_analysis.pid))"
        echo "Wait for it to finish or kill it manually"
        exit 1
    else
        echo "Previous run appears to be finished, removing stale PID file"
        rm -f bfcl_analysis.pid
    fi
fi

# Setup log file
LOG_FILE="bfcl_analysis_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "Log file: $LOG_FILE"

# Build command
CMD="python3 server_deployment.py --config $CONFIG_FILE --workers $WORKERS"
if [[ -n "$RESUME_TASK" ]]; then
    CMD="$CMD --resume $RESUME_TASK"
fi

echo ""
echo "Starting analysis..."
echo "Command: $CMD"
echo ""
echo "=================================================="

# Run the analysis
exec $CMD 2>&1 | tee "$LOG_FILE"