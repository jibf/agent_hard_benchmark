#!/bin/bash

# Multi-turn Task Corrected Analysis - Server Execution Script
# Usage: ./run_multi_turn_corrected_analysis.sh [workers] [task]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default configuration
WORKERS=${1:-32}
TASK=${2:-all}

echo "========================================================="
echo "Multi-turn Task Corrected Analysis - Server Mode"
echo "========================================================="
echo "Start time: $(date)"
echo "Script directory: $SCRIPT_DIR"
echo "Workers: $WORKERS" 
echo "Task: $TASK"
echo ""

# Check if required files exist
echo "Checking required files..."
required_files=(".env" "enhanced_functionality_analyzer.py" "multi_turn_corrected_analysis.py")
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
if command -v nproc >/dev/null 2>&1; then
    echo "CPU cores: $(nproc)"
else
    echo "CPU cores: $(sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')"
fi

if command -v free >/dev/null 2>&1; then
    echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')"
else
    echo "Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024) "GB"}' || echo 'unknown')"
fi

echo "Disk space: $(df -h . | awk 'NR==2 {print $4}')"

# Create output directory
mkdir -p score

# Check for previous runs
if [[ -f "multi_turn_analysis.pid" ]]; then
    echo ""
    echo "WARNING: Found existing PID file. Checking if previous run is still active..."
    if kill -0 "$(cat multi_turn_analysis.pid)" 2>/dev/null; then
        echo "ERROR: Another analysis is already running (PID: $(cat multi_turn_analysis.pid))"
        echo "Wait for it to finish or kill it manually"
        exit 1
    else
        echo "Previous run appears to be finished, removing stale PID file"
        rm -f multi_turn_analysis.pid
    fi
fi

# Setup log file
LOG_FILE="multi_turn_corrected_analysis_$(date +%Y%m%d_%H%M%S).log"
echo ""
echo "Log file: $LOG_FILE"

# Build command
CMD="python3 multi_turn_corrected_analysis.py --workers $WORKERS --task $TASK"

echo ""
echo "Starting multi-turn corrected analysis..."
echo "Command: $CMD"
echo ""
echo "========================================================="

# Save PID for monitoring
echo $$ > multi_turn_analysis.pid

# Run the analysis
exec $CMD 2>&1 | tee "$LOG_FILE"

# Clean up PID file on exit
trap 'rm -f multi_turn_analysis.pid' EXIT