# Enhanced BFCL Functionality Analyzer - Server Deployment Guide

## ✅ Local Testing Completed Successfully

All tests passed:
- Import Test: ✓ PASSED
- Configuration Test: ✓ PASSED  
- Data Loading Test: ✓ PASSED
- Prompt Generation Test: ✓ PASSED
- API Call Test: ✓ PASSED
- Single Case Analysis Test: ✓ PASSED

## 🚀 Server Deployment Instructions

### 1. Prerequisites

#### System Requirements
- **CPU**: 8+ cores recommended (supports up to 32 parallel workers)
- **Memory**: 8GB+ RAM (16GB recommended for optimal performance)
- **Storage**: 2GB+ free space for results and logs
- **OS**: Linux/Unix (Ubuntu, CentOS, etc.) or macOS
- **Python**: 3.8+ 

#### Required Software
```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip git

# Install system dependencies
sudo apt install build-essential python3-dev
```

### 2. Repository Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard

# Switch to analysis branch
git checkout ai-analysis-functionality

# Verify files are present
ls -la enhanced_functionality_analyzer.py server_deployment.py
```

### 3. Environment Configuration

#### Create .env file
```bash
cat > .env << EOF
API_KEY=sk-sgl-MH7bEVVJlBp3RT_P5cPQ6-KfC1qJElBRCfTDHy40Ue4
BASE_URL=http://5.78.122.79:10000/v1/
EOF
```

#### Install Python dependencies
```bash
# Install requirements
pip3 install -r requirements.txt

# Verify installation
python3 -c "import requests, psutil, concurrent.futures; print('Dependencies OK')"
```

### 4. Pre-deployment Testing

```bash
# Run comprehensive tests
python3 test_enhanced_analyzer.py

# Should show:
# SUCCESS: ALL TESTS PASSED - Ready for deployment!
```

### 5. Server Deployment Options

#### Option A: Simple Deployment (Recommended)
```bash
# Make script executable
chmod +x run_server_analysis.sh

# Run with default settings (32 workers)
./run_server_analysis.sh

# Or specify worker count
./run_server_analysis.sh 16

# Or resume from specific task
./run_server_analysis.sh 32 live_multiple
```

#### Option B: Python Direct Execution
```bash
# Run with default configuration
python3 server_deployment.py

# Custom worker count
python3 server_deployment.py --workers 16

# Resume from specific task
python3 server_deployment.py --resume live_multiple

# Check configuration without running
python3 server_deployment.py --dry-run
```

#### Option C: Background Execution
```bash
# Run in background with nohup
nohup ./run_server_analysis.sh 32 > analysis_output.log 2>&1 &

# Check process
ps aux | grep server_deployment

# Monitor progress
tail -f analysis_output.log
```

### 6. Monitoring and Management

#### Check System Resources
```bash
# CPU and memory usage
htop

# Disk usage
df -h .

# Network activity (if needed)
netstat -i
```

#### Monitor Analysis Progress
```bash
# Check log files
tail -f bfcl_analysis_*.log

# Check result files
ls -la score/enhanced_functionality_analysis_*.json

# Check for running processes
ps aux | grep enhanced_functionality_analyzer
```

#### Graceful Shutdown
```bash
# Send SIGTERM for graceful shutdown
kill -TERM $(cat bfcl_analysis.pid)

# Or use SIGINT (Ctrl+C equivalent)
kill -INT $(cat bfcl_analysis.pid)
```

### 7. Configuration Customization

#### Edit deployment_config.json
```json
{
  "num_workers": 32,          # Adjust based on server capacity
  "api_rate_limit": 60,       # API calls per minute
  "batch_size": 50,           # Cases per batch
  "backup_interval": 300,     # Backup every 5 minutes
  "target_tasks": [           # Tasks to analyze
    "live_multiple",
    "multi_turn_long_context",
    "multi_turn_miss_func",
    "live_irrelevance",
    "multi_turn_miss_param", 
    "multi_turn_base",
    "irrelevance",
    "live_simple"
  ]
}
```

### 8. Expected Output and Timeline

#### File Structure After Completion
```
score/
├── enhanced_functionality_analysis_live_multiple.json
├── enhanced_functionality_analysis_multi_turn_long_context.json
├── enhanced_functionality_analysis_multi_turn_miss_func.json
├── enhanced_functionality_analysis_live_irrelevance.json
├── enhanced_functionality_analysis_multi_turn_miss_param.json
├── enhanced_functionality_analysis_multi_turn_base.json
├── enhanced_functionality_analysis_irrelevance.json
├── enhanced_functionality_analysis_live_simple.json
├── enhanced_functionality_analysis_all_tasks.json
└── backups/
    └── 20241210_143022/  # Timestamped backups
```

#### Performance Expectations
- **Total Cases**: 3,233 cases across 8 task types
- **With 32 workers**: ~15-20 minutes completion time
- **With 16 workers**: ~30-40 minutes completion time
- **Memory Usage**: ~4-8GB peak usage
- **Storage**: ~500MB for all results

### 9. Troubleshooting

#### Common Issues

**API Connection Problems**
```bash
# Test API connectivity
curl -H "Authorization: Bearer $API_KEY" "$BASE_URL/models"
```

**Memory Issues**
```bash
# Reduce worker count
python3 server_deployment.py --workers 8

# Monitor memory usage
watch -n 5 'free -h'
```

**Permission Issues**
```bash
# Fix script permissions
chmod +x run_server_analysis.sh
chmod +x server_deployment.py

# Fix directory permissions
chmod 755 score/
```

**Process Hanging**
```bash
# Check for zombie processes
ps aux | grep defunct

# Kill stuck processes
pkill -f enhanced_functionality_analyzer
```

#### Resume Interrupted Analysis
```bash
# Check completed tasks
ls score/enhanced_functionality_analysis_*.json

# Resume from next incomplete task
python3 server_deployment.py --resume <next_task>
```

### 10. Results Analysis

After completion, you'll have:
- Individual task analysis files
- Combined results in `enhanced_functionality_analysis_all_tasks.json`
- Detailed logs for debugging
- Automatic backups in `score/backups/`

The enhanced analyzer provides:
- Model-specific system prompt processing
- Complete multi-turn context analysis
- Function documentation validation
- Comprehensive mismatch type detection

### 11. Performance Optimization

#### For High-Performance Servers
```bash
# Maximum performance (use all CPU cores)
python3 server_deployment.py --workers $(nproc)

# Enable CPU affinity (if available)
taskset -c 0-31 python3 server_deployment.py --workers 32
```

#### For Resource-Constrained Environments
```bash
# Conservative settings
python3 server_deployment.py --workers 4
```

## 🔧 Support and Maintenance

- **Logs**: Check `bfcl_analysis_*.log` files for detailed execution logs
- **Backups**: Automatic backups every 5 minutes in `score/backups/`
- **Resume**: Built-in capability to resume from any interruption point
- **Monitoring**: Real-time progress updates and resource usage tracking

The enhanced analyzer is production-ready with comprehensive error handling, resource management, and progress tracking.