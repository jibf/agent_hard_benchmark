#!/bin/bash

# AgentHard Unified Environment Setup Script
# Single environment approach based on compatibility analysis

set -e

echo "Setting up unified AgentHard environment..."

# Check conda availability
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Anaconda or Miniconda first"
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "agenthard"; then
    echo "Environment 'agenthard' already exists"
    echo "Do you want to:"
    echo "  1) Continue with existing environment (skip creation)"
    echo "  2) Remove and recreate environment"
    echo "  3) Exit"
    read -p "Choose option (1/2/3): " choice
    
    case $choice in
        1)
            echo "Using existing environment..."
            ;;
        2)
            echo "Removing existing environment..."
            conda env remove -n agenthard -y
            echo "Creating new environment agenthard..."
            conda create -n agenthard python=3.11 -y
            ;;
        3)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Using existing environment..."
            ;;
    esac
else
    echo "Creating unified environment agenthard..."
    conda create -n agenthard python=3.11 -y
fi

# Activate environment
echo "Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate agenthard

# Install unified requirements
echo "Installing unified dependencies..."
if python -c "import torch, transformers, openai" 2>/dev/null; then
    echo "Core packages already installed, checking versions..."
    python -c "
import torch, transformers, openai
print(f'Current PyTorch: {torch.__version__}')
print(f'Current Transformers: {transformers.__version__}')
print(f'Current OpenAI: {openai.__version__}')
"
    echo "Do you want to reinstall dependencies? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        pip install -r requirements.txt
    else
        echo "Skipping dependency installation..."
    fi
else
    echo "Installing fresh dependencies..."
    pip install -r requirements.txt
fi

# Install editable packages for each benchmark
echo "Installing benchmark packages..."

# Function to install package with check
install_benchmark_package() {
    local dir_name=$1
    local package_name=$2
    
    if [ -d "$dir_name" ]; then
        if python -c "import $package_name" 2>/dev/null; then
            echo "  $package_name already installed, skipping..."
        else
            echo "  Installing $package_name..."
            cd "$dir_name" && pip install -e . && cd - >/dev/null
        fi
    else
        echo "  $dir_name directory not found, skipping..."
    fi
}

# Install benchmark packages
install_benchmark_package "ToolSandbox" "tool_sandbox"

# NexusBench - Handle nexusflowai version conflict
if [ -d "NexusBench" ]; then
    if python -c "import nexusbench" 2>/dev/null; then
        echo "  nexusbench already installed, skipping..."
    else
        echo "  Installing NexusBench..."
        cd "NexusBench"
        # Install nexusflowai separately with older openai version if needed
        echo "    Note: nexusflowai may conflict with litellm's openai version requirement"
        echo "    Attempting to install with current openai version..."
        pip install -e .
        cd - >/dev/null
    fi
else
    echo "  NexusBench directory not found, skipping..."
fi

install_benchmark_package "tau2-bench" "tau2"

# Special cases without importable module names
if [ -d "DrafterBench" ]; then
    if [ -f "DrafterBench/setup.py" ]; then
        echo "  Installing DrafterBench..."
        cd DrafterBench && pip install -e . && cd - >/dev/null
    else
        echo "  DrafterBench (no setup.py), skipping..."
    fi
else
    echo "  DrafterBench directory not found, skipping..."
fi

if [ -d "gorilla/berkeley-function-call-leaderboard" ]; then
    if python -c "import bfcl_eval" 2>/dev/null; then
        echo "  BFCL already installed, skipping..."
    else
        echo "  Installing BFCL..."
        cd gorilla/berkeley-function-call-leaderboard && pip install -e . && cd - >/dev/null
    fi
else
    echo "  BFCL directory not found, skipping..."
fi

if [ -d "tau-bench" ]; then
    if python -c "import tau_bench" 2>/dev/null; then
        echo "  tau-bench already installed, skipping..."
    else
        echo "  Installing tau-bench..."
        cd tau-bench && pip install -e . && cd - >/dev/null
    fi
else
    echo "  tau-bench directory not found, skipping..."
fi

# ComplexFuncBench (install requirements only, no setup.py)
if [ -d "ComplexFuncBench" ]; then
    echo "  Installing ComplexFuncBench dependencies..."
    # Use requirements.txt instead of pyproject.toml to avoid torch>=2.8.0 issue
    cd ComplexFuncBench
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    cd ..
fi

# MultiChallenge (install requirements only)
if [ -d "multi_challenge" ]; then
    echo "  Installing MultiChallenge dependencies..."
    cd multi_challenge
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    cd ..
fi

# Optional: Install vLLM if needed (commented out by default due to potential conflicts)
# echo "Install vLLM for local model inference? (y/N)"
# read -r response
# if [[ "$response" =~ ^[Yy]$ ]]; then
#     echo "Installing vLLM..."
#     pip install vllm>=0.8.0
# fi
pip install torchvision==0.20.1

echo ""
echo "Unified environment setup completed!"
echo ""
echo "Environment: agenthard"
echo ""
echo "To use the environment:"
echo "  conda activate agenthard"
echo ""
echo "To run all benchmarks:"
echo "  python run_benchmarks.py <api_key> <base_url> <model_name>"
echo ""
echo "Testing environment compatibility..."
echo "Checking key packages:"

# Test imports
conda activate agenthard
python -c "
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    import transformers
    print(f'Transformers: {transformers.__version__}')
    import openai
    print(f'OpenAI: {openai.__version__}')
    import anthropic
    print(f'Anthropic: {anthropic.__version__}')
    print('All key packages imported successfully!')
except Exception as e:
    print(f'Import error: {e}')
    exit(1)
"

echo ""
echo "Ready to run benchmarks!"