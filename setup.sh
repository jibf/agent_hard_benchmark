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

# Install benchmark packages first to get all dependencies
echo "Installing benchmark packages (will install their dependencies)..."

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
# Install all benchmark packages using consistent function
install_benchmark_package "NexusBench" "nexusbench"
install_benchmark_package "tau2-bench" "tau2"
install_benchmark_package "DrafterBench" "drafterbench"
install_benchmark_package "gorilla/berkeley-function-call-leaderboard" "bfcl_eval"  
install_benchmark_package "tau-bench" "tau_bench"
install_benchmark_package "ToolSandbox" "tool_sandbox"

# Now install any remaining dependencies from requirements.txt that weren't covered by benchmark packages
echo "Installing any remaining dependencies from requirements.txt..."
if python -c "import torch, transformers, openai" 2>/dev/null; then
    echo "Core packages already installed via benchmark dependencies"
    echo "Do you want to install additional requirements.txt dependencies? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        pip install -r requirements.txt
    else
        echo "Skipping additional requirements.txt installation..."
    fi
else
    echo "Installing requirements.txt for any missing dependencies..."
    pip install -r requirements.txt
fi

# ComplexFuncBench (install requirements only, no setup.py)
# if [ -d "ComplexFuncBench" ]; then
#     echo "  Installing ComplexFuncBench dependencies..."
#     # Use requirements.txt instead of pyproject.toml to avoid torch>=2.8.0 issue
#     cd ComplexFuncBench
#     if [ -f "requirements.txt" ]; then
#         pip install -r requirements.txt
#     fi
#     cd ..
# fi

# # MultiChallenge (install requirements only)
# if [ -d "multi_challenge" ]; then
#     echo "  Installing MultiChallenge dependencies..."
#     cd multi_challenge
#     if [ -f "requirements.txt" ]; then
#         pip install -r requirements.txt
#     fi
#     cd ..
# fi

# # ACEBench (install requirements only)
# if [ -d "ACEBench" ]; then
#     echo "  Installing ACEBench dependencies..."
#     cd ACEBench
#     if [ -f "requirements.txt" ]; then
#         pip install -r requirements.txt
#     fi
#     cd ..
# else
#     echo "  ACEBench directory not found, skipping..."
# fi

# Optional: Install vLLM if needed (commented out by default due to potential conflicts)
# echo "Install vLLM for local model inference? (y/N)"
# read -r response
# if [[ "$response" =~ ^[Yy]$ ]]; then
#     echo "Installing vLLM..."
#     pip install vllm>=0.8.0
# fi
pip install torchvision==0.20.1
pip install wcwidth

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