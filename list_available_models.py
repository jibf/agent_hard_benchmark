#!/usr/bin/env python3
"""
Script to list available models from configured API endpoints.
This helps users verify their API configuration and see which models are available.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def load_environment():
    """Load environment variables from .env file"""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded environment from {env_path}")
    else:
        print("No .env file found, using system environment variables")

def test_api_connection(api_key, base_url, endpoint_name):
    """Test API connection and list available models"""
    if not api_key:
        print(f"{endpoint_name}: API_KEY not set")
        return []

    if not base_url:
        print(f"{endpoint_name}: BASE_URL not set")
        return []

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        print(f"{endpoint_name}: Testing connection to {base_url}")

        # List available models
        models = client.models.list()
        model_ids = [model.id for model in models.data]

        print(f"{endpoint_name}: Connected successfully")
        print(f"{endpoint_name}: Found {len(model_ids)} models")

        return model_ids

    except Exception as e:
        print(f"{endpoint_name}: Connection failed - {str(e)}")
        return []

def main():
    print("Agent Hard Benchmark - Model Availability Checker")
    print("=" * 60)

    # Load environment variables
    load_environment()

    # Check agent/client model configuration
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")

    print("\nAGENT/CLIENT MODELS:")
    print("-" * 30)
    agent_models = test_api_connection(api_key, base_url, "Agent Endpoint")

    if agent_models:
        print("Available models for agents:")
        for i, model in enumerate(sorted(agent_models), 1):
            print(f"  {i:2d}. {model}")

    # Check user model configuration (for multi-agent benchmarks)
    user_api_key = os.getenv("USER_API_KEY")
    user_base_url = os.getenv("USER_BASE_URL")

    print("\nUSER MODELS (for multi-agent benchmarks):")
    print("-" * 45)

    if user_api_key or user_base_url:
        user_models = test_api_connection(user_api_key, user_base_url, "User Endpoint")

        if user_models:
            print("Available models for users:")
            for i, model in enumerate(sorted(user_models), 1):
                print(f"  {i:2d}. {model}")
    else:
        print("USER_API_KEY and USER_BASE_URL not configured")
        print("This is optional and only needed for multi-agent benchmarks like ToolSandbox")

    print("\nUSAGE TIPS:")
    print("-" * 15)
    print("• Use any model ID from the 'Agent Endpoint' for model parameter")
    print("• Use any model ID from the 'User Endpoint' for --user-model parameter (if applicable)")
    print("• Make sure your model supports the features required by the benchmark")
    print("• For function calling benchmarks, ensure your model supports tool/function calling")

    print("\nCONFIGURATION:")
    print("-" * 17)
    print("Current .env configuration:")
    print(f"  API_KEY: {'Set' if api_key else 'Not set'}")
    print(f"  BASE_URL: {'Set' if base_url else 'Not set'} ({base_url or 'N/A'})")
    print(f"  USER_API_KEY: {'Set' if user_api_key else 'Not set'}")
    print(f"  USER_BASE_URL: {'Set' if user_base_url else 'Not set'} ({user_base_url or 'N/A'})")

if __name__ == "__main__":
    main()