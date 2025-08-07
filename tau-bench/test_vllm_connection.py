import os
import litellm

# Set environment variables for VLLM server mode
os.environ["VLLM_API_BASE"] = "http://localhost:8000/v1"
os.environ["VLLM_API_KEY"] = "dummy-key"

# Test connection
try:
    response = litellm.completion(
        model="vllm/Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": "Hello, this is a test"}],
        max_tokens=10
    )
    print("✅ Connection successful!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
