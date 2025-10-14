#!/usr/bin/env python3
"""
Test the fixed Gemini integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.gpt import FunctionCallGPT

def test_gemini_tools():
    print("=== Testing Fixed Gemini Tool Integration ===")
    print(f"Using BASE_URL: {os.environ.get('BASE_URL')}")

    model = FunctionCallGPT("google/gemini-2.5-pro-thinking-on")

    # Test with tools (same format as successful curl)
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "city name"}
                },
                "required": ["location"]
            }
        }
    }]

    messages = [{"role": "user", "content": "How is the weather in Boston?"}]

    try:
        response = model(messages, tools=tools)
        print(f"✅ SUCCESS")
        print(f"Content: {response.content}")
        print(f"Tool calls: {response.tool_calls}")
        if hasattr(response, 'reasoning_content'):
            print(f"Has reasoning: Yes ({len(response.reasoning_content)} chars)")
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    test_gemini_tools()