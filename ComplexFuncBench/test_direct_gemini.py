#!/usr/bin/env python3
"""
Test Gemini directly with new URL
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_direct_gemini():
    print("=== Testing Gemini Direct Call ===")

    # Use the new endpoint
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="http://5.78.122.79:12500/v1/"
    )

    # Gemini format tools (without OpenAI wrapper)
    tools = [{
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "city name"}
            },
            "required": ["location"]
        }
    }]

    messages = [{"role": "user", "content": "How is the weather in Boston?"}]

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-pro-thinking-on",
            messages=messages,
            tools=tools,
            temperature=0.0,
            max_tokens=1000
        )
        print(f"✅ SUCCESS")
        print(f"Response: {response}")
        print(f"Content: {response.choices[0].message.content}")
        if hasattr(response.choices[0].message, 'tool_calls'):
            print(f"Tool calls: {response.choices[0].message.tool_calls}")
    except Exception as e:
        print(f"❌ FAILED: {e}")

if __name__ == "__main__":
    test_direct_gemini()