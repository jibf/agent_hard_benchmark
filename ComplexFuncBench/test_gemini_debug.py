#!/usr/bin/env python3
"""
Debug Gemini-specific tool call issues
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

def test_gemini_debug():
    print("=== Debugging Gemini Tool Call Issues ===")
    
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="http://5.78.122.79:10000/v1/"
    )
    
    # Simple test case
    messages = [{"role": "user", "content": "What's the weather in Boston?"}]
    tools = [
        {
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
        }
    ]
    
    # Test 1: Qwen (should work)
    print("\n🧪 TEST 1: Qwen model (baseline)")
    try:
        response = client.chat.completions.create(
            model="togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
            messages=messages,
            tools=tools,
            temperature=0.0,
            tool_choice="auto"
        )
        print("✅ Qwen SUCCESS")
        print(f"Content: {response.choices[0].message.content}")
        if hasattr(response.choices[0].message, 'tool_calls'):
            print(f"Tool calls: {response.choices[0].message.tool_calls}")
    except Exception as e:
        print(f"❌ Qwen FAILED: {e}")
    
    # Test 2: Gemini without tools (should work)
    print("\n🧪 TEST 2: Gemini without tools")
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-thinking-off",
            messages=messages,
            temperature=0.0
        )
        print("✅ Gemini without tools SUCCESS")
        print(f"Content: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ Gemini without tools FAILED: {e}")
    
    # Test 3: Gemini with tools (the problematic case)
    print("\n🧪 TEST 3: Gemini with tools")
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-thinking-off",
            messages=messages,
            tools=tools,
            temperature=0.0,
            tool_choice="auto"
        )
        print("✅ Gemini with tools SUCCESS")
        print(f"Content: {response.choices[0].message.content}")
        if hasattr(response.choices[0].message, 'tool_calls'):
            print(f"Tool calls: {response.choices[0].message.tool_calls}")
    except Exception as e:
        print(f"❌ Gemini with tools FAILED: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
    
    # Test 4: Different Gemini models
    print("\n🧪 TEST 4: Different Gemini models")
    gemini_models = [
        "google/gemini-2.5-flash-thinking-off",
        "google/gemini-2.5-pro-thinking-on",
        "google/gemini-1.5-pro",
        "google/gemini-1.5-flash"
    ]
    
    for model in gemini_models:
        print(f"\n  Testing {model}")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=0.0,
                tool_choice="auto"
            )
            print(f"  ✅ {model} SUCCESS")
        except Exception as e:
            print(f"  ❌ {model} FAILED: {e}")
    
    # Test 5: Gemini with different tool_choice values
    print("\n🧪 TEST 5: Gemini with different tool_choice values")
    tool_choices = ["auto", "required", "none"]
    
    for tool_choice in tool_choices:
        print(f"\n  Testing tool_choice='{tool_choice}'")
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.5-flash-thinking-off",
                messages=messages,
                tools=tools,
                temperature=0.0,
                tool_choice=tool_choice
            )
            print(f"  ✅ tool_choice='{tool_choice}' SUCCESS")
        except Exception as e:
            print(f"  ❌ tool_choice='{tool_choice}' FAILED: {e}")
    
    # Test 6: Check if it's a model routing issue
    print("\n🧪 TEST 6: Check model routing")
    try:
        # Try without specifying the google/ prefix
        response = client.chat.completions.create(
            model="gemini-2.5-flash-thinking-off",
            messages=messages,
            tools=tools,
            temperature=0.0,
            tool_choice="auto"
        )
        print("✅ Without google/ prefix SUCCESS")
    except Exception as e:
        print(f"❌ Without google/ prefix FAILED: {e}")

if __name__ == "__main__":
    test_gemini_debug()
