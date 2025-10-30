#!/usr/bin/env python3
"""
Test to isolate the Gemini tool calls in conversation history issue
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_gemini_tool_calls_issue():
    print("=== Isolating Gemini Tool Calls Issue ===")
    
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="http://5.78.122.79:10000/v1/"
    )
    
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
    
    # Test 1: Simple request (should work)
    print("\n🧪 TEST 1: Simple request for new tool call")
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-thinking-off",
            messages=[{"role": "user", "content": "What's the weather in Boston?"}],
            tools=tools,
            temperature=0.0,
            tool_choice="auto"
        )
        print("✅ SUCCESS - Simple request")
        print(f"Content: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ FAILED - Simple request: {e}")
    
    # Test 2: Tool call in conversation history (problematic)
    print("\n🧪 TEST 2: Tool call in conversation history")
    messages_with_tool_call = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Boston"}'
                    }
                }
            ]
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-thinking-off",
            messages=messages_with_tool_call,
            tools=tools,
            temperature=0.0,
            tool_choice="auto"
        )
        print("✅ SUCCESS - Tool call in history")
        print(f"Content: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ FAILED - Tool call in history: {e}")
    
    # Test 3: Tool response in conversation history (problematic)
    print("\n🧪 TEST 3: Tool response in conversation history")
    messages_with_tool_response = [
        {"role": "user", "content": "Hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Boston"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": "The weather in Boston is 72°F and sunny.",
            "tool_call_id": "call_123"
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-thinking-off",
            messages=messages_with_tool_response,
            tools=tools,
            temperature=0.0,
            tool_choice="auto"
        )
        print("✅ SUCCESS - Tool response in history")
        print(f"Content: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ FAILED - Tool response in history: {e}")
    
    # Test 4: Compare with Qwen (should work)
    print("\n🧪 TEST 4: Qwen with tool calls in history (baseline)")
    try:
        response = client.chat.completions.create(
            model="togetherai/Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
            messages=messages_with_tool_response,
            tools=tools,
            temperature=0.0,
            tool_choice="auto"
        )
        print("✅ SUCCESS - Qwen with tool calls in history")
        print(f"Content: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ FAILED - Qwen with tool calls in history: {e}")
    
    # Test 5: Different tool call formats for Gemini
    print("\n🧪 TEST 5: Different tool call formats for Gemini")
    
    # Format 1: Standard format
    standard_tool_call = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "Boston"}'
        }
    }
    
    # Format 2: Without type field
    no_type_tool_call = {
        "id": "call_123",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "Boston"}'
        }
    }
    
    # Format 3: With name at top level (problematic format)
    with_name_tool_call = {
        "id": "call_123",
        "name": "get_weather",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"location": "Boston"}'
        }
    }
    
    formats = [
        ("Standard", standard_tool_call),
        ("No type", no_type_tool_call),
        ("With name", with_name_tool_call)
    ]
    
    for format_name, tool_call in formats:
        print(f"\n  Testing {format_name} format")
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.5-flash-thinking-off",
                messages=[
                    {"role": "user", "content": "Hi"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call]
                    }
                ],
                tools=tools,
                temperature=0.0,
                tool_choice="auto"
            )
            print(f"  ✅ {format_name} SUCCESS")
        except Exception as e:
            print(f"  ❌ {format_name} FAILED: {e}")

if __name__ == "__main__":
    test_gemini_tool_calls_issue()
