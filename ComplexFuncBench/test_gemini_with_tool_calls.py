#!/usr/bin/env python3
"""
Test Gemini with tool calls included in the request (conversation history)
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

def test_gemini_with_tool_calls():
    print("=== Testing Gemini with Tool Calls in Request ===")
    
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="http://5.78.122.79:10000/v1/"
    )
    
    # Messages that include tool calls in the conversation history
    messages_with_tool_calls = [
        {"role": "user", "content": "Hi, I need help with my order."},
        {
            "role": "assistant",
            "content": "",
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
            "content": "{\"weather\": \"The weather in Boston is 72°F and sunny.\"}", # <--- Manual JSON string            
            "tool_call_id": "call_123"
        },
        {"role": "user", "content": "Thanks! Now can you help me with my order?"}
    ]
    
    # Tools for the conversation
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
        },
        {
            "type": "function", 
            "function": {
                "name": "get_order_status",
                "description": "Get order status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "order ID"}
                    },
                    "required": ["order_id"]
                }
            }
        }
    ]
    
    # Test 1: OpenAI direct API with tool calls in request
    print("\n🧪 TEST 1: OpenAI direct API with tool calls in request")
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-thinking-off",
            messages=messages_with_tool_calls,
            tools=tools,
            temperature=0.0,
            tool_choice="auto"
        )
        print("✅ SUCCESS - OpenAI direct API")
        print(f"Content: {response.choices[0].message.content}")
        if hasattr(response.choices[0].message, 'tool_calls'):
            print(f"Tool calls: {response.choices[0].message.tool_calls}")
    except Exception as e:
        print(f"❌ FAILED - OpenAI direct API: {e}")
    
    # Test 2: LiteLLM with tool calls in request
    print("\n🧪 TEST 2: LiteLLM with tool calls in request")
    try:
        response = completion(
            model="google/gemini-2.5-flash-thinking-off",
            messages=messages_with_tool_calls,
            tools=tools,
            temperature=0.0,
            tool_choice="auto",
            api_key=os.getenv("API_KEY"),
            api_base="http://5.78.122.79:10000/v1/",
            custom_llm_provider="openai"
        )
        print("✅ SUCCESS - LiteLLM")
        print(f"Content: {response}")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"Tool calls: {len(response.tool_calls)}")
    except Exception as e:
        print(f"❌ FAILED - LiteLLM: {e}")
    
    # Test 3: Complex conversation with multiple tool calls
    print("\n🧪 TEST 3: Complex conversation with multiple tool calls")
    complex_messages = [
        {"role": "user", "content": "I need help with my account and order."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_auth_123",
                    "type": "function",
                    "function": {
                        "name": "find_user_id_by_name_zip",
                        "arguments": '{"first_name": "John", "last_name": "Doe", "zip": "12345"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": "user_12345",
            "tool_call_id": "call_auth_123"
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_order_456",
                    "type": "function",
                    "function": {
                        "name": "get_order_status",
                        "arguments": '{"order_id": "ORD-123"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": "Order ORD-123 is shipped and will arrive tomorrow.",
            "tool_call_id": "call_order_456"
        },
        {"role": "user", "content": "Great! Can you help me track it?"}
    ]
    
    complex_tools = [
        {
            "type": "function",
            "function": {
                "name": "find_user_id_by_name_zip",
                "description": "Find user by name and zip",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string"},
                        "last_name": {"type": "string"},
                        "zip": {"type": "string"}
                    },
                    "required": ["first_name", "last_name", "zip"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_order_status",
                "description": "Get order status",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"}
                    },
                    "required": ["order_id"]
                }
            }
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-thinking-off",
            messages=complex_messages,
            tools=complex_tools,
            temperature=0.0,
            tool_choice="auto"
        )
        print("✅ SUCCESS - Complex conversation")
        print(f"Content: {response.choices[0].message.content}")
        if hasattr(response.choices[0].message, 'tool_calls'):
            print(f"Tool calls: {response.choices[0].message.tool_calls}")
    except Exception as e:
        print(f"❌ FAILED - Complex conversation: {e}")

if __name__ == "__main__":
    test_gemini_with_tool_calls()
