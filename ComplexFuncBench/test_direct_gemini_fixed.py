#!/usr/bin/env python3
"""
Test Gemini directly with new URL - FIXED VERSION
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

def test_direct_gemini():
    print("=== Testing Gemini Direct Call (FIXED) ===")

    # Use the new endpoint
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url="http://5.78.122.79:10000/v1/"
    )

    # Simple test first
    print("\n🧪 TEST 1: Simple weather query")
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-thinking-off",
            messages=[{"role": "user", "content": "How is the weather in Boston?"}],
            tools=[{
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
            }],
            temperature=0.0,
            tool_choice="auto"
        )
        print("✅ SUCCESS - Simple query")
        print(f"Content: {response.choices[0].message.content}")
        if hasattr(response.choices[0].message, 'tool_calls'):
            print(f"Tool calls: {response.choices[0].message.tool_calls}")
    except Exception as e:
        print(f"❌ FAILED - Simple query: {e}")

    # FIXED: Messages with correct tool_calls format
    print("\n🧪 TEST 2: Complex conversation with FIXED tool_calls format")
    
    # ✅ CORRECT FORMAT - No extra 'name' field at top level
    fixed_litellm_messages = [
        {'role': 'user', 'content': "Hi! I recently received my order, and I'd like to exchange a couple of items. Could you help me with that?"},
        {'role': 'assistant', 'content': 'Of course! I can help with that. First, I need to verify your identity. Can you please provide the email address associated with your account?'},
        {'role': 'user', 'content': "I'm sorry, but I don't remember the email address associated with my account. However, I can provide my order number and any other details you might need."},
        {'role': 'assistant', 'content': 'No problem. I can also find your account using your full name and zip code.'},
        {'role': 'user', 'content': 'Great! My name is Yusuf Rossi, and my zip code is 19122.'},
        {
            'role': 'assistant', 
            'content': None, 
            'tool_calls': [
                {
                    'id': 'sgl_gemini_tool_call_56cb0d45_6464_4fb9_b757_b75fa38fa3c1',
                    'type': 'function',  # ✅ Correct: type at top level
                    'function': {       # ✅ Correct: function object
                        'name': 'find_user_id_by_name_zip',
                        'arguments': '{"last_name": "Rossi", "first_name": "Yusuf", "zip": "19122"}'
                    }
                    # ❌ REMOVED: 'name' field at top level (this was the problem!)
                }
            ]
        },
        {
            'role': 'tool', 
            'content': 'yusuf_rossi_9620', 
            'tool_call_id': 'sgl_gemini_tool_call_56cb0d45_6464_4fb9_b757_b75fa38fa3c1'
        }
    ]
    
    # Tools for the complex conversation
    complex_tools = [
        {
            "type": "function",
            "function": {
                "name": "find_user_id_by_name_zip",
                "description": "Find user id by first name, last name, and zip code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "first_name": {"type": "string", "description": "The first name"},
                        "last_name": {"type": "string", "description": "The last name"},
                        "zip": {"type": "string", "description": "The zip code"}
                    },
                    "required": ["first_name", "last_name", "zip"]
                }
            }
        }
    ]

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash-thinking-off",
            messages=fixed_litellm_messages,
            tools=complex_tools,
            temperature=0.0,
            max_tokens=1000,
            tool_choice="auto"
        )
        print("✅ SUCCESS - Complex conversation with fixed format")
        print(f"Content: {response.choices[0].message.content}")
        if hasattr(response.choices[0].message, 'tool_calls'):
            print(f"Tool calls: {response.choices[0].message.tool_calls}")
    except Exception as e:
        print(f"❌ FAILED - Complex conversation: {e}")

    # LiteLLM test with fixed format
    print("\n🧪 TEST 3: LiteLLM with FIXED format")
    try:
        response = completion(
            model="google/gemini-2.5-flash-thinking-off",
            messages=fixed_litellm_messages,
            tools=complex_tools,
            temperature=0.0,
            max_tokens=1000,
            tool_choice="auto",
            api_key=os.getenv("API_KEY"),
            api_base="http://5.78.122.79:10000/v1/",
            custom_llm_provider="openai"
        )
        print("✅ SUCCESS - LiteLLM with fixed format")
        print(f"Content: {response.content}")
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"Tool calls: {len(response.tool_calls)}")
    except Exception as e:
        print(f"❌ FAILED - LiteLLM: {e}")

if __name__ == "__main__":
    test_direct_gemini()
