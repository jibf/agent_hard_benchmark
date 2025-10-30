#!/usr/bin/env python3
"""
Test Gemini directly with new URL
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

from litellm import completion

load_dotenv()

def test_direct_gemini():
    print("=== Testing Gemini Direct Call ===")

    # Use the new endpoint
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        # api_key = "sk-sgl-MH7bEVVJlBp3RT_P5cPQ6-KfC1qJElBRCfTDHy40Ue4",
        base_url="http://5.78.122.79:10000/v1/"
    )

    # Gemini format tools (without OpenAI wrapper)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information for a specific city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "city name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    messages = [{"role": "user", "content": "How is the weather in Boston?"}]

    # try:
    #     response = client.chat.completions.create(
    #         model="google/gemini-2.5-flash-thinking-off",
    #         messages=messages,
    #         tools=tools,
    #         temperature=0.0,
    #         max_tokens=1000,
    #         tool_choice="auto"
    #     )
    #     print(f"✅ SUCCESS")
    #     print(f"Response: {response}")
    #     print(f"Content: {response.choices[0].message.content}")
    #     if hasattr(response.choices[0].message, 'tool_calls'):
    #         print(f"Tool calls: {response.choices[0].message.tool_calls}")
    # except Exception as e:
    #     print(f"❌ FAILED: {e}")


    # try:
    #     response = completion(
    #         model="google/gemini-2.5-flash-thinking-off",
    #         messages=messages,
    #         tools=tools,
    #         temperature=0.0,
    #         max_tokens=1000,
    #         tool_choice="auto",
    #         api_key=os.getenv("API_KEY"),
    #         api_base="http://5.78.122.79:10000/v1/",
    #         custom_llm_provider="openai"
    #     )
    #     print(f"✅ SUCCESS")
    #     print(f"Response: {response}")
    #     print(f"Content: {response.choices[0].message.content}")
    #     if hasattr(response.choices[0].message, 'tool_calls'):
    #         print(f"Tool calls: {response.choices[0].message.tool_calls}")
    # except Exception as e:
    #     print(f"❌ FAILED: {e}")


    # litellm_messat 'function', 'function': {'name': 'find_user_id_by_name_zip', 'description': 'Find user id by first name, last name, and zip code. If the user is not found, the function\n\nwill return an error message. By default, find user id by email, and only call this function\nif the user is not found by email or cannot remember email.', 'parameters': {'properties': {'first_name': {'description': "The first name of the customer, such as 'John'.", 'title': 'First Name', 'type': 'string'}, 'last_name': {'description': "The last name of the customer, such as 'Doe'.", 'title': 'Last Name', 'type': 'string'}, 'zip': {'description': "The zip code of the customer, such as '12345'.", 'title': 'Zip', 'type': 'string'}}, 'required': ['first_name', 'last_name', 'zip'], 'title': 'parameters', 'type': 'object'}}}]


    litellm_messages = [
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
    kwargs ={'temperature': 0.0, 'custom_llm_provider': 'openai', 'api_key': os.getenv("API_KEY"), 'api_base': 'http://5.78.122.79:10000/v1', 'tool_choice': 'auto'}

    # Use Gemini model specifically
    model = "google/gemini-2.5-flash-thinking-off"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=litellm_messages,
            tools=complex_tools,
            temperature=0.0,
            max_tokens=1000,
            tool_choice="auto"
        )
        print(f"✅ SUCCESS")
        print(f"Response: {response}")
        print(f"Content: {response.choices[0].message.content}")
        if hasattr(response.choices[0].message, 'tool_calls'):
            print(f"Tool calls: {response.choices[0].message.tool_calls}")
    except Exception as e:
        print(f"❌ FAILED: {e}")


    try:
        response = completion(
            model=model,
            messages=litellm_messages,
            tools=tools,
            temperature=0.0,
            max_tokens=1000,
            tool_choice="auto",
            api_key=os.getenv("API_KEY"),
            api_base="http://5.78.122.79:10000/v1/",
            custom_llm_provider="openai"
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