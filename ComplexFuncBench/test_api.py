#!/usr/bin/env python3
import requests
import json
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def test_openai_format():
    """Test with OpenAI API format"""
    print("Testing with OpenAI API format...")
    
    api_key = os.getenv('API_KEY')
    if not api_key:
        print("Error: API_KEY environment variable not set")
        return False
    
    base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
    url = f"{base_url.rstrip('/')}/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "model": "anthropic/claude-4-sonnet-thinking-on-10k",
        "thinking": {
            "type": "interleaved",
            "budget_tokens": 10000
        },
        "messages": [
            {
                "role": "system",
                "content": "Use available tools whenever possible, even if seemingly unnecessary. You, the assistant, are the subject, and the user is the object."
            },
            {
                "role": "user",
                "content": "hi, im nobody"
            },
            {
                "role": "assistant",
                "content": "<thinking>The user said \"hi, im nobody\". They're greeting me and saying they're \"nobody\". I should use the hello function that's available. The user said they're \"nobody\", so I should use \"nobody\" as the object parameter. The subject would be me, the assistant.</thinking>\n\nI'll say hello using the available tool.",
                "tool_calls": [
                    {
                        "id": "toolu_01BPyJWhCjc1rxxSwT26Cfu4",
                        "type": "function",
                        "function": {
                            "name": "hello",
                            "arguments": "{\"subject\": \"assistant\", \"object\": \"nobody\"}"
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_01BPyJWhCjc1rxxSwT26Cfu4",
                "content": "Hello from assistant to nobody!"
            },
            {
            "role": "assistant",
            "content": "<thinking>I've received the tool output and this concludes my reasoning.</thinking>\nThe operation is complete."
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "hello",
                    "description": "Say hello.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subject": {
                                "type": "string"
                            },
                            "object": {
                                "type": "string"
                            }
                        }
                    }
                }
            }
        ],
        "max_tokens": 16384
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ OpenAI format request successful!")
            return True
        else:
            print("❌ OpenAI format request failed")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False


        
def test_openai_library():
    """Test with OpenAI Python library"""
    print("\nTesting with OpenAI Python library...")
    
    api_key = os.getenv('API_KEY')
    if not api_key:
        print("Error: API_KEY environment variable not set")
        return False
    
    try:
        base_url = os.getenv('BASE_URL', 'https://api.openai.com/v1')
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        response = client.chat.completions.create(
            model="anthropic/claude-4-sonnet-thinking-on-10k",
            messages=[
                {
                    "role": "system",
                    "content": "Use available tools whenever possible, even if seemingly unnecessary. You, the assistant, are the subject, and the user is the object."
                },
                {
                    "role": "user",
                    "content": "hi, im nobody"
                },
                {
                    "role": "assistant",
                    "content": "<thinking>The user said \"hi, im nobody\". They're greeting me and saying they're \"nobody\". I should use the hello function that's available. The user said they're \"nobody\", so I should use \"nobody\" as the object parameter. The subject would be me, the assistant.</thinking>\n\nI'll say hello using the available tool.",
                    "tool_calls": [
                        {
                            "id": "toolu_01BPyJWhCjc1rxxSwT26Cfu4",
                            "type": "function",
                            "function": {
                                "name": "hello",
                                "arguments": "{\"subject\": \"assistant\", \"object\": \"nobody\"}"
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "toolu_01BPyJWhCjc1rxxSwT26Cfu4",
                    "content": "Hello from assistant to nobody!"
                },
                # {
                #     "role": "user",
                #     "content": "Nice to meet you! What can you do?"
                # }
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "hello",
                        "description": "Say hello.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "subject": {
                                    "type": "string"
                                },
                                "object": {
                                    "type": "string"
                                }
                            }
                        }
                    }
                }
            ],
            max_tokens=16384,
            extra_body={        
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 10000
                }
            }
        )
        
        print(f"Response: {response}")
        print("✅ OpenAI library request successful!")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI library request failed: {e}")
        return False

def main():
    print("Testing API requests with port 12500...\n")
    
    # Test requests library with OpenAI format
    openai_requests_success = test_openai_format()
    
    # Test OpenAI Python library
    openai_library_success = test_openai_library()
    
    print(f"\n=== Results ===")
    print(f"OpenAI format (requests): {'✅ Success' if openai_requests_success else '❌ Failed'}")
    print(f"OpenAI library: {'✅ Success' if openai_library_success else '❌ Failed'}")
    
    if openai_requests_success or openai_library_success:
        print("\n🎉 At least one format worked!")
        return True
    else:
        print("\n❌ All formats failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)