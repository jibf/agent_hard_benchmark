#!/usr/bin/env python3
"""
Test script to find optimal concurrency limits for custom API
"""

import os
import time
import asyncio
import aiohttp
from dotenv import load_dotenv
from openai import OpenAI
import concurrent.futures

# Load environment variables
load_dotenv()

def test_single_request():
    """Test a single request to verify API is working"""
    print("🔍 Testing single request...")
    
    client = OpenAI(
        base_url="http://5.78.122.79:10000/v1",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello'"}],
            max_tokens=10
        )
        print(f"✅ Single request successful: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"❌ Single request failed: {e}")
        return False

def test_concurrent_requests(concurrency_level):
    """Test concurrent requests at a specific level"""
    print(f"🔍 Testing {concurrency_level} concurrent requests...")
    
    def make_request():
        client = OpenAI(
            base_url="http://5.78.122.79:10000/v1",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'hello'"}],
                max_tokens=10
            )
            return True, response.choices[0].message.content
        except Exception as e:
            return False, str(e)
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
        futures = [executor.submit(make_request) for _ in range(concurrency_level)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    end_time = time.time()
    duration = end_time - start_time
    
    successful = sum(1 for success, _ in results if success)
    failed = concurrency_level - successful
    
    print(f"   ✅ Successful: {successful}/{concurrency_level}")
    print(f"   ❌ Failed: {failed}/{concurrency_level}")
    print(f"   ⏱️  Duration: {duration:.2f}s")
    
    if failed > 0:
        print(f"   📝 Error examples:")
        for success, error in results:
            if not success:
                print(f"      - {error[:100]}...")
    
    return successful == concurrency_level

def find_optimal_concurrency():
    """Find the optimal concurrency level"""
    print("🚀 Starting concurrency limit test...")
    print("=" * 50)
    
    # Test single request first
    if not test_single_request():
        print("❌ API is not working. Please check your configuration.")
        return
    
    print("\n" + "=" * 50)
    print("🔍 Testing different concurrency levels...")
    
    # Test different concurrency levels
    levels = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30]
    
    optimal_level = 1
    
    for level in levels:
        print(f"\n📊 Testing concurrency level: {level}")
        success = test_concurrent_requests(level)
        
        if success:
            optimal_level = level
            print(f"   ✅ Level {level} works perfectly")
        else:
            print(f"   ❌ Level {level} has failures")
            break
        
        # Small delay between tests
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print(f"🎯 Optimal concurrency level: {optimal_level}")
    print(f"💡 Recommended settings for tau-bench:")
    print(f"   --max-concurrency {optimal_level}")
    print(f"   --num-episodes 10")
    
    # Conservative recommendation
    conservative_level = max(1, optimal_level - 2)
    print(f"\n🛡️  Conservative recommendation:")
    print(f"   --max-concurrency {conservative_level}")
    print(f"   --num-episodes 10")

if __name__ == "__main__":
    find_optimal_concurrency() 