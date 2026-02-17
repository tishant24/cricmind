# scripts/first_api_call.py
"""
Your first API call - Understanding the basics
"""

import requests
import json
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

def first_api_call():
    """Make your first API call and understand the response"""
    
    print("="*70)
    print("🏏 YOUR FIRST API CALL - LEARNING STEP BY STEP")
    print("="*70)
    
    # Step 1: Get API key from environment
    api_key = os.getenv('CRICAPI_KEY')
    print(f"\n📋 STEP 1: API Key loaded")
    print(f"   Key (masked): {api_key[:8]}...{api_key[-8:]}")
    
    # Step 2: Prepare the URL and parameters
    base_url = "https://api.cricapi.com/v1/currentMatches"
    params = {
        'apikey': api_key,
        'offset': 0
    }
    
    print(f"\n📋 STEP 2: Request prepared")
    print(f"   URL: {base_url}")
    print(f"   Parameters: {params}")
    
    # Step 3: Make the HTTP GET request
    print(f"\n📋 STEP 3: Sending HTTP GET request...")
    print(f"   ⏳ Waiting for response from Cricket Data servers...")
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        
        print(f"   ✅ Response received!")
        
        # Step 4: Check response status
        print(f"\n📋 STEP 4: Analyzing response")
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print(f"   ✅ SUCCESS! (200 = OK)")
        elif response.status_code == 401:
            print(f"   ❌ UNAUTHORIZED (401 = Wrong API key)")
            return
        else:
            print(f"   ⚠️ Unexpected status: {response.status_code}")
            return
        
        # Step 5: Parse JSON response
        print(f"\n📋 STEP 5: Converting JSON to Python dictionary")
        data = response.json()
        
        print(f"   ✅ JSON parsed successfully")
        print(f"   Response structure: {list(data.keys())}")
        
        # Step 6: Explore the data
        print(f"\n📋 STEP 6: Exploring the data structure")
        
        if 'data' in data:
            matches = data['data']
            print(f"   📊 Total matches found: {len(matches)}")
            
            if len(matches) > 0:
                # Show first match details
                print(f"\n" + "="*70)
                print(f"📱 FIRST MATCH DETAILS:")
                print("="*70)
                
                first_match = matches[0]
                
                # Print all fields in first match
                for key, value in first_match.items():
                    print(f"   {key:20s}: {value}")
                
                # Step 7: Extract specific information
                print(f"\n" + "="*70)
                print(f"📊 EXTRACTED KEY INFORMATION:")
                print("="*70)
                
                for i, match in enumerate(matches[:5], 1):  # First 5 matches
                    print(f"\n{i}. Match ID: {match.get('id', 'N/A')}")
                    print(f"   Name: {match.get('name', 'N/A')}")
                    print(f"   Type: {match.get('matchType', 'N/A')}")
                    print(f"   Status: {match.get('status', 'N/A')}")
                    
                    # Teams
                    teams = match.get('teams', [])
                    if teams:
                        print(f"   Teams: {' vs '.join(teams)}")
                    
                    # Venue
                    venue = match.get('venue', 'N/A')
                    print(f"   Venue: {venue}")
        
        # Step 8: Save raw response for learning
        print(f"\n📋 STEP 8: Saving response to file")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'data/raw/learning_first_api_call_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        print(f"   ✅ Saved to: {filename}")
        print(f"   👀 Open this file in VSCode to see the full structure!")
        
        # Summary
        print(f"\n" + "="*70)
        print(f"✅ API CALL COMPLETE - SUMMARY")
        print("="*70)
        print(f"   • Made HTTP GET request")
        print(f"   • Received status code: {response.status_code}")
        print(f"   • Parsed JSON response")
        print(f"   • Found {len(matches)} matches")
        print(f"   • Saved data to file")
        
        print(f"\n🎓 WHAT YOU LEARNED:")
        print(f"   1. How to load API keys from .env")
        print(f"   2. How to make HTTP GET requests")
        print(f"   3. How to check response status codes")
        print(f"   4. How to parse JSON data")
        print(f"   5. How to extract information from nested data")
        print(f"   6. How to save data to files")
        
        return data
        
    except requests.exceptions.Timeout:
        print(f"   ❌ Request timed out (server too slow)")
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Connection error (check internet)")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")

if __name__ == "__main__":
    first_api_call()