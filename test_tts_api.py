import urllib.request
import urllib.parse
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_tts():
    print("Testing TTS API...")
    url = f"{BASE_URL}/tts/synthesize"
    data = {
        "text": "你好",
        "source_language": "chinese",
        "target_language": "min_nan"
    }
    json_data = json.dumps(data).encode('utf-8')
    
    req = urllib.request.Request(url, data=json_data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req) as response:
            if response.status != 200:
                print(f"Error: API returned status {response.status}")
                return
                
            response_body = response.read().decode('utf-8')
            data = json.loads(response_body)
            print("TTS Response:", json.dumps(data, indent=2))
            
            audio_url = data.get("audio_url")
            if audio_url:
                full_url = f"http://localhost:8000{audio_url}"
                print(f"Checking audio file at: {full_url}")
                
                try:
                    with urllib.request.urlopen(full_url) as audio_response:
                        if audio_response.status == 200:
                            print("Success! Audio file exists and is accessible.")
                            content = audio_response.read()
                            print(f"Audio file size: {len(content)} bytes")
                        else:
                            print(f"Failed to fetch audio file. Status: {audio_response.status}")
                except urllib.error.HTTPError as e:
                    print(f"Failed to fetch audio file: {e}")
            else:
                print("No audio_url in response")

    except urllib.error.URLError as e:
        print(f"Connection error: {e}. Is the server running?")
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_tts()
