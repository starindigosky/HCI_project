import urllib.request
import urllib.parse
import json
import time

BASE_URL = "http://localhost:8000/api/v1"


def test_tts_api_call():
    print("Testing TTS API with Hanzi input '好'...")
    url = f"{BASE_URL}/tts/synthesize"
    data = {
        "text": "你好",  # Translated to "汝好", then should be converted effectively.
        "source_language": "chinese",
        "target_language": "min_nan",
    }
    json_data = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url, data=json_data, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 200:
                print("Success! API returned 200.")
                res = json.loads(response.read().decode("utf-8"))
                print("Response:", res)

                # Check if audio url works
                audio_url = res.get("audio_url")
                if audio_url:
                    full_url = f"http://localhost:8000{audio_url}"
                    print(f"Testing audio download from: {full_url}")
                    with urllib.request.urlopen(full_url) as f:
                        print(f"Download status: {f.status}")
                        print(f"Content length: {len(f.read())}")
            else:
                print(f"Failed with status {response.status}")
                print(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    test_tts_api_call()
