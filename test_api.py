import requests
import json

url = "http://localhost:9000/translate"
payload = {
    "text": "I eat rice",
    "source_language": "English",
    "target_language": "Bangla"
}

response = requests.post(url, json=payload)
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
