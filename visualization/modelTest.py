import requests
import json

url = "http://localhost:11434/v1/chat/completions"
headers = {"Authorization": "Bearer your_api_key", "Content-Type": "application/json"}
payload = {
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "测试提示"}],
    "max_tokens": 500
}
response = requests.post(url, headers=headers, json=payload)
print(response.json())
