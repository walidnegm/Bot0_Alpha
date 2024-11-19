import requests

response = requests.post(
    "http://127.0.0.1:8080",
    json={
        "prompt": "Hello, this is a test message.",
        "n_predict": 128,
        "temperature": 0.7,
        "top_p": 0.9
    },
    headers={"Content-Type": "application/json"}
)

if response.status_code == 200:
    print("Response:", response.json())
else:
    print(f"Error {response.status_code}: {response.text}")