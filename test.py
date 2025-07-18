import requests

# ✅ Your Firebase Function URL
FIREBASE_FUNCTION_URL = "https://europe-west4-aimanagerfirebasebackend.cloudfunctions.net/device_heartbeat"

# ✅ Replace with actual user_id & device_id
payload = {
    "user_id": "3W50EJGZZ3hMxMk9RwOBEGgdOak2",
    "device_id": "1738976964995"
}

headers = {
    "Content-Type": "application/json"
}

print("📡 Sending test request to Firebase function...")
response = requests.post(FIREBASE_FUNCTION_URL, json=payload, headers=headers)

print(f"🔄 HTTP Status Code: {response.status_code}")
print(f"✅ Response: {response.text}")
