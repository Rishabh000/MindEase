import requests

response = requests.post("http://localhost:8000/chat", json={"message": "I'm feeling happy"})
print(response.json())
