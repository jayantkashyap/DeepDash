import requests

# json_data = {"time": "6"}

response = requests.get("http://localhost:5000/predict")

print(response.text)
