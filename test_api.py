import requests

# json_data = {"time": "1"}

# response = requests.get("http://localhost:5000/", json=json_data)

for i in range(10):
    json_data = {"msg": f"{i}"}
    response = requests.post(
        "http://localhost:5000/train")

print(response.text)
