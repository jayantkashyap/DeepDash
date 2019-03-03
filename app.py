from flask import Flask, request, jsonify
from models.handle import handle
app = Flask(__name__)


@app.route('/getStatus', methods=['POST'])
def get_status():
    json_data = request.get_json()
    message = json_data["msg"]
    print(message)
    return ""


@app.route('/', methods=['GET', 'POST'])
def wait():
    json_data = request.get_json()
    time = json_data['time']
    return handle(time)


if __name__ == "__main__":
    app.run()
