import flask
from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/')
def greetings():
    return "Hello"


if __name__ == "__main__":
    app.run()
