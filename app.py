import json
import flask
from flask import request

app = flask.Flask(__name__)

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Dragons Recommend System for Startups</h1>"

@app.route('/ping')
def ping():
    result = {'status': 'ok'}
    return result


@app.route('/easyrecommend', methods=['POST'])
def query():
    data = json.loads(request.json)

    return data


@app.route('/personrecommend', methods=['POST'])
def update_index():
    data = json.loads(request.json)

    return data

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)

