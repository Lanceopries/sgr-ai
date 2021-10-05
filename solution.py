import json
import flask
from flask import request

app = flask.Flask(__name__)

@app.route('/ping')
def ping():
    result = {'status': 'ok'}
    return result


@app.route('/query', methods=['POST'])
def query():
    data = json.loads(request.json)

    return data


@app.route('/update_index', methods=['POST'])
def update_index():
    data = json.loads(request.json)

    return data

