import json
import flask
import pandas as pd
import numpy as np
import sklearn
from flask import request


companies_dict = pd.read_excel('Companies.xlsx', sheet_name=None)
deals_dict = pd.read_excel('Deals.xlsx', sheet_name=None)
services_dict = pd.read_excel('Services.xlsx', sheet_name=None)

main_df = companies_dict['Датасет']



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

