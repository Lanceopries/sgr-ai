import json
import flask
import pandas as pd
from flask import request
from flask_cors import CORS, cross_origin
import numpy as np

# global dict for all data
data_dict = {}

HEROKU_ON = True

DATA_LOADED = False

app = flask.Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

# A welcome message to test our server
@app.route('/api/')
def index():
    return "<h1>Dragons Recommend System for Startups</h1>"


# initialization
@app.route('/api/init')
def init_data():

    global HEROKU_ON
    global DATA_LOADED

    if HEROKU_ON:
        path = ''
    else:
        path = 'D:\heroku_test\\'

    data_dict['engi_centres_services_df'] = pd.read_csv(path + 'engi_centres_services.csv')
    data_dict['accelerator_services_df'] = pd.read_csv(path + 'accelerators_services.csv')
    data_dict['business_incubs_services_df'] = pd.read_csv(path + 'business_incubs.csv')
    data_dict['institutes_services_df'] = pd.read_csv(path + 'institutes.csv')
    data_dict['pilot_services_df'] = pd.read_csv(path + 'pilot.csv')
    data_dict['venture_fond_services_df'] = pd.read_csv(path + 'venture_fond_services.csv')

    DATA_LOADED = True

    result = {'status': 'ok'}
    return result


@app.route('/api/ping')
def ping():

    global DATA_LOADED

    if DATA_LOADED:
        result = {'status': 'ok'}
    else:
        result = {'status': 'data not loaded'}
    return result


# 'Сервис',
# 'Дата основания',
# "Фильтр 'Рынок' для Инновационных компаний",
# "Фильтр 'Технологии' для Инновационных компаний",
# 'Бизнес-модель для Инновационных компаний'
@app.route('/api/easyrecommend', methods=['POST'])
def query():
    data = request.json

    engi_score = pd.Series(np.zeros(len(data_dict['engi_centres_services_df']['Рынок']))).astype(int)

    for market_type in data['start_up']['market_type']:
        engi_score += data_dict['engi_centres_services_df']['Рынок'].apply(lambda x: x.find(market_type) >= 0).astype(int)

    placeholder_copy = data_dict['engi_centres_services_df'].copy()
    placeholder_copy['rating'] = engi_score

    filtered_result = placeholder_copy.sort_values('rating', ascending=False).iloc[:10]
    if filtered_result.size >= 0:
        result = filtered_result['Название объекта'].to_numpy().tolist()
    else:
        result = 'Рекомендаций нет'

    return flask.jsonify(result)


# 'Сервис',
# 'Дата основания',
# 'Технологическая ниша компании для Инновационных компаний',
# 'Стадия развития компании для Инновационных компаний',
# "Фильтр 'Рынок' для Инновационных компаний",
# "Фильтр 'Технологии' для Инновационных компаний",
# 'Бизнес-модель для Инновационных компаний',
# 'основной ОКВЭД',
# 'МСП да/нет',
# 'Категория МСП',
# 'Закупки / контракты компании',
# 'Патенты компании',
# 'резидент технопарков',
# 'Продукты компании',
# 'Экспортер',
# 'Участник инновационного кластера города Москвы',
# 'Участник Сколково',
# 'Организация аккредитована на Бирже контрактного производства',
# ' Опубликован на Навигаторе по стартап-экосистеме Москвы (navigator.innoagency.ru/main/list-company)',
# 'Инновационная компания',
# 'Стартап'
# @app.route('/api/personrecommend', methods=['POST'])
# def update_index():
#     data = json.loads(request.json)
#
#     return data



if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)

