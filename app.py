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

    data_dict['engi_centres_services_df'] = pd.read_csv(path + 'engi_centres_services.csv').fillna('NoneType')
    data_dict['accelerator_services_df'] = pd.read_csv(path + 'accelerators_services.csv').fillna('NoneType')
    data_dict['business_incubs_services_df'] = pd.read_csv(path + 'business_incubs.csv').fillna('NoneType')
    data_dict['institutes_services_df'] = pd.read_csv(path + 'institutes.csv').fillna('NoneType')
    data_dict['pilot_services_df'] = pd.read_csv(path + 'pilot.csv').fillna('NoneType')
    data_dict['venture_fond_services_df'] = pd.read_csv(path + 'venture_fond_services.csv').fillna('NoneType')

    data_dict['engi_centres_services_df'].rename(columns={'Название объекта': 'name', 'Рынок': 'market_type', 'Технологии': 'tech_type', 'Сервисы': 'service'}, inplace=True)
    data_dict['accelerator_services_df'].rename(columns={'Название набора': 'name', 'Рынок': 'market_type', 'Технологии': 'tech_type', 'Сервисы': 'service'}, inplace=True)
    data_dict['business_incubs_services_df'].rename(columns={'Название объекта': 'name', 'Рынок': 'market_type', 'Технологии': 'tech_type', 'Сервисы': 'service'}, inplace=True)
    data_dict['institutes_services_df'].rename(columns={'Название объекта': 'name', 'Сервисы': 'service'}, inplace=True)
    data_dict['pilot_services_df'].rename(columns={'Название объекта': 'name', 'Рынок': 'market_type', 'Технологии': 'tech_type'}, inplace=True)
    data_dict['venture_fond_services_df'].rename(columns={'Название объекта': 'name', 'Рынок': 'market_type', 'Технологии': 'tech_type', 'Сервисы': 'service'}, inplace=True)

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

    placeholder_df_dict = {}
    score_series_dict = {}

    for key in data_dict.keys():
        placeholder_df_dict[key] = data_dict[key].copy()
        score_series_dict[key] = pd.Series(np.zeros(data_dict[key].shape[0])).astype(int)

    for field in data['start_up'].keys():
        for filter_type in data['start_up'][field]:
            for key in data_dict.keys():
                if field in data_dict[key].columns.values:
                    score_series_dict[key] += data_dict[key][field].apply(lambda x: x.find(filter_type) >= 0).astype(int)

    result_df_list = []

    for key in data_dict.keys():
        placeholder_df_dict[key]['rating'] = score_series_dict[key] / score_series_dict[key].max()
        placeholder_df_dict[key]['type'] = key
        placeholder_df_dict[key].rename(columns={'Название объекта': 'name'}, inplace=True)

        result_df_list += [placeholder_df_dict[key][['name', 'type', 'rating']]]


    result_df = pd.concat(result_df_list)

    result_list = []

    filtered_result = result_df.sort_values('rating', ascending=False).iloc[:10]

    for index in range(filtered_result.shape[0]):
        elem = filtered_result.iloc[index]
        result_list += [
            {
                'name': elem['name'],
                'type': elem['type'],
                'rating': elem['rating']
            }
        ]

    return flask.jsonify(result_list)


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

