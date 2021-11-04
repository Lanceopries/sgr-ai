import json
import flask
import pandas as pd
from flask import request
from flask_cors import CORS, cross_origin
import numpy as np
import pickle
from space_bandits import LinearBandits

# global dict for all data
data_dict = {}

HEROKU_ON = True

DATA_LOADED = False

if HEROKU_ON:
    path = ''
else:
    path = 'D:\heroku_test\\'

matching_df = pd.read_csv(path + 'matching_df.csv', header=None)
need_list = matching_df.iloc[1][1:].tolist()
service_list = matching_df[1][1:].tolist()

matching_list = []
matching_dict = {}
for i in range(2, 22):
    for j in range(2, 12):
        if matching_df.iloc[i, j] == '1':
            matching_list += [[i - 1, j - 1]]
            matching_dict[service_list[i - 1]] = need_list[j - 1]


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
    data_dict['corporate_services_df'] = pd.read_csv(path + 'corporate.csv').fillna('NoneType')

    data_dict['engi_centres_services_df'].rename(columns={'Название объекта': 'name', 'Рынок': 'market_type', 'Технологии': 'tech_type', 'Сервисы': 'service'}, inplace=True)
    data_dict['accelerator_services_df'].rename(columns={'Название набора': 'name', 'Рынок': 'market_type', 'Технологии': 'tech_type', 'Сервисы': 'service', 'Стадия стартапа': 'evo_stage'}, inplace=True)
    data_dict['business_incubs_services_df'].rename(columns={'Название объекта': 'name', 'Рынок': 'market_type', 'Технологии': 'tech_type', 'Сервисы': 'service', 'Стадия стартапа': 'evo_stage'}, inplace=True)
    data_dict['institutes_services_df'].rename(columns={'Название объекта': 'name', 'Сервисы': 'service', 'Стадия стартапа': 'evo_stage'}, inplace=True)
    data_dict['pilot_services_df'].rename(columns={'Название объекта': 'name', 'Рынок': 'market_type', 'Технологии': 'tech_type'}, inplace=True)
    data_dict['venture_fond_services_df'].rename(columns={'Название объекта': 'name', 'Рынок': 'market_type', 'Технологии': 'tech_type', 'Сервисы': 'service', 'Стадия стартапа': 'evo_stage'}, inplace=True)
    data_dict['corporate_services_df'].rename(columns={'Коммерческое наименование': 'name', 'Рыночные ниши': 'market_type', 'Технологии': 'tech_type', 'Бизнес-модель': 'b_model'}, inplace=True)

    data_dict['pilot_services_df']['service'] = 'Тестирование продукта'
    data_dict['corporate_services_df']['service'] = 'Инвестиции'

    global bandit_model
    bandit_model = pickle.load(open(path + 'bandit_model.pkl', 'rb'))

    global context_columns
    context_columns = pickle.load(open(path + 'context_columns.pkl', 'rb'))

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

    #TODO сделать кандидатную модель по сервисам через matching_dict
    candidate_filter = data['start_up']['service'][0]
    candidate_services = [k for k, v in matching_dict.items() if v == candidate_filter]

    placeholder_df_dict = {}
    score_series_dict = {}

    for key in data_dict.keys():

        mask = pd.Series(np.zeros(data_dict[key].shape[0])).astype(int)

        for service in candidate_services:
            mask += data_dict[key]['service'].apply(lambda x: x.find(service) >= 0).astype(int)

        mask = mask > 0

        placeholder_df_dict[key] = data_dict[key][mask].copy()
        score_series_dict[key] = pd.Series(np.zeros(placeholder_df_dict[key].shape[0])).astype(int)

    for field in data['start_up'].keys():
        for filter_type in data['start_up'][field]:
            for key in placeholder_df_dict.keys():
                if field in placeholder_df_dict[key].columns.values:
                    score_series_dict[key] += placeholder_df_dict[key][field].apply(lambda x: int(x.find(filter_type) >= 0) / np.log1p(len(x))) * np.log1p(len(placeholder_df_dict[key].columns.values))

    result_df_list = []

    for key in data_dict.keys():
        placeholder_df_dict[key]['rating'] = score_series_dict[key] #/ score_series_dict[key].max()
        placeholder_df_dict[key]['type'] = key
        placeholder_df_dict[key].rename(columns={'Название объекта': 'name'}, inplace=True)

        result_df_list += [placeholder_df_dict[key][['name', 'type', 'rating']]]


    result_df = pd.concat(result_df_list)

    result_list = []

    filtered_result = result_df.sort_values('rating', ascending=False).iloc[:10]
    filtered_result['rating'] = filtered_result['rating'] / filtered_result['rating'].max()
    filtered_result.dropna(inplace=True)

    for index in range(filtered_result.shape[0]):
        elem = filtered_result.iloc[index]
        result_list += [
            {
                'name': elem['name'].encode("utf-8", "ignore").decode('utf-8'),
                'type': elem['type'],
                'rating': elem['rating']
            }
        ]

    return flask.jsonify(result_list)

@app.route('/api/personalrecommend', methods=['POST'])
def personal_query():

    data = request.json

    candidate_filter = data['start_up']['service'][0]
    candidate_services = [k for k, v in matching_dict.items() if v == candidate_filter]

    placeholder_df_dict = {}
    score_series_dict = {}

    for key in data_dict.keys():

        mask = pd.Series(np.zeros(data_dict[key].shape[0])).astype(int)

        for service in candidate_services:
            mask += data_dict[key]['service'].apply(lambda x: x.find(service) >= 0).astype(int)

        mask = mask > 0

        placeholder_df_dict[key] = data_dict[key][mask].copy()
        score_series_dict[key] = pd.Series(np.zeros(placeholder_df_dict[key].shape[0])).astype(int)



    startup_context = np.zeros(len(context_columns))

    bandit_expectations = bandit_model.expected_values(context=startup_context)
    most_expected_actions = np.argsort(bandit_expectations)[-2:]

    #TODO
    context_dict = {
        0: ['venture_fond_services_df']
    }

    for field in data['start_up'].keys():
        for filter_type in data['start_up'][field]:
            for key in placeholder_df_dict.keys():
                if field in placeholder_df_dict[key].columns.values:
                    score_series_dict[key] += placeholder_df_dict[key][field].apply(lambda x: int(x.find(filter_type) >= 0) / np.log1p(len(x))) * np.log1p(len(placeholder_df_dict[key].columns.values))

    result_df_list = []

    for key in data_dict.keys():
        placeholder_df_dict[key]['rating'] = score_series_dict[key] #/ score_series_dict[key].max()
        placeholder_df_dict[key]['type'] = key
        placeholder_df_dict[key].rename(columns={'Название объекта': 'name'}, inplace=True)

        result_df_list += [placeholder_df_dict[key][['name', 'type', 'rating']]]


    result_df = pd.concat(result_df_list)

    result_list = []

    filtered_result = result_df.sort_values('rating', ascending=False).iloc[:10]
    filtered_result['rating'] = filtered_result['rating'] / filtered_result['rating'].max()
    filtered_result.dropna(inplace=True)

    for index in range(filtered_result.shape[0]):
        elem = filtered_result.iloc[index]
        result_list += [
            {
                'name': elem['name'].encode("utf-8", "ignore").decode('utf-8'),
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

