import json
import flask
import pandas as pd
import numpy as np
from flask import request
import os

#
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))

# companies_dict = pd.read_excel('D:\heroku_test\Companies.xlsx', sheet_name=None)
# deals_dict = pd.read_excel('D:\heroku_test\Deals.xlsx', sheet_name=None)
# services_dict = pd.read_excel('D:\heroku_test\Services.xlsx', sheet_name=None)

companies_dict = pd.read_excel('Companies.xlsx', sheet_name=None)
deals_dict = pd.read_excel('Deals.xlsx', sheet_name=None)
services_dict = pd.read_excel('Services.xlsx', sheet_name=None)

main_df = companies_dict['Датасет']

engi_centres_services_df = services_dict['Список инжиниринговых центров'][['Название объекта', 'Рынок', 'Технологии', 'Сервисы']]

app = flask.Flask(__name__)

# A welcome message to test our server
@app.route('/')
def index():
    return "<h1>Dragons Recommend System for Startups</h1>"

@app.route('/ping')
def ping():
    result = {'status': 'ok'}
    return result


@app.route('/test_df')
def test_df():
    result = {'status': 'ok'}
    return result


# 'Сервис',
# 'Дата основания',
# "Фильтр 'Рынок' для Инновационных компаний",
# "Фильтр 'Технологии' для Инновационных компаний",
# 'Бизнес-модель для Инновационных компаний'
@app.route('/easyrecommend', methods=['POST'])
def query():
    data = json.loads(request.json)
    mask = engi_centres_services_df['Рынок'].apply(lambda x: x.find('Healthcare') >= 0)
    filtered_result = engi_centres_services_df[mask]
    if filtered_result.size >= 0:
        result = engi_centres_services_df[mask]['Название объекта'].to_numpy().tolist()
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
@app.route('/personrecommend', methods=['POST'])
def update_index():
    data = json.loads(request.json)

    return data

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)

