import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()
headers = {
    'X-API-KEY': os.getenv('X-API-KEY0')
}
params = {
    'type': 'movie',
    'selectFields': ['id', 'name', 'year', 'rating', 'ageRating', 'votes', 'seasonsInfo', 'budget', 'audience',
                     'seriesLength', 'totalSeriesLength', 'genres', 'countries', 'networks', 'fees',
                     'sequelsAndPrequels', 'updatedAt', 'createdAt'],
    'next': ''
}

def get_data(next_token, films, headers, params):
    global success
    films = films
    while next_token:
        try:
            params['next'] = next_token
            req_i = requests.get('https://api.poiskkino.dev/v1.5/movie?year=2000-2025&rating.kp=1-10', headers=headers,
                                       params=params)
            req_i.raise_for_status()
            meta_data_i = req_i.json()
            films.append(meta_data_i["docs"])
            next_token = meta_data_i["next"]
            success += 1
            time.sleep(1)

        except requests.exceptions.HTTPError as e:
            print(f"HTTP ошибка {e} на {success+1} запросе")
            break
        except Exception as e:
            print(f"Ошибка {e} на {success+1} запросе")
            break
    return films, next_token

req = requests.get('https://api.poiskkino.dev/v1.5/movie?year=2000-2025&rating.kp=1-10', headers=headers,
                         params=params)
req.raise_for_status()
meta_data = req.json()
success = 1
films = []
films.extend(meta_data['docs'])
next_token = meta_data["next"]

key_count = sum(1 for line in open('.env', 'r'))
for key in range (key_count):
    print(f'-----------\nКлюч №{key+1}')
    films, next_token = get_data(next_token, films, headers, params)
    print(f'Совершено запросов: {success}')
    headers['X-API-KEY'] = os.getenv(f'X-API-KEY{key+1}')

with open(f"output/films_{success}.json", "w", encoding="utf-8") as f:
    json.dump(films, f, indent=4, ensure_ascii=False)

