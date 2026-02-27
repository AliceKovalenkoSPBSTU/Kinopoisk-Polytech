import requests
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()
headers = {
    'X-API-KEY': os.getenv('X-API-KEY')
}
params = {
    'type': 'movie',
    'selectFields': ['id', 'name', 'year', 'rating', 'ageRating', 'votes', 'seasonsInfo', 'budget', 'audience',
                     'seriesLength', 'totalSeriesLength', 'genres', 'countries', 'networks', 'fees',
                     'sequelsAndPrequels', 'updatedAt', 'createdAt'],
    'next': ''
}

req = requests.get('https://api.poiskkino.dev/v1.5/movie?year=2000-2025&rating.kp=1-10', headers=headers,
                         params=params)
req.raise_for_status()
meta_data = req.json()
films = []
films.extend(meta_data['docs'])
next_token = meta_data["next"]
success = 1

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
        time.sleep(3)

    except requests.exceptions.HTTPError as e:
        print(f"HTTP ошибка {e} на {success+1} запросе")
        break
    except Exception as e:
        print(f"Ошибка на {success+1} запросе")
        break

with open(f"output/films_{success}.json", "w", encoding="utf-8") as f:
    json.dump(films, f, indent=4, ensure_ascii=False)