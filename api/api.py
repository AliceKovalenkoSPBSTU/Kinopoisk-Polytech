import requests
import json
import time
import os
from dotenv import load_dotenv
from tqdm import tqdm

def load_api(success):
    load_dotenv()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    api_keys = []
    i = 0
    while True:
        key = os.getenv(f'X-API-KEY{i}')
        if key is None:
            break
        api_keys.append(key)
        i += 1

    headers = {
        'X-API-KEY': api_keys[0]
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
    success += 1
    films.extend(meta_data['docs'])
    next_token = meta_data["next"]

    if not api_keys:
        raise ValueError("Не найдено ни одного API-ключа")

    print(f"Найдено {len(api_keys)} API-ключей")
    def get_data(next_token, films, headers, params, inner_pbar, success):
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
                inner_pbar.update(1)
                time.sleep(1)

            except requests.exceptions.HTTPError as e:
                print(f"HTTP ошибка {e} на {success+1} запросе")
                break
            except Exception as e:
                print(f"Ошибка {e} на {success+1} запросе")
                break
        return films, next_token, success

    with tqdm(total=len(api_keys), desc="Процесс выполнения", position=0) as global_pbar:
        for idx,key in enumerate (api_keys):
            print(f'\n{"-" * 40}\n')
            print(f'Ключ №{idx + 1}/{len(api_keys)}')

            with tqdm(desc=f"Ключ №{idx + 1}", position=1, leave=False) as inner_pbar:
                films, next_token, success = get_data(next_token, films, headers, params, inner_pbar, success)
            print(f'Совершено запросов: {success}')
            global_pbar.update(1)

            if idx < len(api_keys) - 1:
                headers['X-API-KEY'] = os.getenv(f'X-API-KEY{idx+1}')
                time.sleep(1)

    with open(f"output/films_{success}.json", "w", encoding="utf-8") as f:
        json.dump(films, f, indent=4, ensure_ascii=False)

