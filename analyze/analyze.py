import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

json_path = '../api/output/films_316.json'

with open(json_path, 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

films_list = []
for item in raw_data:
    if isinstance(item, dict):
        films_list.append(item)
    elif isinstance(item, list):
        for film in item:
            if isinstance(film, dict):
                films_list.append(film)

df = pd.DataFrame(films_list)
print(f"Загружено фильмов: {df.shape[0]}")

if 'rating' in df.columns:
    df = pd.concat([df.drop(columns=['rating']), pd.json_normalize(df['rating']).add_prefix('rating_')], axis=1)
if 'fees' in df.columns:
    df = pd.concat([df.drop(columns=['fees']), pd.json_normalize(df['fees']).add_prefix('fees_')], axis=1)
if 'budget' in df.columns:
    df = pd.concat([df.drop(columns=['budget']), pd.json_normalize(df['budget']).add_prefix('budget_')], axis=1)
if 'votes' in df.columns:
    df = pd.concat([df.drop(columns=['votes']), pd.json_normalize(df['votes']).add_prefix('votes_')], axis=1)

if 'genres' in df.columns:
    df['genres_str'] = df['genres'].apply(lambda x: ', '.join([g.get('name','') for g in x]) if isinstance(x, list) else '')
    df['genre_count'] = df['genres'].apply(lambda x: len(x) if isinstance(x, list) else 0)
if 'countries' in df.columns:
    df['countries_str'] = df['countries'].apply(lambda x: ', '.join([c.get('name','') for c in x]) if isinstance(x, list) else '')
if 'audience' in df.columns:
    df['audience_total'] = df['audience'].apply(lambda x: sum(i.get('value',0) for i in x) if isinstance(x, list) else 0)
if 'sequelsAndPrequels' in df.columns:
    df['has_sequel'] = df['sequelsAndPrequels'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)

df.columns = [col.replace('.', '_') for col in df.columns]

rating_col = 'rating_kp'
commercial_col = 'fees_world_value'

df = df.dropna(subset=[rating_col])
df = df[df[rating_col] > 0]
if commercial_col in df.columns:
    df[commercial_col] = df[commercial_col].fillna(0)

if 'fees_world_currency' in df.columns and not df['fees_world_currency'].dropna().empty:
    main_currency = df['fees_world_currency'].mode()[0]
else:
    main_currency = 'RUB'
df['fees_world_mln'] = df[commercial_col] / 1_000_000
commercial_plot_col = 'fees_world_mln'
fee_label = f'Сборы в мире (млн {main_currency})'

os.makedirs('output/plots', exist_ok=True)

col_ru = {
    'rating_kp': 'Рейтинг Кинопоиска',
    'fees_world_mln': fee_label,
    'year': 'Год выхода',
    'ageRating': 'Возрастной рейтинг',
    'genre_count': 'Количество жанров',
    'has_sequel': 'Есть сиквел или приквел',
    'votes_kp': 'Голоса на Кинопоиске',
    'audience_total': 'Общее количество зрителей',
    'budget_value': 'Бюджет (млн)',
    'votes_imdb': 'Голоса IMDb',
    'rating_imdb': 'Рейтинг IMDb',
    'totalSeriesLength': 'Общая длительность сериала',
    'rating_filmCritics': 'Рейтинг кинокритиков',
    'rating_russianFilmCritics': 'Рейтинг российских критиков',
    'rating_tmdb': 'Рейтинг TMDB',
    'votes_filmCritics': 'Голоса кинокритиков',
    'votes_russianFilmCritics': 'Голоса российских критиков',
    'votes_tmdb': 'Голоса TMDB',
    'fees_world_value': 'Сборы в мире',
    'fees_russia_value': 'Сборы в России',
    'fees_usa_value': 'Сборы в США',
    'cluster': 'Кластер'
}

plt.style.use('seaborn-v0_8')
print("\n1. Кластеризация фильмов по показателям рейтинга и коммерческого успеха:")
features = df[[rating_col, commercial_plot_col]].dropna()
if len(features) > 100:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df.loc[features.index, 'cluster'] = kmeans.fit_predict(scaled)

    cluster_stats = df.groupby('cluster')[[rating_col, commercial_plot_col]].mean().round(2)
    cluster_stats.columns = ['Средний рейтинг', fee_label]
    #print(cluster_stats.to_string())

    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(data=df, x=rating_col, y=commercial_plot_col,
                              hue='cluster', palette='viridis', alpha=0.85, s=70)
    plt.title(' 1. Кластеризация фильмов по рейтингу и коммерческому успеху\n', fontsize=15, pad=25)
    plt.xlabel(col_ru['rating_kp'], fontsize=13)
    plt.ylabel(col_ru['fees_world_mln'], fontsize=13)
    plt.legend(title='Кластеры фильмов', title_fontsize=12)
    plt.grid(True, alpha=0.3)

    description = "Кластер 0: Низкий рейтинг и малый бюджет\n" \
                  "Кластер 1: Средний рейтинг и средний бюджет \n" \
                  "Кластер 2: Высокий рейтинг и низкий бюджет\n" \
                  "Кластер 3: Высокий рейтинг и высокий бюджет"
    plt.figtext(0.5, 0.01, description, ha='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig('output/plots/task1_klaster.png', dpi=300, bbox_inches='tight')
    plt.show()

print("2. Зависимость рейтинга фильма от различных факторов:")

df_numeric = df.select_dtypes(include=[np.number]).copy()
df_numeric = df_numeric.loc[:, df_numeric.std() > 0.01]
df_numeric = df_numeric.rename(columns={col: col_ru.get(col, col) for col in df_numeric.columns})

plt.figure(figsize=(16, 12))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt='.2f',
            linewidths=0.5, annot_kws={"size": 9})
plt.title('2.1 Зависимость рейтинга фильма от различных факторов.\nКорреляционная матрица всех значимых факторов\n'
          , fontsize=15, pad=20)
plt.tight_layout()
plt.savefig('output/plots/task2_1_korr.png', dpi=300, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.scatterplot(data=df, x='year', y=rating_col, ax=axes[0,0], alpha=0.7)
axes[0,0].set_title('Рейтинг в зависимости от года выхода')
axes[0,0].set_xlabel(col_ru['year'])
axes[0,0].set_ylabel(col_ru['rating_kp'])

sns.boxplot(data=df, x='ageRating', y=rating_col, ax=axes[0,1])
axes[0,1].set_title('Рейтинг по возрастному рейтингу')
axes[0,1].set_xlabel(col_ru['ageRating'])
axes[0,1].set_ylabel(col_ru['rating_kp'])

sns.boxplot(data=df, x='genre_count', y=rating_col, ax=axes[1,0])
axes[1,0].set_title('Рейтинг в зависимости от количества жанров')
axes[1,0].set_xlabel(col_ru['genre_count'])
axes[1,0].set_ylabel(col_ru['rating_kp'])

sns.boxplot(data=df, x='has_sequel', y=rating_col, ax=axes[1,1])
axes[1,1].set_title('Рейтинг: есть ли сиквел или приквел')
axes[1,1].set_xlabel(col_ru['has_sequel'])
axes[1,1].set_ylabel(col_ru['rating_kp'])

plt.suptitle('2.2 Зависимость рейтинга фильма от различных факторов.\n'
             'Основные факторы, влияющие на рейтинг фильма', fontsize=14)
plt.tight_layout()
plt.savefig('output/plots/task2_2_rating_factors.png', dpi=300, bbox_inches='tight')
plt.show()

print("3. Зависимость коммерческого успеха фильма от различных факторов:")
if commercial_plot_col in df.columns:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.scatterplot(data=df, x='year', y=commercial_plot_col, ax=axes[0,0], alpha=0.7)
    axes[0,0].set_title('Сборы в зависимости от года')
    axes[0,0].set_xlabel(col_ru['year'])
    axes[0,0].set_ylabel(col_ru['fees_world_mln'])

    sns.boxplot(data=df, x='ageRating', y=commercial_plot_col, ax=axes[0,1])
    axes[0,1].set_title('Сборы по возрастному рейтингу')
    axes[0,1].set_xlabel(col_ru['ageRating'])
    axes[0,1].set_ylabel(col_ru['fees_world_mln'])

    sns.boxplot(data=df, x='has_sequel', y=commercial_plot_col, ax=axes[1,0])
    axes[1,0].set_title('Сборы: есть ли сиквел или приквел')
    axes[1,0].set_xlabel(col_ru['has_sequel'])
    axes[1,0].set_ylabel(col_ru['fees_world_mln'])

    sns.scatterplot(data=df, x=rating_col, y=commercial_plot_col, ax=axes[1,1], alpha=0.7)
    axes[1,1].set_title('Сборы vs Рейтинг')
    axes[1,1].set_xlabel(col_ru['rating_kp'])
    axes[1,1].set_ylabel(col_ru['fees_world_mln'])

    plt.suptitle('3. Зависимость коммерческого успеха фильма от различных факторов', fontsize=14)
    plt.tight_layout()
    plt.savefig('output/plots/task3_fee_factors.png', dpi=300, bbox_inches='tight')
    plt.show()

print("4. Зависимость рейтинга и сборов фильма от времени выхода:")
yearly = df.groupby('year').agg({
    rating_col: 'mean',
    commercial_plot_col: 'mean',
    'id': 'count'
}).reset_index()
yearly.columns = ['year', 'mean_rating', 'mean_fees_mln', 'count']

fig, ax1 = plt.subplots(figsize=(13, 7))
ax1.plot(yearly['year'], yearly['mean_rating'], color='blue', marker='o', label='Средний рейтинг')
ax1.set_xlabel('Год выхода')
ax1.set_ylabel('Средний рейтинг Кинопоиска', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(yearly['year'], yearly['mean_fees_mln'], color='green', marker='s', label='Средние сборы')
ax2.set_ylabel(fee_label, color='green')
ax2.tick_params(axis='y', labelcolor='green')

plt.title('4. Зависимости рейтинга и сборов фильма от времени выхода за 2000–2025 годы', fontsize=14)
fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.9))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/plots/task4_dynamics_years.png', dpi=300, bbox_inches='tight')
plt.show()

print("5. Прогнозирование рейтинга фильма на основе его характеристик:")

features = ['year', 'ageRating', 'genre_count', 'has_sequel', 'votes_kp', 'audience_total']
features = [f for f in features if f in df.columns]

X = df[features].fillna(0)
y = df[rating_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
print(f"Точность модели R²: {r2:.3f} ")

plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.75, s=60, color='royalblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=3,
         label='Совпадение')
plt.xlabel('Реальный рейтинг Кинопоиска')
plt.ylabel('Предсказанный рейтинг')
plt.title('5.1 Прогнозирование рейтинга фильма\n'
          f'Сравнение реального и предсказанного значения (R² = {r2:.3f})', fontsize=14, pad=20)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/plots/task5_1_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

coef = pd.Series(model.coef_, index=features)
coef_ru = coef.rename(index=col_ru).sort_values()

plt.figure(figsize=(10, 7))
coef_ru.plot(kind='barh', color='skyblue')
plt.title('5.2 Влияние факторов на предсказанный рейтинг фильма\n', fontsize=14)
plt.xlabel('Коэффициент влияния')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/plots/task5_2_factors_influence.png', dpi=300, bbox_inches='tight')
plt.show()
