import numpy as np
import pandas as pd
import re
import scipy
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

def train_model():
    # Getting info from URL
    df_1 = pd.read_html(
        "http://www.citystar.ru/detal.htm?d=43&nm=%CE%E1%FA%FF%E2%EB%E5%ED%E8%FF%20-%20%CF%F0%EE%E4%E0%EC%20%EA%E2%E0%F0%F2%E8%F0%F3%20%E2%20%E3.%20%CC%E0%E3%ED%E8%F2%EE%E3%EE%F0%F1%EA%E5")[
        16]
    df_2 = pd.read_html(
        "http://www.citystar.ru/detal.htm?d=43&nm=%CE%E1%FA%FF%E2%EB%E5%ED%E8%FF+%2D+%CF%F0%EE%E4%E0%EC+%EA%E2%E0%F0%F2%E8%F0%F3+%E2+%E3%2E+%CC%E0%E3%ED%E8%F2%EE%E3%EE%F0%F1%EA%E5&pN=2")[
        16]
    df_3 = pd.read_html(
        "http://www.citystar.ru/detal.htm?d=43&nm=%CE%E1%FA%FF%E2%EB%E5%ED%E8%FF+%2D+%CF%F0%EE%E4%E0%EC+%EA%E2%E0%F0%F2%E8%F0%F3+%E2+%E3%2E+%CC%E0%E3%ED%E8%F2%EE%E3%EE%F0%F1%EA%E5&pN=3")[
        16]
    df_4 = pd.read_html(
        "http://www.citystar.ru/detal.htm?d=43&nm=%CE%E1%FA%FF%E2%EB%E5%ED%E8%FF+%2D+%CF%F0%EE%E4%E0%EC+%EA%E2%E0%F0%F2%E8%F0%F3+%E2+%E3%2E+%CC%E0%E3%ED%E8%F2%EE%E3%EE%F0%F1%EA%E5&pN=4")[
        16]
    df_5 = pd.read_html(
        "http://www.citystar.ru/detal.htm?d=43&nm=%CE%E1%FA%FF%E2%EB%E5%ED%E8%FF+%2D+%CF%F0%EE%E4%E0%EC+%EA%E2%E0%F0%F2%E8%F0%F3+%E2+%E3%2E+%CC%E0%E3%ED%E8%F2%EE%E3%EE%F0%F1%EA%E5&pN=5")[
        16]
    df_6 = pd.read_html(
        "http://www.citystar.ru/detal.htm?d=43&nm=%CE%E1%FA%FF%E2%EB%E5%ED%E8%FF+%2D+%CF%F0%EE%E4%E0%EC+%EA%E2%E0%F0%F2%E8%F0%F3+%E2+%E3%2E+%CC%E0%E3%ED%E8%F2%EE%E3%EE%F0%F1%EA%E5&pN=6")[
        16]
    df_7 = pd.read_html(
        "http://www.citystar.ru/detal.htm?d=43&nm=%CE%E1%FA%FF%E2%EB%E5%ED%E8%FF+%2D+%CF%F0%EE%E4%E0%EC+%EA%E2%E0%F0%F2%E8%F0%F3+%E2+%E3%2E+%CC%E0%E3%ED%E8%F2%EE%E3%EE%F0%F1%EA%E5&pN=7")[
        16]
    df_8 = pd.read_html(
        "http://www.citystar.ru/detal.htm?d=43&nm=%CE%E1%FA%FF%E2%EB%E5%ED%E8%FF+%2D+%CF%F0%EE%E4%E0%EC+%EA%E2%E0%F0%F2%E8%F0%F3+%E2+%E3%2E+%CC%E0%E3%ED%E8%F2%EE%E3%EE%F0%F1%EA%E5&pN=8")[
        16]
    df_9 = pd.read_html(
        "http://www.citystar.ru/detal.htm?d=43&nm=%CE%E1%FA%FF%E2%EB%E5%ED%E8%FF+%2D+%CF%F0%EE%E4%E0%EC+%EA%E2%E0%F0%F2%E8%F0%F3+%E2+%E3%2E+%CC%E0%E3%ED%E8%F2%EE%E3%EE%F0%F1%EA%E5&pN=9")[
        16]
    df_10 = pd.read_html(
        "http://www.citystar.ru/detal.htm?d=43&nm=%CE%E1%FA%FF%E2%EB%E5%ED%E8%FF+%2D+%CF%F0%EE%E4%E0%EC+%EA%E2%E0%F0%F2%E8%F0%F3+%E2+%E3%2E+%CC%E0%E3%ED%E8%F2%EE%E3%EE%F0%F1%EA%E5&pN=10")[
        16]

    # Filtering data
    columns_index = [1, 2, 3, 5, 6, 7, 8, 9, 10, 13]
    columns_name = ['date', 'type', 'area', 'floor', 'total_space',
                    'living_space', 'kitchen', 'description', 'price', 'agency']

    flats_df = pd.concat([df_1.loc[2:, columns_index], df_2.loc[2:, columns_index], df_3.loc[2:, columns_index],
                          df_4.loc[2:, columns_index], df_5.loc[2:, columns_index], df_6.loc[2:, columns_index],
                          df_7.loc[2:, columns_index], df_8.loc[2:, columns_index], df_9.loc[2:, columns_index],
                          df_10.loc[2:, columns_index]], ignore_index=True)
    flats_df.columns = columns_name
    flats_df = flats_df.dropna(how='all').reset_index(drop=True)

    # Model building
    X = preproseccing_data(flats_df)
    y = flats_df['price'].apply(lambda x: float(x))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = RidgeCV(alphas=(0.1, 1.0, 10.0))
    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    joblib.dump(model, 'flats_price.model')
    print('Model Training Finished.\n\tTrain MSE: {}\n\tTest MSE: {}'
          .format(round(mean_squared_error(y_train, train_preds)), round(mean_squared_error(y_test, test_preds))))


def preproseccing_data(flats_info_df):
    # Dates
    flats_info_df['day'] = flats_info_df['date'].apply(lambda x: int(x[:2]))
    flats_info_df['month'] = flats_info_df['date'].apply(lambda x: int(x[3:5]))

    # Type
    flats_info_df.dropna(subset=['type'], inplace=True)
    flats_info_df['rooms_count'] = flats_info_df.apply(rooms_count, axis=1)

    # Categorial features
    vectorizer_feats = DictVectorizer()
    feats = ['type', 'area', 'agency']
    flats_feats = vectorizer_feats.fit_transform(flats_info_df[feats].fillna('-').T.to_dict().values())

    # Floor
    floor_pattern = re.compile('(\d+)[/]')
    amount_floor_pattern = re.compile('[/](\d+)')
    flats_info_df['floor_amount'] = flats_info_df.floor.apply(lambda x: int(amount_floor_pattern.findall(x)[0]))
    flats_info_df['floor'] = flats_info_df.floor.apply(lambda x: int(floor_pattern.findall(x)[0]))
    flats_info_df['floor_perc'] = flats_info_df['floor'] / flats_info_df['floor_amount']

    # Flat space
    flats_info_df['total_space'] = flats_info_df['total_space'].apply(lambda x: float(x))
    flats_info_df['living_space'] = flats_info_df['living_space'].apply(lambda x: float(x))
    flats_info_df['kitchen'] = flats_info_df['kitchen'].apply(lambda x: float(x))

    # Description
    vectorizer_description = TfidfVectorizer(ngram_range=(1, 3))
    flats_description = vectorizer_description.fit_transform(flats_info_df['description'].fillna('-'))

    df_columns = ['day', 'month', 'floor', 'floor_perc', 'rooms_count', 'total_space', 'living_space', 'kitchen']
    #X = scipy.sparse.hstack([np.matrix(flats_info_df[df_columns]), flats_feats, flats_description])
    X = flats_info_df[df_columns]
    return X


def rooms_count(row):
    rooms_list = list()
    if 'Однокомнатная' in row['type']:
        return 1
    elif 'Двухкомнатная' in row['type']:
        return 2
    elif 'Трехкомнатная' in row['type']:
        return 3
    elif 'Четырехкомнатная' in row['type']:
        return 4
    elif 'Многокомнатная' in row['type']:
        return 5