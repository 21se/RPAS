import glob
import os
import re

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model

DB_NAME = 'rpas'
DB_USER = 'postgres'
DB_PWD = '1234'
DB_HOST = 'localhost'


def get_db_handles(dbname, user, password, host):
    connection = psycopg2.connect(dbname=dbname, user=user, password=password, host=host)
    cursor = connection.cursor(cursor_factory=DictCursor)

    return connection, cursor


def load_raw_data(hub, well, parameter):
    conn, cursor = get_db_handles(DB_NAME, DB_USER, DB_PWD, DB_HOST)

    excel_path = cwd + f'/data/Куст {hub}/Скважина {well}.xlsx'
    excel = pd.read_excel(excel_path, engine='openpyxl')
    df = pd.DataFrame(excel).drop(index=0)

    column_name = ''
    for column in df.columns:
        if parameter in column:
            column_name = column
            break

    if column_name == '':
        print('Unknown parameter')
        return

    timestamp_column_name = df.columns[0]

    query = f"""
        INSERT INTO
            hubs(id)
        VALUES
            ({hub})
        ON CONFLICT
            DO NOTHING;
        """
    cursor.execute(query)

    query = f"""
        INSERT INTO
            wells(id, hub_id)
        VALUES
            ({well}, {hub})
        ON CONFLICT
            DO NOTHING;
        """
    cursor.execute(query)

    query = f"""
            SELECT 
                id
            FROM 
                raw_data
            WHERE
                parameter = '{parameter}' AND
                well_id = {well};
            """
    cursor.execute(query)

    record = cursor.fetchone()
    if record:
        return record['id']

    query = f"""
        INSERT INTO 
            raw_data(parameter, well_id)
        VALUES 
            ('{parameter}', {well})
        RETURNING
            id;
        """
    cursor.execute(query)

    data_id = cursor.fetchone()['id']

    for index, row in df.iterrows():
        datetime = row[timestamp_column_name].isoformat(sep=' ')
        value = row[column_name] if not pd.isna(row[column_name]) else 'NULL'

        query = f"""
            INSERT INTO 
                raw_data_values(raw_data_id, datetime, data_value) 
            VALUES
                ({data_id}, '{datetime}', {value});
            """
        cursor.execute(query)

    conn.commit()

    return data_id


def create_data(data_type, raw_data_id, corrections, normalization_value):
    if data_type not in ('train', 'test'):
        print(f'There is no data type for "{data_type}"')
        return

    conn, cursor = get_db_handles(DB_NAME, DB_USER, DB_PWD, DB_HOST)

    query = f"""
            SELECT 
                id
            FROM 
                {data_type}_data
            WHERE
                raw_data_id = '{raw_data_id}' AND
                corrections = {corrections} AND 
                normalization_value = {normalization_value};
            """
    cursor.execute(query)
    record = cursor.fetchone()
    if record:
        return record['id']

    query = f"""
        SELECT
            datetime, data_value
        FROM 
            raw_data_values
        WHERE
            raw_data_id = {raw_data_id};
        """
    cursor.execute(query)

    raw_data_values = cursor.fetchall()
    df = pd.DataFrame(raw_data_values).fillna(value=np.nan).dropna()

    if corrections:
        df = df.drop(df[df[df.columns[1]] < 1].index)
        remove_anomalies(pd.to_numeric(df[df.columns[1]]), drop_level=0.2)

    start = 0
    end = int(len(df) * 0.9)

    array = df[df.columns[1]].values
    array /= normalization_value
    values = array.tolist()[start:end]

    query = f"""
        INSERT INTO 
            {data_type}_data(raw_data_id, corrections, normalization_value)
        VALUES 
            ({raw_data_id}, {corrections}, {normalization_value})
        RETURNING
            id;
    """
    cursor.execute(query)

    data_id = cursor.fetchone()['id']

    for datetime64, value in zip(df[df.columns[0]].values[start:end], values):
        datetime = pd.to_datetime(datetime64).isoformat(sep=' ')

        query = f"""
                INSERT INTO 
                    {data_type}_data_values({data_type}_data_id, datetime, data_value) 
                VALUES
                    ({data_id}, '{datetime}', {value});
                """
        cursor.execute(query)

    conn.commit()

    return data_id


def train_model(network_name, data_id):
    conn, cursor = get_db_handles(DB_NAME, DB_USER, DB_PWD, DB_HOST)
    os.makedirs("db_data/models", exist_ok=True)

    query = f"""
        SELECT 
            id
        FROM 
            models
        WHERE 
            network = '{network_name}' AND 
            train_data_id = {data_id};
        """
    cursor.execute(query)

    record = cursor.fetchone()
    if record:
        return record['id']

    query = f"""
            SELECT
                datetime, data_value
            FROM 
                train_data_values
            WHERE
                train_data_id = {data_id};
            """
    cursor.execute(query)

    train_data_values = cursor.fetchall()

    df = pd.DataFrame(train_data_values)
    train_values = df[df.columns[1]].values.tolist()

    lookback_window = 1
    x = []
    y = []
    for i in range(lookback_window, len(train_values)):
        x.append(train_values[i - lookback_window:i])
        y.append(train_values[i])

    if network_name == 'MLP':
        model = Sequential()
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mae')

        model.fit(x, y, epochs=100, verbose=2)
    else:
        print(f'There is no network structure for "{network_name}"')
        return

    weights_path = f'/db_data/models/{network_name}_{data_id}.h5'
    model.save(cwd + weights_path)

    query = f"""
        INSERT INTO 
            models(train_data_id, network)
        VALUES 
            ({data_id}, '{network_name}')
        RETURNING
            id;
        """
    cursor.execute(query)

    conn.commit()

    return cursor.fetchone()['id']


def create_prediction_data(model, model_id, test_data_id):
    conn, cursor = get_db_handles(DB_NAME, DB_USER, DB_PWD, DB_HOST)

    query = f"""
            SELECT 
                id
            FROM 
                prediction_data
            WHERE
                model_id = {model_id} AND
                test_data_id = {test_data_id};
            """
    cursor.execute(query)

    record = cursor.fetchone()
    if record:
        return record['id']

    query = f"""
            SELECT
                datetime, data_value
            FROM 
                test_data_values
            WHERE
                test_data_id = {test_data_id};
            """
    cursor.execute(query)

    test_data_values = cursor.fetchall()

    df = pd.DataFrame(test_data_values)
    test_values = df[df.columns[1]].values.tolist()

    prediction_values = model.predict(test_values).flatten().tolist()

    query = f"""
        INSERT INTO 
            prediction_data(model_id, test_data_id)
        VALUES 
            ({model_id}, {test_data_id})
        RETURNING
            id;
    """
    cursor.execute(query)

    data_id = cursor.fetchone()['id']

    for datetime64, value in zip(df[df.columns[0]].values, prediction_values):
        datetime = pd.to_datetime(datetime64).isoformat(sep=' ')

        query = f"""
                    INSERT INTO 
                        prediction_data_values(prediction_data_id, datetime, data_value) 
                    VALUES
                        ({data_id}, '{datetime}', {value});
                    """
        cursor.execute(query)

    conn.commit()

    return data_id


def get_model(model_id):
    conn, cursor = get_db_handles(DB_NAME, DB_USER, DB_PWD, DB_HOST)
    query = f"""
        SELECT
            train_data_id, network
        FROM
            models
        WHERE
            id = {model_id};
    """
    cursor.execute(query)

    record = cursor.fetchone()
    if not record:
        return

    train_data_id, network = record['train_data_id'], record['network']
    weights_path = f'{cwd}/db_data/models/{network}_{train_data_id}.h5'

    return load_model(weights_path)


def remove_anomalies(df, neighbours=100, drop_level=0.2):
    for i in range(len(df.values)):
        values = df.values[max(0, i - neighbours):min(len(df.values), i + neighbours)]
        avg = sum(values) / len(values)
        ratio = df.values[i] / avg

        if 1 + drop_level < ratio or ratio < 1 - drop_level:
            df.values[i] = avg


def main():
    os.makedirs("db_data", exist_ok=True)

    parameter = 'Давление до УР'
    network_name = 'MLP'
    corrections = True
    normalization_value = 160
    train_data = (
        ('206', '2061'),
        ('202', '2021'),
        ('209', '2091')
    )

    test_data_list = []
    for filename in glob.glob(f'{cwd}/data/*/*'):
        match = re.match('.+Куст (\d+)\\\\Скважина (\d+)\.xlsx', filename)
        if not match:
            continue
        hub, well = match.groups()

        raw_data_id = load_raw_data(hub, well, parameter)
        test_data_list.append(create_data('test', raw_data_id, corrections, normalization_value))

    # В случае если данные уже загружены возвращает существующий id
    for hub, well in train_data:
        raw_data_id = load_raw_data(hub, well, parameter)
        train_data_id = create_data('train', raw_data_id, corrections, normalization_value)

        model_id = train_model(network_name, train_data_id)
        model = get_model(model_id)

        for test_data_id in test_data_list:
            create_prediction_data(model, model_id, test_data_id)


if __name__ == '__main__':
    cwd = os.getcwd()
    main()
