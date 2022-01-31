from glob import glob
import xlsxwriter
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error

PATH = os.getcwd()
# DB_PATH = PATH + '/db_data/rpas_t.accdb'
NORM_VALUE = 160
TRAIN_KUST = 202
TRAIN_SKV = 2


def train_model():
    df = pd.DataFrame(
        pd.read_excel(PATH + '/data/Куст {}'.format(TRAIN_KUST) + '/Скважина {}{}.xlsx'.format(TRAIN_KUST, TRAIN_SKV),
                      engine='openpyxl')).drop(
        index=0).dropna()
    df = df.drop(
        df[(df[df.columns[1]] < 1) | (df[df.columns[2]] < 1) | (df[df.columns[3]] < 1) | (df[df.columns[4]] < 1)].index)

    for i in range(1, 5):
        remove_anomalies(pd.to_numeric(df[df.columns[i]]), drop_level=0.2)

    array = df[df.columns[1]].values
    df = array.tolist()
    train_df = df[0:int(len(df) * 0.9)]

    lookback_window = 1
    x, y = [], []
    for i in range(lookback_window, len(train_df)):
        x.append(train_df[i - lookback_window:i])
        y.append(train_df[i])

    x = np.array(x)
    y = np.array(y)

    x = x / NORM_VALUE
    y = y / NORM_VALUE

    model = Sequential()
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')

    print('Train...')
    model.fit(x, y, epochs=100, verbose=2)
    print('Finished')

    return model


def remove_anomalies(df, neighbours=100, drop_level=0.2):
    for i in range(len(df.values)):
        values = df.values[max(0, i - neighbours):min(len(df.values), i + neighbours)]
        avg = sum(values) / len(values)
        ratio = df.values[i] / avg

        if 1 + drop_level < ratio or ratio < 1 - drop_level:
            df.values[i] = avg


def predict(model, model_name, drop_anomalies=True, ):
    for dir in glob(PATH + '/data/Куст*'):
        for filename in glob(dir + '/Скважина*'):
            if 'предсказание' in filename or '.png' in filename:
                continue

            filename = filename.replace('\\', '/')
            filename_list = filename.split('/')
            cust = filename_list[-2]
            skv = filename_list[-1].split('.')[0]

            workbook_paths = {'source': f'{PATH}/db_data/source/',
                              'train': f'{PATH}/db_data/train/',
                              'predicted': f'{PATH}/db_data/predicted/{model_name}'}

            workbooks = {}
            for key in workbook_paths:
                dir = workbook_paths[key] + f'/{cust}'
                os.makedirs(dir, exist_ok=True)
                workbooks[key] = xlsxwriter.Workbook(dir + '/' + skv.replace(" ", "_") + '_' + key + '.xlsx', )

            p_df = pd.DataFrame(pd.read_excel(filename, engine='openpyxl')).drop(index=0).dropna()
            p_df = p_df.drop(p_df[(p_df[p_df.columns[1]] < 1) | (p_df[p_df.columns[2]] < 1) | (
                        p_df[p_df.columns[3]] < 1) | (p_df[p_df.columns[4]] < 1)].index)

            for i in range(1, 5):
                if drop_anomalies:
                    remove_anomalies(pd.to_numeric(p_df[p_df.columns[i]]), drop_level=0.2)
                    p_df[p_df.columns[i]] /= NORM_VALUE

            parameter = p_df.columns[1]

            p_array = p_df[parameter].values
            p_df_list = p_array.tolist()

            p_df_list = np.array(p_df_list)
            p_df_list = p_df_list

            st = 0
            fi = 1500

            train = p_df_list[st:fi]
            try:
                predicted = model.predict(train).tolist()
            except:
                print(train)
                continue

            predicted = [i[0] for i in predicted]
            error = train - predicted

            mae = mean_absolute_error(train, predicted)

            x_ticks = np.arange(st, fi)

            formats = {}
            worksheets = {}
            for key in workbooks:
                worksheets[key] = workbooks[key].add_worksheet(name=skv.replace(' ', '_') + '_' + key)
                formats[key] = workbooks[key].add_format({'num_format': 'dd/mm/yy hh:mm'})

            for key in worksheets:

                values = p_df[parameter].values
                ticks = np.arange(len(values))
                date = p_df[p_df.columns[0]].values
                columns = ['Дата', 'Индекс', 'Значения']

                if key == 'train':
                    values = train
                    ticks = x_ticks
                    date = p_df[p_df.columns[0]].values[st:fi]
                elif key == 'predicted':
                    values = train
                    ticks = x_ticks
                    date = p_df[p_df.columns[0]].values[st:fi]
                    columns.append('Смоделированные значения')
                    columns.append('Ошибка')
                    columns[2] = 'Исходные значения'

                worksheet = worksheets[key]
                worksheet.write_row('A1', columns)
                worksheet.write_column('A2', data=date, cell_format=formats[key])
                worksheet.write_column('B2', data=ticks)
                worksheet.write_column('C2', data=values)

                if key == 'predicted':
                    worksheet.write_column('D2', data=predicted)
                    worksheet.write_column('E2', data=error)

                chart = workbooks[key].add_chart({'type': 'line'})
                chart.set_y_axis({'name': parameter})
                chart.set_x_axis({'name': 'Время', 'num_font': {'rotation': -45}})

                chart.add_series({
                    'categories': [worksheet.name, 1, 0, len(ticks), 0],
                    'values': [worksheet.name, 1, 2, len(ticks), 2],
                    'name': columns[2]
                })

                if key == 'predicted':
                    chart.add_series({
                        'categories': [worksheet.name, 1, 0, len(ticks), 0],
                        'values': [worksheet.name, 1, 3, len(ticks), 3],
                        'name': columns[3]
                    })

                worksheet.insert_chart('G2', chart)

                workbooks[key].close()


if __name__ == '__main__':
    model = load_model(PATH + '/data/RPAS_MLP.h5')
    # model = train_model()
    # model.save(PATH + "/data/RPAS_MLP.h5")
    predict(model, 'mlp', drop_anomalies=True)
