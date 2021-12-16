from glob import glob
import pyodbc
import xlsxwriter
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, max_error as sk_max_error

PATH = 'C:/Users/a.abornev/Google Диск/data/'
DB_PATH = 'C:/Users/a.abornev/Google Диск/data/rpas.accdb'

def load():

  model = load_model(PATH + '/RPAS_MLP.h5')

  p_kust = 202
  p_skv = 2

  p_df = pd.DataFrame(pd.read_excel(os.path.join(PATH, 'Куст {}'.format(p_kust),
                                                 'Скважина {}{}.xlsx'.format(p_kust, p_skv)), engine='openpyxl')).drop(
    index=0).dropna()
  p_df = p_df.drop(p_df[(p_df[p_df.columns[1]] < 1) | (p_df[p_df.columns[2]] < 1) | (p_df[p_df.columns[3]] < 1) | (p_df[p_df.columns[4]] < 1)].index)

  p_array = p_df[p_df.columns[1]].values
  p_df = p_array.tolist()

  p_df = np.array(p_df)
  p_df = p_df / 160

  st = int(len(p_df) * 0.9) + 5
  fi = st + 100

  predicted = p_df

  p = model.predict(predicted).tolist()

  t = []

  for i in p:
    t.append(i[0])

  p = t

  y = predicted[st:fi]
  p = p[st:fi]
  x_ticks = np.arange(st, fi)

  mlp_mae = mean_absolute_error(y, p)
  mlp_me = sk_max_error(y, p)
  print('MLP Avg Loss: ', mlp_mae)
  print('MLP Max error:', mlp_me)

  return model


def predict(model):

  workbook = xlsxwriter.Workbook(PATH + '/file.xlsx')

  for dir in glob(PATH + '/Куст*'):
    for filename in glob(dir + '/Скважина*'):
      if 'предсказание' in filename or '.png' in filename:
        continue
      print(filename)

      p_df = pd.DataFrame(pd.read_excel(filename, engine='openpyxl')).drop(index=0).dropna()
      p_df = p_df.drop(p_df[(p_df[p_df.columns[1]] < 1) | (p_df[p_df.columns[2]] < 1) | (p_df[p_df.columns[3]] < 1) | (p_df[p_df.columns[4]] < 1)].index)

      p_array = p_df[p_df.columns[1]].values
      p_df = p_array.tolist()

      p_df = np.array(p_df)
      p_df = p_df / 160

      st = 000
      fi = 1500

      predicted = p_df
      try:
        p = model.predict(predicted).tolist()
      except:
        print(predicted)
        continue

      t = []

      for i in p:
        t.append(i[0])

      p = t

      x_ticks = np.arange(st,fi)
      y=predicted[st:fi]
      p=p[st:fi]

      sheet_name = filename[filename.find('Скважина'):filename.find('.')]
      worksheet = workbook.add_worksheet(name=sheet_name)

      columns = ['Time', 'Исходные значения', 'MLP значения', 'Отклонения']
      worksheet.write_row('A1', columns)
      worksheet.write_column('A2', data=x_ticks)
      worksheet.write_column('B2', data=y)
      worksheet.write_column('C2', data=p)
      worksheet.write_column('D2', y-p)

      data_start_loc = [0, 1] # xlsxwriter rquires list, no tuple
      data_end_loc = [data_start_loc[0] + len(x_ticks), 2]

      chart = workbook.add_chart({'type': 'line'})
      chart.set_y_axis({'name': 'Давление до УР'})
      chart.set_x_axis({'name': 'Time', 'num_font':  {'rotation': -45}})

      chart.add_series({
          'categories': [worksheet.name, 1, 0, len(x_ticks), 0],
          'values': [worksheet.name, 1, 1, len(x_ticks), 1],
          'name': "Исходные значения"
      })

      chart.add_series({
          'categories': [worksheet.name, 1, 0, len(x_ticks), 0],
          'values': [worksheet.name, 1, 2, len(x_ticks), 2],
          'name': "MLP значения"
      })

      worksheet.insert_chart('G2', chart)

  workbook.close()

def init_db():

  conn = pyodbc.connect(rf'Driver=\{{Microsoft Access Driver (*.mdb, *.accdb)\}};DBQ={DB_PATH};')





if __name__ == '__main__':



  #model = load()
  #predict(model)