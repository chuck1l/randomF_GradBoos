import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import warnings
warnings.filterwarnings("ignore")

class stock_clean(object):
    def __init__(self, name, data):
        self.name = name
        self.df = data
    # Initial look at all data
    def view_data(self):
        self.info = self.df.info()
        self.describe = self.df.describe()
        self.shape = self.df.shape
        self.head = self.df.head()
        print(
            'Data Info:', self.info,
            'Data Describe:', self.describe,
            'Data Shape:', self.shape,
            'Data Head:', self.head
        )
        return None
    # Creating a date column that is data only (no time), and 
    # a day of the week column. Change date to datetime format
    def date_col(self):
        self.df['datetime'] = pd.to_datetime(self.df.time, unit='s')
        self.df.drop('time', axis=1, inplace=True)
        self.df['date'] = self.df.datetime.dt.date
        self.df['year'] = self.df.datetime.dt.year
        self.df['month'] = self.df.datetime.dt.month
        self.df['dow'] = self.df.datetime.dt.dayofweek
        self.df['day'] = self.df.datetime.dt.day
        self.df['hour'] = self.df.datetime.dt.hour
        self.df.index = self.df['datetime']
        return None
    # Drop columns if they're all NaN
    def drop_cols(self, columns):
        self.df.drop(columns, axis=1, inplace=True)
        return None
    def drop_na_col(self):
        self.df = self.df.dropna(axis = 1, how ='all')
        return None
    # Create dummy variabels for the day of the week column
    def day_of_week_dummify(self):
        self.df = pd.get_dummies(self.df, prefix=['time', 'dow'])
        return None
    # Creating columns for tomorrow's high and low (future predictions)
    def new_cols(self):
        self.df['prev_close'] = self.df['close'].shift(periods=1)
        self.df['gap%'] = (self.df['open'] - self.df['prev_close']) / self.df['prev_close'] * 100
        self.df['trw_high'] = self.df['high'].shift(periods=-1)
        self.df['trw_high'].fillna(self.df['open'], inplace=True)
        self.df['trw_low'] = self.df['low'].shift(periods=-1)
        self.df['trw_low'].fillna(self.df['open'], inplace=True)

    # Format column names pythonically 
    def format_cols(self):
        cols = {'MA': 'ma_20_1d', 'MA.1': 'ma_50_1d', 'MA.2': 'ma_100_1d', 
                'MA.3': 'ma_200_1d', 'EMA': 'ema_20_1d', 'EMA.1': 'ema_50_1d', 
                'Basis': 'bb_basis', 'Upper': 'bb_upper', 'Lower': 'bb_lower',
                'MA.4': 'ma_20_scale', 'MA.5': 'ma_50_scale', 'MA.6': 'ma_100_scale', 
                'MA.7': 'ma_200_scale'}
        self.df.rename(columns=cols, inplace=True)
        self.df.columns = self.df.columns.str.replace(' ', '_')
        self.df.columns = self.df.columns.str.lower()
        self.df.drop(['min', 'max', 'histogram', 'signal'], axis=1, inplace=True)
        return 
    
    # Remove rows with NaN from history
    def drop_na_rows(self):
        self.df = self.df.dropna(axis=0, how='any')
        return None
    # Create a 5 day high and low, attempt to predict in the future
    def five_day_mean(self):
        temp_lst_h = [np.nan] * 4
        temp_lst_l = [np.nan] * 4
        for i in range(5, self.df.shape[0]+1):
            temp_lst_h.append(self.df['high'].iloc[i-5:i].mean())
            temp_lst_l.append(self.df['low'].iloc[i-5:i].mean())
        temp_df1 = pd.DataFrame(temp_lst_h, columns=['prev_5day_high_mean'])
        temp_df1.index = self.df['datetime']
        temp_df2 = pd.DataFrame(temp_lst_l, columns=['prev_5day_low_mean'])
        temp_df2.index = self.df['datetime']
        self.df = pd.concat([self.df, temp_df1, temp_df2], axis=1)
        return None
    def filter_gappers(self, threshold):
        mask = self.df['gap%'] >= threshold 
        self.df = self.df[mask]
        return None

if __name__ == '__main__':
    pass