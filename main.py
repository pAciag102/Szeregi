import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
import statsmodels.api as sm
from pandas.plotting._matplotlib import lag_plot
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acovf,acf,pacf,pacf_yw,pacf_ols
import warnings
warnings.filterwarnings("ignore")

import tkinter as tk

class mclass:

    def __init__(self, window):
        self.box = Entry(window)
        self.plotButton = Button(window, text="func_1", command=self.plot)
        self.plotButton2 = Button(window, text="func_2", command=self.plot2)
        self.plotButton3= Button(window, text="func_3", command=self.plot3)
        self.plotButton4 = Button(window, text="func_4", command=self.plot4)
        self.plotButton5 = Button(window, text="func_5", command=self.plot5)
        
        self.box.pack()
        self.plotButton.pack()
        self.plotButton2.pack()
        self.plotButton3.pack()
        self.plotButton4.pack()
        self.plotButton5.pack()

    def plot(self):
        df = pd.read_csv('data/airline_passengers.csv', index_col='Month', parse_dates=True)
        df.index.freq = 'MS'
        print(df.head())
        train_data = df.iloc[:109]
        test_data = df.iloc[108:]
        fitted_model = ExponentialSmoothing(train_data['Thousands of Passengers'],
                                            trend='mul',
                                            seasonal='mul',
                                            seasonal_periods=12).fit()
        test_predictions = fitted_model.forecast(36)
        print(test_predictions)
        plt.interactive(False)
        train_data['Thousands of Passengers'].plot(legend=True, label='Train')
        test_data['Thousands of Passengers'].plot(legend=True, label='Test')
        test_predictions.plot(legend=True, label='Prediction', xlim=['1958-01-01', '1961-01-01'])
        plt.show()
        print(test_data.describe())
        print(mean_absolute_error(test_data, test_predictions))
        print(np.sqrt(mean_squared_error(test_data, test_predictions)))

    def plot2(self):
        df2 = pd.read_csv('data/samples.csv', index_col=0, parse_dates=True)
        plt.interactive(False)
        df2['a'].plot()
        plt.show()

    def plot3(self):
        df = pd.read_csv('data/macrodata.csv', index_col=0, parse_dates=True)
        df.head()
        ax = df['realgdp'].plot()
        ax.autoscale(axis='x',tight=True)
        ax.set(ylabel='REAL GDP');
        # Tuple unpacking
        gdp_cycle, gdp_trend = hpfilter(df['realgdp'], lamb=1600)
        df['trend'] = gdp_trend
        df[['trend', 'realgdp']].plot().autoscale(axis='x', tight=True);
        df[['trend', 'realgdp']]['2000-03-31':].plot(figsize=(12, 8)).autoscale(axis='x', tight=True);
        plt.interactive(False)
        plt.show()

    def plot4(self):
        airline = pd.read_csv('data/airline_passengers.csv', index_col='Month', parse_dates=True)
        airline.dropna(inplace=True)
        result = seasonal_decompose(airline['Thousands of Passengers'],
                                    model='multiplicative')  # model='mul' also works
        result.plot();
        airline['6-month-SMA'] = airline['Thousands of Passengers'].rolling(window=6).mean()
        airline['12-month-SMA'] = airline['Thousands of Passengers'].rolling(window=12).mean()
        airline['EWMA12'] = airline['Thousands of Passengers'].ewm(span=12, adjust=False).mean()
        airline[['Thousands of Passengers', 'EWMA12']].plot();
        airline[['Thousands of Passengers', 'EWMA12', '12-month-SMA']].plot(figsize=(12, 8)).autoscale(axis='x',
                                                                                                       tight=True);
        plt.interactive(False)
        plt.show()

    def plot5(self):
        df1 = pd.read_csv('data/airline_passengers.csv', index_col='Month', parse_dates=True)
        df1.index.freq = 'MS'
        df2 = pd.read_csv('data/DailyTotalFemaleBirths.csv', index_col='Date', parse_dates=True)
        df2.index.freq = 'D'
        df = pd.DataFrame({'a': [13, 5, 11, 12, 9]})
        arr = acovf(df['a'])
        arr2 = acovf(df['a'], unbiased=True)
        arr3 = acf(df['a'])
        arr4 = pacf_yw(df['a'], nlags=4, method='mle')
        lag_plot(df1['Thousands of Passengers']);
        lag_plot(df2['Births']);
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        from statsmodels.tsa.statespace.tools import diff
        title = 'Autocorrelation: Daily Female Births'
        lags = 40
        plot_acf(df2, title=title, lags=lags);
        title = 'Autocorrelation: Airline Passengers'
        lags = 40
        plot_acf(df1, title=title, lags=lags);
        plt.interactive(False)
        plt.show()
        



window = Tk()
# TODO Zmeniń xD
window.title('Karniak')
window.geometry('600x800')
start = mclass(window)
window.mainloop()
