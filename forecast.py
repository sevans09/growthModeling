#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt


SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


df = pd.read_csv('/Users/SookHee/Desktop/monthly_logs.csv')

# logs per month
df.plot(color = "black")
plt.title('Monthly Logs (Bot Less than 80% Confident)', size = 15)
plt.xlabel('Months from January 1 2018')
plt.grid('on')

df = pd.read_csv('/Users/SookHee/Desktop/triage_monthly.csv')

m = Prophet(yearly_seasonality=True)
m.fit(df, algorithm='Newton')

#future
future = m.make_future_dataframe(periods=365)
future.tail()

#forecast
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

#plot trend
fig1 = m.plot(forecast, color = "black")
fig2 = m.plot_components(forecast, color = "black")
