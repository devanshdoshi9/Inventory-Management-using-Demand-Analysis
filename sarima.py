import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle

# Cleaning Steps
df=pd.read_csv('wheat_demand_haryana.csv')
df.columns=["Month","Sales"]
df.drop(106,axis=0,inplace=True)
df.drop(105,axis=0,inplace=True)
df['Month']=pd.to_datetime(df['Month'])
df.set_index('Month',inplace=True)

# Differencing
df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)
df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)

# SARIMA model
model=sm.tsa.statespace.SARIMAX(df['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()

# Dumping in Pickle file
pickle.dump(results, open('wheat_sarima.pkl', 'wb'))
