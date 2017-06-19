import math
import pandas as pd
import quandl
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


quandl.ApiConfig.api_key = 'DGkvtVPUqy_Xgaa795mB'
data = quandl.get_table('WIKI/PRICES')

data = data[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]
data['HL_PCT'] = (data['adj_high'] - data['adj_low']) / data['adj_low'] * 100
data['PCT_change'] = (data['adj_close'] - data['adj_open']) / data['adj_open'] * 100

data = data[['adj_close', 'HL_PCT', 'PCT_change', 'adj_volume']]

forecast_col = 'adj_close'
data.fillna(0, inplace=True)

forecast_out = int(math.ceil(0.01*len(data)))
data['label'] = data[forecast_col].shift(-forecast_out)
data.dropna(inplace=True)


x = np.array(data.drop(['label'], 1))
x = preprocessing.scale(x)
y = np.array(data['label'])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

classifier = LinearRegression()
classifier.fit(x_train, y_train)
accuracy = classifier.score(x_test, y_test)

print(data.head())

print('accuracy is :', accuracy * 100, '%')

