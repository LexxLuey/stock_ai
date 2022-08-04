import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from nsepy import get_history
import tensorflow as tf
from django.conf import settings
from tensorflow import keras
import socket
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error,accuracy_score
import os
import urllib.request
import json
import datetime as dt


def predict(name):
    if "_" in name:
        name = name.replace("_", " ")

    print("Request Recieved for {}".format(name))
    """
    txt = "NIFTY_50"

    """
    today = dt.date.today()
    try:
        # data = get_history(symbol= name, start=date(today.year - 1,today.month, today.day ), end= today)
        api_key = "VYKNL7H7WAZG4XAW"

        # American Airlines stock market prices
        ticker = name

        # JSON file with all the stock market data for AAL from the last 20 years
        url_string = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_key}"

        # Save data to this file
        # if "." in name:
        #     name = name.replace(".", "_")
        file_to_save = f'stock_market_data-{ticker}.csv'

        # If you haven't already saved data,
        # Go ahead and grab the data from the url
        # And store date, low, high, volume, close, open values to a Pandas DataFrame
        if not os.path.exists(file_to_save):
            with urllib.request.urlopen(url_string) as url:
                data = json.loads(url.read().decode())
                # extract stock market data
                data = data['Time Series (Daily)']
                df = pd.DataFrame(
                    columns=['Date', 'Low', 'High', 'Close', 'Open'])
                for k, v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    data_row = [date.date(), float(v['3. low']), float(v['2. high']),
                                float(v['4. close']), float(v['1. open'])]
                    df.loc[-1, :] = data_row
                    df.index = df.index + 1
            
            print('Data saved to : %s' % file_to_save)
            df.to_csv(file_to_save)
            df = pd.read_csv(file_to_save, usecols = ['Close'], low_memory = True)

        # If the data is already there, just load it from the CSV
        else:
            print('File already exists. Loading data from CSV')
            df = pd.read_csv(file_to_save, usecols = ['Close'], low_memory = True)

    except socket.gaierror:
        return "Invalid Stock Ticker or Connection Error"
    #data[['Close']].plot()

    data = df.filter(['Open','Close','High','Low'])

    data = df.filter(['Close'])

    data = pd.DataFrame(data = df)

    print(f" data = {data} ")

    print(data.isna().any())

    pivot = int(len(data) * 0.8)

    training_data = data.iloc[:pivot]
    testing_data = data.iloc[pivot:]

    print("testing data",testing_data)

    past_60 = training_data.tail(60)
    test_data = past_60.append(testing_data)

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_training_data = scaler.fit_transform(training_data.values.reshape(-1,1))
    scaled_testing_data = scaler.transform(test_data.values.reshape(-1,1))
    scaled_final_data = scaler.transform(testing_data)

    data = scaler.transform(data)

    x_test, y_test = [], []

    #print("scaled shape",scaled_testing_data.shape[0])

    for i in range(60,scaled_testing_data.shape[0]):
        x_test.append(scaled_testing_data[i - 60:i])
        y_test.append(scaled_testing_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)

    y_test = y_test.reshape(-1,1)

    print("y_test value")

    print(x_test.shape, y_test.shape)

    model = keras.models.load_model('stock_prediction_model')


    predictions = model.predict(x_test)
    print(f"predictions = {predictions} ")
    # p = scaler.inverse_transform(predictions)
    # print(f" popopo = {p} ")

    n_data = [data[len(data)+1 - 61:len(data+1),0]]

    n_data = np.array(n_data)

    print(n_data.shape)

    n_data = np.reshape(n_data,(n_data.shape[0], n_data.shape[1], 1))

    predi = model.predict(n_data)
    print(f" predi = {predi} ")

    predi_result = scaler.inverse_transform(predi)

    #result = predi_result.astype(np.int)
    print(predi_result)
    return predi_result[0][0]



