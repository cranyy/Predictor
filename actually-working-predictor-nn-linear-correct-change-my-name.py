import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import multiprocessing as mp
import platform
import logging
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import ta
from sklearn.linear_model import LinearRegression

# Define the URL for the S&P 500 company list
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def linear_regression_model(stock_data):
    stock_data = stock_data.dropna()
    X = stock_data[['7_day_mean', '30_day_mean', '365_day_mean']]
    y = stock_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    last_30_days = stock_data[['7_day_mean', '30_day_mean', '365_day_mean']].tail(30)
    last_30_days_scaled = scaler.transform(last_30_days)
    future_prices = model.predict(last_30_days_scaled)
    return mse, future_prices

def add_RSI(stock_data, period=14):
    rsi = ta.momentum.RSIIndicator(close=stock_data["Close"], window=period)
    stock_data["RSI"] = rsi.rsi()

def add_MACD(stock_data, short_period=12, long_period=26, signal_period=9):
    macd = ta.trend.MACD(close=stock_data["Close"], window_slow=long_period, window_fast=short_period, window_sign=signal_period)
    stock_data["MACD"] = macd.macd()
    stock_data["MACD_Signal"] = macd.macd_signal()

def add_Bollinger_Bands(stock_data, period=20, std_dev=2):
    bollinger = ta.volatility.BollingerBands(close=stock_data["Close"], window=period, window_dev=std_dev)
    stock_data["Bollinger_High"] = bollinger.bollinger_hband()
    stock_data["Bollinger_Low"] = bollinger.bollinger_lband()

def get_stock_data(ticker, start, end):
    stock_data = yf.download(ticker, start=start, end=end)
    return stock_data

def add_features(stock_data):
    stock_data['7_day_mean'] = stock_data['Close'].rolling(window=7).mean()
    stock_data['30_day_mean'] = stock_data['Close'].rolling(window=30).mean()
    stock_data['365_day_mean'] = stock_data['Close'].rolling(window=365).mean()
    add_RSI(stock_data)
    add_MACD(stock_data)
    add_Bollinger_Bands(stock_data)
    return stock_data

def get_action(current_price, future_price):
    if future_price > current_price * 1.01:
        return 'BUY'
    elif future_price < current_price * 0.99:
        return 'SELL'
    else:
        return 'UNKNOWN'

def neural_network_model(stock_data):
    stock_data = stock_data.dropna()
    X = stock_data[['7_day_mean', '30_day_mean', '365_day_mean', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']]
    y = stock_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(50, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    last_30_days = stock_data[['7_day_mean', '30_day_mean', '365_day_mean', 'RSI', 'MACD', 'MACD_Signal', 'Bollinger_High', 'Bollinger_Low']].tail(30)
    last_30_days_scaled = scaler.transform(last_30_days)
    future_prices = model.predict(last_30_days_scaled)
    return mse, future_prices

start = dt.datetime(2010, 1, 1)
end = dt.datetime.now()

# Retrieve S&P 500 stock tickers from Wikipedia
wiki_table = pd.read_html(url)[0]
sp500_list = wiki_table['Symbol'].tolist()[:10]

mse_df = pd.DataFrame(columns=['Symbol', 'LR_MSE', 'NN_MSE', '1d_Action', '7d_Action', '1month_Action',
                               '7_day_mean', '30_day_mean', '365_day_mean', 'RSI', 'MACD', 'MACD_Signal',
                               'Bollinger_High', 'Bollinger_Low'])

for symbol in tqdm(sp500_list):
    try:
        stock_data = get_stock_data(symbol, start, end)
        stock_data = add_features(stock_data)
        lr_mse, lr_future_prices = linear_regression_model(stock_data)
        nn_mse, nn_future_prices = neural_network_model(stock_data)

        action_1d_nn = get_action(stock_data['Close'].iloc[-1], nn_future_prices[-1])
        action_7d_nn = get_action(stock_data['Close'].iloc[-1], nn_future_prices[-7])
        action_1month_nn = get_action(stock_data['Close'].iloc[-1], nn_future_prices[0])

        lr_future_price_1d = lr_future_prices[-1]
        lr_future_price_7d = lr_future_prices[-7]
        lr_future_price_1month = lr_future_prices[0]

        action_1d_lr = get_action(stock_data['Close'].iloc[-1], lr_future_price_1d)
        action_7d_lr = get_action(stock_data['Close'].iloc[-1], lr_future_price_7d)
        action_1month_lr = get_action(stock_data['Close'].iloc[-1], lr_future_price_1month)

        last_30_days = stock_data.tail(30)
        last_30_days_mean = last_30_days[['7_day_mean', '30_day_mean', '365_day_mean', 'RSI', 'MACD', 'MACD_Signal',
                                           'Bollinger_High', 'Bollinger_Low']].mean()

        new_row = pd.DataFrame({
            'Symbol': [symbol],
            'LR_MSE': [lr_mse],
            'NN_MSE': [nn_mse],
            '1d_Action_NN': [action_1d_nn],
            '7d_Action_NN': [action_7d_nn],
            '1month_Action_NN': [action_1month_nn],
            '1d_Action_LR': [action_1d_lr],
            '7d_Action_LR': [action_7d_lr],
            '1month_Action_LR': [action_1month_lr],
            '7_day_mean': [last_30_days_mean['7_day_mean']],
            '30_day_mean': [last_30_days_mean['30_day_mean']],
            '365_day_mean': [last_30_days_mean['365_day_mean']],
            'RSI': [last_30_days_mean['RSI']],
            'MACD': [last_30_days_mean['MACD']],
            'MACD_Signal': [last_30_days_mean['MACD_Signal']],
            'Bollinger_High': [last_30_days_mean['Bollinger_High']],
            'Bollinger_Low': [last_30_days_mean['Bollinger_Low']]
        })

        mse_df = mse_df.append(new_row, ignore_index=True)
    except Exception as e:
        print(f"Error processing {symbol}: {e}")

mse_df.drop(['1d_Action', '7d_Action', '1month_Action'], axis=1, inplace=True)
mse_df.to_csv('mse_comparison1.csv', index=False)
print("Analysis completed and saved to mse_comparison.csv")

