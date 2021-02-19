# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 13:16:56 2020

@author: Vassili
"""

#Import the necessary libraries
from alpha_vantage.timeseries import TimeSeries #Historical stock data
from alpha_vantage.techindicators import TechIndicators #Historical stock tech indicator data
from keras.layers import Input, Dense, Dropout, LSTM, Activation, concatenate
from keras.models import Model
from keras import optimizers
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from matplotlib import pyplot as plt
import os.path
import time
#Set a seed to achieve a replicability of the results
import numpy as np
np.random.seed(1508)
import tensorflow as tf
tf.compat.v1.set_random_seed(1508)

#########################
#########Dataset#########
#########################

#Enter your own AlphaVantage Key
#Get one for free under https://www.alphavantage.co/support/#api-key
ts = TimeSeries(key='ENTER YOUR OWN KEY',output_format='pandas')
ti = TechIndicators(key='ENTER YOUR OWN KEY', output_format='pandas')

#Choose the stock you want to build the machine learning model on
stock='MSFT'

#The path where you write and read the historical and tech indicator data of your chosen stock
path_hist='C:/Users/Vassili/Desktop/ML Unibo/'+'hist_'+ stock +'.csv'
path_ti='C:/Users/Vassili/Desktop/ML Unibo/'+'ti_'+ stock +'.csv'


#In case there exists already the historical stock data, it will read the .csv file
if os.path.exists(path_hist):
    hist_old=pd.read_csv(path_hist,index_col='date')
#If not, it will get the data from Alpha Vantage
else:
    #Get historical data from your chosen stock
    hist_old, meta_data = ts.get_daily_adjusted(symbol=stock,outputsize='full')
    #Save the data as a .csv file
    hist_old.to_csv(path_hist)    

#The historical data consists of daily observatios for each trading day of
#'date','1. open','2. high','3. low','4. close','5. adjusted close',
#'6. volume', '7. dividend amount'  and '8. split coefficient'.    
                    
#Since the stock market has changed a lot over the years, use only the last 2000 days    
hist_old = hist_old[0:2000]
#Reset the index, only the order is important, not the dates
hist_old = hist_old.reset_index()

#Stock splits have a huge impact on stock price charts.
#Alpha Vantage only provides adjusted closing prices,
#so in order to also use properly the open, high and low prices
#you have to calculate the adjusted version of them with the help
#of the splitting coefficients.
#Note: The dividends will be ignored.
multiplicator=1
for i in range(len(hist_old)-1):
    multiplicator=multiplicator*hist_old.loc[i,'8. split coefficient']
    hist_old.loc[i,'adj_mult']=multiplicator
    hist_old.iloc[i+1,1:5]=hist_old.iloc[i+1,1:5]/multiplicator

#Change the order of the data     
hist = hist_old.sort_index(axis=0 ,ascending=False)
#Drop the unnecessary columns
hist = hist.drop(['date','5. adjusted close', '7. dividend amount', '8. split coefficient','adj_mult'],axis=1)
#Create an array
hist = hist.values

#Number of past days closing prices, that are used to make a prediction of tomorrow's close price
days_used=50

#The percentage of data that will be used for training
train_test_split = 0.9 
k = int(hist.shape[0] * train_test_split)

#Scale the data between 0 and 1, to improve the convergence
sc = MinMaxScaler(feature_range=(0,1))
sc.fit_transform(hist[:k,]) #fitting only on the training data
hist_scaled = sc.transform(hist) #then use it and transform the whole data set

#The ohlcv_history_scaled will be 1 of our 2 x input parameters for the machine learning model. 
#Each value in it is a numpy array containing 50 open, high, low, close, volume values, going from old to new. 
#So for each x value we are getting the [i : i + days_used] stock prices
#and for the y value the [i + days_used] stock price.
ohlcv_history_scaled=[]
close_next_day_scaled=[]
for i in range(len(hist_scaled) - days_used):
    ohlcv_history_scaled.append(hist_scaled[i:i + days_used])
    close_next_day_scaled.append(hist_scaled[:, 3][i + days_used])
ohlcv_history_scaled=np.array(ohlcv_history_scaled)
close_next_day_scaled=np.array(close_next_day_scaled)
close_next_day_scaled = np.expand_dims(close_next_day_scaled, -1)

close_next_day=[] 
for i in range(len(hist) - days_used):
    close_next_day.append(hist[:, 3][i + days_used])
close_next_day=np.array(close_next_day)    
close_next_day = np.expand_dims(close_next_day, -1)
#Note: We want to predict the next days closing price,
#that's why we use the index 3.

#In case there exists already the historical stock tech indicator data, it will read the .csv file
if os.path.exists(path_ti):
    technical_indicators=pd.read_csv(path_ti,index_col='date')
#If not, it will get the data from Alpha Vantage
else:
    technical_indicators = []
    #Get the first part of the historical stock tech indicator data
    sma_1, meta_data = ti.get_sma(symbol=stock,interval='daily',time_period=5)
    sma_2, meta_data = ti.get_sma(symbol=stock,interval='daily',time_period=20)
    sma_1=sma_1[-(2000-days_used+1):][:2000-days_used]
    sma_2=sma_2[-(2000-days_used+1):][:2000-days_used]
    macd, meta_data = ti.get_macd(symbol=stock,interval='daily',series_type='close')
    macd=macd[1:(2000-days_used+1)]
    stoch, meta_data = ti.get_stoch(symbol=stock,interval='daily')
    stoch=stoch[1:(2000-days_used+1)]
    
    #Attention: Since Alpha Vantage API call frequency is only 5 calls per minute,
    #you have to wait 1 minute before the code continues
    t_end = time.time() + 70
    while time.time() < t_end:
        print (time.time)
    
    #Get the second part of the historical stock tech indicator data    
    rsi_1, meta_data = ti.get_rsi(symbol=stock,interval='daily',time_period=5)
    rsi_2, meta_data = ti.get_rsi(symbol=stock,interval='daily',time_period=20)
    rsi_1=rsi_1[-(2000-days_used+1):][:2000-days_used]
    rsi_2=rsi_2[-(2000-days_used+1):][:2000-days_used]
    adx_1, meta_data = ti.get_adx(symbol=stock,interval='daily',time_period=5)
    adx_2, meta_data = ti.get_adx(symbol=stock,interval='daily',time_period=20)
    adx_1=adx_1[-(2000-days_used+1):][:2000-days_used]
    adx_2=adx_2[-(2000-days_used+1):][:2000-days_used]
    
    #Add all the tech indicators together and save them as a csv. file
    technical_indicators=pd.concat([sma_1,sma_2,macd,stoch,rsi_1,rsi_2,adx_1,adx_2], axis=1, join='inner')
    technical_indicators.to_csv(path_ti)

#Create an array    
technical_indicators = np.array(technical_indicators)

#Scale the data between 0 and 1, to improve the convergence
ti_sc = MinMaxScaler(feature_range=(0,1))
m = int(technical_indicators.shape[0] * train_test_split)
ti_sc.fit_transform(technical_indicators[:m,]) #fitting only on the training data
technical_indicators_scaled = ti_sc.transform(technical_indicators) #then use it and transform the whole data set
    

#Split the dataset up into train and test sets
n = int(ohlcv_history_scaled.shape[0] * train_test_split)

ohlcv_train = ohlcv_history_scaled[:n]
ti_train = technical_indicators_scaled[:n]
y_train = close_next_day_scaled[:n]

ohlcv_test = ohlcv_history_scaled[n:]
ti_test = technical_indicators_scaled[n:]
y_test = close_next_day_scaled[n:]

unscaled_y_test = close_next_day[n:]

#Used later to scale the outpout back into normal values
y_sc = MinMaxScaler(feature_range=(0,1))
y_sc.fit(close_next_day[:n])

#########################
##########Model##########
#########################

#The LSTM branch is used for the historical stock timeseries data
lstm_input = Input(shape=(days_used, 5), name='lstm_input')
#The tech indicator branch is used for the tech indicators
dense_input = Input(shape=(technical_indicators.shape[1],), name='tech_input')

x_1 = LSTM(50, name='lstm')(lstm_input) #50 LSTM Cells
x_1 = Dropout(0.2, name='lstm_dropout')(x_1) #prevent overfitting
lstm_branch = Model(inputs=lstm_input, outputs=x_1)

x_2 = Dense(20, name='tech_dense')(dense_input)
x_2 = Activation("relu", name='tech_relu')(x_2)
x_2 = Dropout(0.2, name='tech_dropout')(x_2)
technical_indicators_branch = Model(inputs=dense_input, outputs=x_2)

#Add the 2 branches together
combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

y = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
y = Dense(1, activation="linear", name='dense_out')(y)
      
model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=y)
adam = optimizers.Adam(lr=0.001) #achieved good results with this learning rate
model.compile(optimizer=adam, loss='mse')
model.summary()
plot_model(model, to_file='machine_learning_model.png') #plots the model visually

#########################
#########Training########
#########################

#Train the model with the training data and evaluate it on the test set
model.fit(x=[ohlcv_train, ti_train], y=y_train, batch_size=32, epochs=50, shuffle=False)
evaluation = model.evaluate([ohlcv_test, ti_test], y_test)
print("MSE (normalised data):",evaluation)

#########################
########Evaluation#######
#########################

#Predict the next days closing stock price with the test data
y_test_predicted = model.predict([ohlcv_test, ti_test])
y_test_predicted = y_sc.inverse_transform(y_test_predicted)

#Predict the next days closing stock price with all the data
y_predicted = model.predict([ohlcv_history_scaled,technical_indicators_scaled])
y_predicted = y_sc.inverse_transform(y_predicted)

#Calculate the mean squared error and print the root of it
mse = np.mean(np.square(unscaled_y_test - y_test_predicted))
print('Root MSE:',np.sqrt(mse))

#In order to get a better quality on the plots when zooming 
%config InlineBackend.figure_format = 'retina'

#Plot the real and the predicted stock price time series on the whole dataset
real = plt.plot(close_next_day, label='real')
pred = plt.plot(y_predicted, label='predicted')

plt.legend(['Real', 'Predicted'])
plt.title(stock)
plt.xlabel("Trading Days")
plt.ylabel("Stock Price")
plt.show()

#Plot the real and the predicted stock price time series on the testset
real = plt.plot(unscaled_y_test, label='real')
pred = plt.plot(y_test_predicted, label='predicted')

#Trading Algorithm 1:
#Lists that save the algorithm steps     
buy_or_sell=[]
win_or_loss=[]
buys=[]
sells=[]
#The predicted closing prices
predicted_close=model.predict([ohlcv_test, ti_test])
for i in range(len(ohlcv_test)-1):
    i=i+1
    #If the predicted closing price (of the next day) is higher than todays closing price, buy the stock
    if predicted_close[i][0]>ohlcv_test[i,-1,3]:
        buy_or_sell.append(1)
        buys.append((i-1, y_sc.inverse_transform(predicted_close[i-1][0].reshape(1, -1))))
        #If the next days closing price is higher than todays closing price, it was a good decision, it counts as win
        if ohlcv_test[i,-1,3]<y_test[i,0]:
            win_or_loss.append(1)
        #If the next days closing price is the same as todays closing price, the decision didn't change smth., it wasn't a win neither a loss
        elif ohlcv_test[i,-1,3]==y_test[i,0]:
            win_or_loss.append(0)
        #If the next days closing price is lower than todays closing price, it was a bad decision, it counts as loss
        elif ohlcv_test[i,-1,3]>y_test[i,0]:
            win_or_loss.append(-1)     
    #If the predicted closing price (of the next day) is lower than todays closing price, sell the stock
    elif predicted_close[i][0]<ohlcv_test[i,-1,3]:
        buy_or_sell.append(-1)
        sells.append((i-1, y_sc.inverse_transform(predicted_close[i-1][0].reshape(1, -1))))
        #If the next days closing price is higher than todays closing price, it was a bad decision, it counts as loss
        if ohlcv_test[i,-1,3]<y_test[i,0]:
            win_or_loss.append(-1)
        #If the next days closing price is the same as todays closing price, the decision didn't change smth., it wasn't a win neither a loss
        elif ohlcv_test[i,-1,3]==y_test[i,0]:
            win_or_loss.append(0)
        #If the next days closing price is lower than todays closing price, it was a good decision, it counts as win
        elif ohlcv_test[i,-1,3]>y_test[i,0]:
            win_or_loss.append(1)
    #If the predicted closing price (of the next day) is the same as todays closing price, do nothing
    elif predicted_close[i][0]==ohlcv_test[i,-1,3]:
        buy_or_sell.append(0)
        win_or_loss.append(0)

#Print the performance of the first trading algorithm
#Note: The performance can go from -1 to 1, so 50% wins and 50% losses account to a performance of 0        
print('Performance Trading Algorithm 1:',sum(win_or_loss)/len(win_or_loss))    
#Plot the buying and selling points and prices, green=buy, red=sell
plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00',s=10)
plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000',s=10)
plt.legend(['Real', 'Predicted','Buy','Sell'])
plt.title(stock+" with Trading Algorithm 1 ")
plt.xlabel("Trading Days")
plt.ylabel("Stock Price")
plt.show()

#Plot the real and the predicted stock price time series on the testset
real = plt.plot(unscaled_y_test, label='real')
pred = plt.plot(y_test_predicted, label='predicted')

#I noticed that, especially for stocks that had very big gains on the last 200 trading days (Tesla, Apple, ...)
#the algorithm would suggest for the vast majority of the time to sell the stock, since it compares the predicted stock prices with real data.
#The machine learning model can't keep up with a big uprise of a stock, if this uprise happens on the test set.
#That is why we need a second trading algorithm, with having these events in mind.

#Trading Algorithm 2:
#Lists that save the algorithm steps
buy_or_sell=[]
win_or_loss=[]
buys=[]
sells=[]
#The predicted closing prices
predicted_close=model.predict([ohlcv_test, ti_test])
for i in range(len(ohlcv_test)-1):
    i=i+1
    #If the predicted closing price (of the next day) is higher than todays predicted closing price, buy the stock
    if predicted_close[i][0]>predicted_close[i-1][0]:
        buy_or_sell.append(1)
        buys.append((i-1, y_sc.inverse_transform(predicted_close[i-1][0].reshape(1, -1))))
        #If the next days closing price is higher than todays closing price, it was a good decision, it counts as win
        if ohlcv_test[i,-1,3]<y_test[i,0]:
            win_or_loss.append(1)
        #If the next days closing price is the same as todays closing price, the decision didn't change smth., it wasn't a win neither a loss
        elif ohlcv_test[i,-1,3]==y_test[i,0]:
            win_or_loss.append(0)
        #If the next days closing price is lower than todays closing price, it was a bad decision, it counts as loss
        elif ohlcv_test[i,-1,3]>y_test[i,0]:
            win_or_loss.append(-1)     
    #If the predicted closing price (of the next day) is lower than todays predicted closing price, sell the stock
    elif predicted_close[i][0]<predicted_close[i-1][0]:
        buy_or_sell.append(-1)
        sells.append((i-1, y_sc.inverse_transform(predicted_close[i-1][0].reshape(1, -1))))
        #If the next days closing price is higher than todays closing price, it was a bad decision, it counts as loss
        if ohlcv_test[i,-1,3]<y_test[i,0]:
            win_or_loss.append(-1)
        #If the next days closing price is the same as todays closing price, the decision didn't change smth., it wasn't a win neither a loss
        elif ohlcv_test[i,-1,3]==y_test[i,0]:
            win_or_loss.append(0)
        #If the next days closing price is lower than todays closing price, it was a good decision, it counts as win
        elif ohlcv_test[i,-1,3]>y_test[i,0]:
            win_or_loss.append(1)
    #If the predicted closing price (of the next day) is the same as todays closing price, do nothing
    elif predicted_close[i][0]==predicted_close[i-1][0]:
        buy_or_sell.append(0)
        win_or_loss.append(0)

#Print the performance of the second trading algorithm
#Note: The performance can go from -1 to 1, so 50% wins and 50% losses account to a performance of 0        
print('Performance Trading Algorithm 2:',sum(win_or_loss)/len(win_or_loss))    
#Plot the buying and selling points and prices, green=buy, red=sell        
plt.scatter(list(list(zip(*buys))[0]), list(list(zip(*buys))[1]), c='#00ff00',s=10)
plt.scatter(list(list(zip(*sells))[0]), list(list(zip(*sells))[1]), c='#ff0000',s=10)
plt.legend(['Real', 'Predicted','Buy','Sell'])
plt.title(stock+" with Trading Algorithm 2 ")
plt.xlabel("Trading Days")
plt.ylabel("Stock Price")
plt.show()
