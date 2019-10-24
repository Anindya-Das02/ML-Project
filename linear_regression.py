import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
import datetime
'''
linear regression model considering 8000 datasets.
'''
f = pd.read_csv("AirQualityUCI.csv")
rmse_array_train = []
rmse_array = []
r2_array = []
coeff_array = []
intercept_array = []
gas_name = list(f.columns.values)
for i in range(2,15):
    co_conc = f.iloc[:8000,[i]].values
    dates = f.iloc[:8000,[0]].values 
    times = f.iloc[:8000,[1]].values

    tick_dates = []
    for j in range(len(dates)): 
        dts = dates[j][0]
        tms = times[j][0]
        r = dts.split('-')
        t = tms.split(':')
        dtr = datetime.datetime(int(r[2]),int(r[1]),int(r[0]),int(t[0]))
        tick_dates.append([dtr.timestamp()])

    x_train,x_test,y_train,y_test = train_test_split(tick_dates,co_conc,test_size=0.2,random_state=0)
    reg = LinearRegression()
    reg.fit(x_train,y_train)
    y_predictions = reg.predict(x_test)

    print("gas/air pollutant: {}".format(gas_name[i]))
    print("size x_train:",len(x_train))
    print("size y_train:",len(y_train))
    print("size y_test:",len(y_test))
    print("size x_test:",len(x_test))
    print("size y_predict:",len(y_predictions))
    print("coeff:",reg.coef_[0][0])
    print("intercept",reg.intercept_[0])
    print("R^2 on train set: {}".format(reg.score(x_train,y_train)))
    print("R^2 on test set: {}".format(reg.score(x_test,y_test)))
    print("RMSE:{}".format(sqrt(mean_squared_error(y_test,y_predictions))))
    #rmse_array_train.append(sqrt(mean_squared_error(y_train, y_predictions)))
    rmse_array.append(sqrt(mean_squared_error(y_test, y_predictions)))
    r2_array.append(reg.score(x_test,y_test))
    coeff_array.append(reg.coef_[0][0])
    intercept_array.append(reg.intercept_[0])

    plt.scatter(x_train,y_train,color="blue",label="train")
    plt.scatter(x_test,y_test,color="green",label="test")
    plt.plot(x_train,reg.predict(x_train),color="red",label="predict")
    plt.xlabel("timestamp")
    plt.ylabel("conc")
    plt.title("Gas {}".format(gas_name[i]))
    plt.legend()
    plt.show()

    print("-----")
print("avg RMSE : {}".format(np.mean(rmse_array)))
print("avg R^2 : {}".format(np.mean(r2_array)))
print("avg Coeff value : {}".format(np.mean(coeff_array)))
print("avg intercept value : {}".format(np.mean(intercept_array)))
print("--done--")


