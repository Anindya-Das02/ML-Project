import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
import datetime
from math import sqrt


'''
Decision tree regression model for 300 rows
note: 1000 rows of data is causing memory error
because of less processing power
'''

f = pd.read_csv("AirQualityUCI.csv")

rmse_array_train = []
rmse_array = []
r2_array = []
gas_name = list(f.columns.values)
for i in range(2,15):
    conc = f.iloc[:300,[i]].values
    dates = f.iloc[:300,0:1].values
    times = f.iloc[:300,1:2].values
    timestamp_array = []
    for j in range(len(dates)):
        dts = dates[j][0]
        tms = times[j][0]
        r = dts.split('-')
        t = tms.split(':')
        dtr = datetime.datetime(int(r[2]),int(r[1]),int(r[0]),int(t[0]))
        timestamp_array.append([dtr.timestamp()])

    x_train,x_test,y_train,y_test = train_test_split(timestamp_array,conc,test_size=0.2,random_state=0)
    reg = DecisionTreeRegressor()
    reg.fit(x_train,y_train)

    y_predictions = reg.predict(y_test)

    print("----")
    print("gas name: {}".format(gas_name[i]))
    print("Coefficient of determination R^2 <-- on train set: {}".format(reg.score(x_train, y_train))) 
    print("Coefficient of determination R^2 <-- on test set: {}".format(reg.score(x_test, y_test)))
    print("RMSE:{}".format(sqrt(mean_squared_error(y_test,y_predictions))))
    rmse_array.append(sqrt(mean_squared_error(y_test,y_predictions)))
    r2_array.append(reg.score(x_test,y_test))

    x_grid = np.arange(np.min(x_train),np.max(x_train),1)
    x_grid = x_grid.reshape((len(x_grid),1))

    plt.scatter(x_train,y_train,color="blue",label="train")
    plt.scatter(x_test,y_test,color="green",label="test")
    plt.xlabel("timestamp")
    plt.ylabel("AQI")
    plt.title(gas_name[i])
    plt.plot(x_grid,reg.predict(x_grid),color="red",label="predict")
    plt.legend()
    plt.show()

print("*** RESULTS ****")
print("avg RMSE : {}".format(np.mean(rmse_array)))
print("avg R^2 : {}".format(np.mean(r2_array)))
print("== END ==")