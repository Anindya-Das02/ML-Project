import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import datetime
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


f = pd.read_csv("AirQualityUCI.csv")
dates = f.iloc[:300,0:1].values
times = f.iloc[:300,1:2].values
co_con = f.iloc[:300,3:4].values
tick_dates = []
for i in range(len(dates)): 
    dts = dates[i][0]
    tms = times[i][0]
    r = dts.split('-')
    t = tms.split(':')
    dtr = datetime.datetime(int(r[2]),int(r[1]),int(r[0]),int(t[0]))
    tick_dates.append([dtr.timestamp()])

x_train,x_test,y_train,y_test = train_test_split(tick_dates,co_con,test_size=0.3,random_state=0)

reg = RandomForestRegressor(n_estimators=100)
reg.fit(x_train,y_train)


print("Coefficient of determination R^2 <-- on train set: {}".format(reg.score(x_train, y_train)))
print("Coefficient of determination R^2 <-- on test set: {}".format(reg.score(x_test, y_test)))
print("RMSE:{}".format(np.sqrt(mean_squared_error(y_test,reg.predict(y_test)))))
tr = []
for i in range(len(x_train)):
    tr.append(x_train[i][0])

x_train  = tr
x_grid = np.arange(min(x_train),max(x_train),0.1)
x_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x_train,y_train,color="blue",label="train")
plt.scatter(x_test,y_test,color="green",label="test")
plt.plot(x_grid,reg.predict(x_grid))
plt.legend()
plt.show()



