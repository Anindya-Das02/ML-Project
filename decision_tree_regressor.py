import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import datetime

f = pd.read_csv("AirQualityUCI.csv")

dates = f.iloc[:300,[0]].values 
times = f.iloc[:300,[1]].values
co_con = f.iloc[:300,[3]].values 
tick_dates = []
for i in range(len(dates)): 
    dts = dates[i][0]
    tms = times[i][0]
    r = dts.split('-')
    t = tms.split(':')
    dtr = datetime.datetime(int(r[2]),int(r[1]),int(r[0]),int(t[0]))
    tick_dates.append([dtr.timestamp()])

x_train,x_test,y_train,y_test = train_test_split(tick_dates,co_con,test_size=0.2,random_state=0)

# print(x_train)
# print(y_train)



reg = DecisionTreeRegressor()
reg.fit(x_train,y_train)
print("Coefficient of determination R^2 <-- on train set: {}".format(reg.score(x_train, y_train)))
print("Coefficient of determination R^2 <-- on test set: {}".format(reg.score(x_test, y_test)))
tr = []
for i in range(len(x_train)):
    tr.append(x_train[i][0])

x_train  = tr
x_grid = np.arange(np.min(x_train),np.max(x_train),1)
x_grid = x_grid.reshape((len(x_grid),1))





plt.scatter(x_train,y_train,color="blue",label="train")
plt.scatter(x_test,y_test,color="green",label="test")
plt.plot(x_grid,reg.predict(x_grid),color="red",label="predict")
plt.legend()
plt.show()






