import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime

f = pd.read_csv("AirQualityUCI.csv")
for i in range(2,15):
    co_conc = f.iloc[:8000,[i]].values
    dates = f.iloc[:8000,[0]].values 
    times = f.iloc[:8000,[1]].values

    tick_dates = []
    for i in range(len(dates)): 
        dts = dates[i][0]
        tms = times[i][0]
        r = dts.split('-')
        t = tms.split(':')
        dtr = datetime.datetime(int(r[2]),int(r[1]),int(r[0]),int(t[0]))
        tick_dates.append([dtr.timestamp()])

    x_train,x_test,y_train,y_test = train_test_split(tick_dates,co_conc,test_size=0.2,random_state=0)
    reg = LinearRegression()
    reg.fit(x_train,y_train)
    y_predictions = reg.predict(x_test)

    print("x_train:",len(x_train))
    print("y_train:",len(y_train))
    print("y_test:",len(y_test))
    print("x_test:",len(x_test))
    print("y_predict:",len(y_predictions))
    print("coeff:",reg.coef_)
    print("intercept",reg.intercept_)
    print("R^2 on train set: {}".format(reg.score(x_train,y_train)))
    print("R^2 on test set: {}".format(reg.score(x_test,y_test)))

    plt.scatter(x_train,y_train,color="blue",label="train")
    plt.scatter(x_test,y_test,color="green",label="test")
    plt.plot(x_train,reg.predict(x_train),color="red",label="predict")
    plt.xlabel("timestamp")
    plt.ylabel("conc")
    plt.legend()
    plt.show()
print("--done--")


