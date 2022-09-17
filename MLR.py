import pandas as pd
import numpy as np
import pickle

df = pd.read_csv('MLR.data')

x = df[['AT','V','AP','RH']].values
y = df['PE'].values


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
sv = model.fit(x_train,y_train)

pickle.dump(sv, open('MLR.pkl','wb'))