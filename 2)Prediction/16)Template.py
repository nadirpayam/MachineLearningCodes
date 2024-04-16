#kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#veri yükleme
datas = pd.read_csv('maaslar.csv')

#data frame dilimleme
x = datas.iloc[:,1:2] 
y = datas.iloc[:,2:]

#numpy dizi(array) dönüşümü
X = x.values
Y = y.values

#linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#polynomial regression
#doğrusal modelmayan (nonlinear model) oluşturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

#4.dereceden polinom
poly_reg3 = PolynomialFeatures(degree=4)
x_poly3 = poly_reg3.fit_transform(X)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

#görselleştirme
plt.scatter(X,Y, color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

plt.scatter(X,Y, color='red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)),color='blue')
plt.show()

#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

