#kütüphaneleri ekledik
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#veriyi ekledik
datas = pd.read_csv('tenis.csv')

#kategorik verileri nümerik hale çevirdik
from sklearn import preprocessing
outlook = datas.iloc[:,0:1].values
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

windy = datas.iloc[:,3:4].values
windy = ohe.fit_transform(windy).toarray()

play = datas.iloc[:,4:5].values
play = ohe.fit_transform(play).toarray() 

#verileri ayırdık
emperature = datas.iloc[:,1:2].values
humidity = datas.iloc[:,2:3].values

#verileri data frame formatına çevirdik
newOutlook = pd.DataFrame(data=outlook, index = range(14),columns=['overcast','rainy','sunny'])
newWindy = pd.DataFrame(data=windy[:,-1], index = range(14),columns=['windy'])
newPlay = pd.DataFrame(data=play[:,-1], index = range(14),columns=['play'])
newEmperature = pd.DataFrame(data=emperature, index = range(14),columns=['emperature'])
newHumidity = pd.DataFrame(data=humidity, index = range(14),columns=['humidity'])

#verileri birleştirdik
s = pd.concat([newOutlook,newEmperature,newWindy,newPlay], axis=1)

#verileri train ve test olarak ayırdık
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,newHumidity, test_size=0.33, random_state=0)

#verilerde tahminde bulunduk
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#backward elimination
import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int), values=s, axis=1)
X_l = s.iloc[:,[0,1,2,3,4,5]].values #tüm kolonlarımızı aldık 6 kolonumuz vardı burda indeks 0'dan başlıyor
X_l = np.array(X_l,dtype=float)
model = sm.OLS(newHumidity,X_l).fit() # bağımlı değişken boy ile bağımsız olan değişkenler arasında modeli kurduk
print(model.summary()) #modelin çıktısı

#windy verisinin p_value'su yüksek olduğu için onu eledik yeniden backward elimination yaptık
X = np.append(arr=np.ones((14,1)).astype(int), values=s, axis=1)
X_l = s.iloc[:,[0,1,2,3,5]].values #tüm kolonlarımızı aldık 6 kolonumuz vardı burda indeks 0'dan başlıyor
X_l = np.array(X_l,dtype=float)
model2 = sm.OLS(newHumidity,X_l).fit() # bağımlı değişken boy ile bağımsız olan değişkenler arasında modeli kurduk
print(model2.summary()) #modelin çıktısı

#yeni bir tahminde bulunduk yine multiple linear regresssion ile
regressor.fit(x_train.iloc[:,:4],y_train) # iloc'ta [:,:4] ile 4.sütun hariç her sütunu almış olduk
y_pred2 = regressor.predict(x_test.iloc[:,:4])







