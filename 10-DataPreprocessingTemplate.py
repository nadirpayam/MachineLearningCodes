#1)Kütüphanelerin yüklenmesi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#2)Veri ön işleme 

#verilerin yüklenmesi
datas = pd.read_csv('eksikveriler.csv')

#eksik verileri düzeltme 
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
Yas = datas.iloc[:,1:4].values
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])

#encoder aşaması: nominal veya ordinal verilerden numeric veri elde etme
ulke = datas.iloc[:,0:1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(datas.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

#data frame oluşturma ve data frame'leri birleştirme (yani numpy dizileri data frame'e dönüştürme işlemi)
result = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
result2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
cinsiyet = datas.iloc[:,-1].values
result3 = pd.DataFrame(data=cinsiyet, index = range(22), columns = ['cinsiyet'])
s = pd.concat([result,result2], axis = 1)
s2 = pd.concat([s,result3], axis = 1)

#verileri train ve test olarak ayırma işlemi
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s,result3, test_size=0.33, random_state=0)

#feature scaling işlemi
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)