import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

datas = pd.read_csv('veriler.csv')

from sklearn import preprocessing

ulke = datas.iloc[:,0:1].values
yas = datas.iloc[:,1:4].values
cinsiyet = datas.iloc[:,-1].values

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

result = pd.DataFrame(data=ulke, index = range(22),columns=['fr','tr','us'])
print(result)

result2 = pd.DataFrame(data=yas, index = range(22), columns = ['boy','kilo','yas'])
print(result2)

result3 = pd.DataFrame(data=cinsiyet, index = range(22), columns = ['cinsiyet'])

s = pd.concat([result,result2], axis=1)

s2 = pd.concat([s,result3],axis=1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,result3, test_size=0.33, random_state=0)

#feature scaling aşaması
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


"""
 Feature Selection:
     sayısal veriler bazen çok büyük aralıklarda olabiliyor mesela bir veri 30 iken diğer veri 200 olabiliyor bu da
     verilerin arasındaki bağı uzaklaştırıyor 
     feature selection ile verileri belirli bir sayı aralağına getirebiliyoruz mesela -1 ve 1 arasına getirebiliriz tüm verileri
"""
