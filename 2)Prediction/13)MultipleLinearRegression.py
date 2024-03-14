import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

datas = pd.read_csv('veriler.csv')

#encoder aşaması: nominal veya ordinal verilerden numeric veri elde etme
ulke = datas.iloc[:,0:1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(datas.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

c = datas.iloc[:,-1:].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(datas.iloc[:,-1])
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()

Yas = datas.iloc[:,1:4].values
#data frame oluşturma ve data frame'leri birleştirme (yani numpy dizileri data frame'e dönüştürme işlemi)
result = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
result2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
cinsiyet = datas.iloc[:,-1].values
result3 = pd.DataFrame(data=c[:,:1], index = range(22), columns = ['cinsiyet'])
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

#multiple linear regression ile cinsiyet tahmini
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test) #test verisi verdik ve tahmin yapmasını istedik

#boy tahmini için model kuralım
boy = s2.iloc[:,3:4].values #boy verilerini aldık
sol = s2.iloc[:,:3] #boydan öncekileri aldık
sag = s2.iloc[:,4:] #boydan sonrakileri aldık
veri = pd.concat([sol,sag],axis=1) #boy olmadan verileri birleştirmiş olduk

x_train, x_test, y_train, y_test = train_test_split(veri,boy, test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred2 = r2.predict(x_test)


"""
    multiple linear regression ile cinsiyet tahmininin üstündeki kodlar veriyi hazırlamak için yazılan kodlardır
    regressor.fit'e parametre olarak x_train ve y_train'i verdik ve bu metot bu iki veri arasında bir bağlantı,model kuracak kendini eğitecek
    
    bu kod dosyasında yaptıklarımız
        verileri modele hazır hale getirdik
        multiple linear regresyon ile cinsiyet tahmininde bulunduk
        multiple linear regresyon ile boy tahmininnde bulunduk
"""

