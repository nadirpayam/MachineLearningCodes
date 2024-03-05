#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#veri yükleme
datas = pd.read_csv('veriler.csv')

#ön işleme
from sklearn import preprocessing

#sütunları çektik
ulke = datas.iloc[:,0:1].values
yas = datas.iloc[:,1:4].values
cinsiyet = datas.iloc[:,-1].values

#ulke sütununa onehotencoder uyguladık
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

#ulke sütununu dataframe'e çevirdik
result = pd.DataFrame(data=ulke, index = range(22),columns=['fr','tr','us'])
print(result)

#yas sütununu dataframe'e çevirdik
result2 = pd.DataFrame(data=yas, index = range(22), columns = ['boy','kilo','yas'])
print(result2)

#cinsiyet sütunun data frame'e çevirdik
result3 = pd.DataFrame(data=cinsiyet, index = range(22), columns = ['cinsiyet'])
print(result3)

#concat ile iki dataframe'i birleştirdik
s = pd.concat([result,result2], axis=1)
print(s)

s2 = pd.concat([s,result3],axis=1)
print(s2)

"""
Bu derste dataframe oluşturmayı gördük, normalde consolda veriler array şeklindedir indexi gözükmez sutun adı gözükmez ama data frame'de öyle değil

Adımlar
-önce dataframe'e çevireceğimiz veriyi çekeriz mesela ulke
-sonra pd.DataFrame ile o veriyi dataframe'e çeviririz

DataFrame'leri birleştimre
-pd.concat ile iki data frame'i birleştirebiliriz
-axis parametresini 0 verirsek alt alta, 1 verirsek yan yana birleşir data frame'ler

DataFrame parametreleri
    range: kaç satırını alacağımızı belirleriz
    columns: oluşacak sütun isimlerini belirleriz
"""