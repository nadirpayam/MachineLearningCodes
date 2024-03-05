import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

datas = pd.read_csv('satislar.csv')
aylar = datas[['Aylar']]
satislar = datas[['Satislar']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar,satislar, test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train) 

tahmin = lr.predict(x_test)

#görselleştirme 
#verileri hizalama
x_train = x_train.sort_index()
y_train = y_train.sort_index()

#grafikleri çizme
plt.plot(x_train,y_train)
plt.plot(x_test, tahmin)

#grafikte front end tasarımı
plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")

"""
 bu kısımda oluşturduğumuz simple linear regresyon modelini görselleştirdik
 verileri hizalama:
     veriler train ve test olarak 10.satırda random olarak ayrılmıştı ve indexleri karışıktı biz veriyi indexine göre sıraladık ki
     grafik düzgün çizilsin
 grafikleri çizme:
     plt.plot metoduna parametre olarak x ve y değişkenlerini verdik ve grafik oluştu
     ilk çizilen grafik train verilerinin grafiğini oluşturdu
     ikinci grafik bizim modelde elde etmek istediğimiz doğruyu çizdi (çünkü tahmini buraya ekledik)

"""