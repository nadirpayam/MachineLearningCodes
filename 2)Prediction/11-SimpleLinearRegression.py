import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

datas = pd.read_csv('satislar.csv')
aylar = datas[['Aylar']]
satislar = datas[['Satislar']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar,satislar, test_size=0.33, random_state=0)

#model inşası (linear regresyon)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train) 

#modelin uygulanması
tahmin = lr.predict(x_test)



"""
bu bölümde model inşasından önce veriyi hazır hale getirmek için aşağıdaki adımları yaptık
-veriyi yükledik
-bağımlı ve bağımsız kolonları ayırdık (aylar, satislar) (iloc ile de yapabilirdik bunu)
-verileri teset ve train olarak ayırdık
-ay verilerini sayısal olarak belirli bir aralığa getirdik (feature scaling)

model inşası
     modeli oluşturmak için train verilerini veririz örneğin burada X_train'i verdik burdaki ay verilerinin sonucundaki satislar verilerini
     Y_train'den aldı ve aralarındaki doğrusal bağlantıyı kurdu model
     sonrasında ise X_test verilerinden bize Y_test verilerini tahmin etmesini isteyeceğiz, modelin tahmin ettiği değerle Y_test v
     erilerini karşılaştırıp modelin başarısını ölçeceğiz
     
modelin uygulanması
    lr.predict
        burada parametre olarak verilen veriden tahminde bulunuruz (lr.fit ile eğittik modeli şimdi lr.predict ile tahminde bulunduk)
        x_test'i vererek bize tahmini y_test sonuçlarını vermesini istedik ve bunu tahmin değişkenine atadık
        sonra ise variable explorer'dan bizim orjinal y_test'imiz ile tahmin adlı veri setini karşılaştırabiliriz
        
ÖZET
    modeli x_train ve y_train verileriyle eğittik ardından x_test verisini verip bundan y_test verisini tahmin etmesini istedik
"""