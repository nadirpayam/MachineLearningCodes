#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()
y_olcekli = np.ravel(sc2.fit_transform(Y.reshape(-1,1)))

#SVR 
from sklearn.svm import SVR

svr_reg = SVR(kernel='rbf') 
svr_reg.fit(x_olcekli,y_olcekli) #iki değer arasındaki bağlantıyı kurduk

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')

print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))

"""
*SVR Tanımı ve Problem (Destek Vektör Regresyonu - Support Vector Regression)
-Destek vektör regresyonunda amaç şudur: bir doğru vardır polinal veya doğrusal olabilir bu doğruya b dersek bu doğrunun önüne ve arkasına a ve c eklenerek bir margin oluştururulur ve bu margin aralığında kalan noktalar önem arz eder.
 > a b c gibi düşenebiliriz işte bu a-b ve b-c arasındaki noktalar önemlidir. margin'in dışındaki noktalarda hata olarak kabul edilir ve tahminde kullanılmaz. 
-çizilen en geçerli doğru margin değerini minimize eden doğrudur

*Neden verileri scale ediyoruz?
-SVR aşırı aykırı verilerle çalışamıyor bu yüzden verileri scaler ettik 15.satırda, 
 svr'de scaler kullanmak zorunlu yani

*SVR paremetreleri
-kernel parametresinde default olarak rbf kullanırız 
 radial busses function'dır fonksiyonun türünü belirl burada, 
 istersek polinomal yapabiliriz
-diğer kernel türleri
 > SVR(kernel='poly', degree=3) # degree, polinom derecesini belirtir (varsayılan değer 3'tür)
 > SVR(kernel='sigmoid'): Sigmoidal çekirdek, veri noktalarını sigmoidal fonksiyonlar kullanarak dönüştürür.
   Bu çekirdek genellikle sınıflandırma problemlerinde kullanılır.
"""