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

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0) 
r_dt.fit(X,Y)

plt.scatter(X, Y, color='red')
plt.plot(X,r_dt.predict(X), color = 'blue')

print(r_dt.predict([[7]]))
print(r_dt.predict([[9]]))

"""
*Karar Ağacı Kullanarak Tahmin Yöntemi
-Desicion Tree algoritması sınıflandırma için kullanılır genelde ama tahmin için de kullanılabilir.
-Örneği boy ve kilo'dan yaşı tahmin edelim
 1)Boy 145'e göre verileri böldük (büyük olanlar küçük 
   olanlar)
   1.1)Kilosu 75'den büyük olanlar ve küçük olanlar
      1.1.1)Boş(yaprak)
      1.1.2)Boy'u 165'ten büyük olanlar ve küçük olanlar
   1.2)Boş(yaprak)
-Yani burda anlatılmak istenen şu: verilerin dağılımına göre(bunu entrypo ile belirleyebiliriz) ikili uzayda veriyi belirlediğimiz değerden ikiye bölüyoruz sonra ordan 2'ye bölüyoruz ve böyle devam ediyor. En son ne kadar boş kutu yani yaprak kaldıysa ikili uzayda da bölme sayısı o kadar olmuştur.
-Daha sonra oluşan her kutuda yaş ortalamasını yazarız. Ve öğrenme süreci bitmiş olur.

-burada ölçekleme yapmamıza gerek yok pek
-random state: default parametredeir 0 verilip geçilebilir direkt
"""