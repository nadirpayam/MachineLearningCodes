#kütüphaneleri yükledik
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#verileri çağırdık
datas = pd.read_csv('maaslar.csv') #unvan,eğitim seviyesi,maaş

x = datas.iloc[:,1:2] #eğitim seviyesi sütununu aldık
y = datas.iloc[:,2:] #maaş sütununu aldık

#dataframe yerine verilerin değerlerini alalım tahmin yaparken lazım olacak
X = x.values
Y = y.values

#linear regresyonla bir deneyelim verileri göstermeyi
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y) #burda verileri fit fonksiyona vermemiz gerekti için X,Y attık içine çünkü x,y'de veriler dataframe formatında

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures  #class'ı import ettik
poly_reg = PolynomialFeatures(degree=4) #nesneyi oluşturduk
x_poly = poly_reg.fit_transform(X) #X verilerini linear dünyadan polinomal dünyaya aktardık

#polinomal dünyada çevirdiğimiz verileri linear regression'a sokalım
lin_reg2 = LinearRegression() 
lin_reg2.fit(x_poly,y) #burda polinomal dünyadaki x verilerine baksın, y verileriyle arasında ilişki kursun
plt.scatter(X, Y, color="red") #X, Y verilerine göre grafiğin yatay ve dikey çizgilerini oluşturduk
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)),color="blue") #x koordinatı olarak X, y koordinatı olarak bizim tahmin ettirdiğimiz X'i verdik
plt.show()

"""
     Mantık: Bazı verilerden yola çıkarak bir veriyi tahmin ederiz bu aşamada bir değerden bir değeri tahmin ediyorsak basit doğrusal regresyon,
             bir değeri birden fazla değerle tahmin edersek çoklu doğrusal regresyon kullanıyorduk. Polinomal regresyonda ise bir değeri birden fazla
             değerden veya tek bir değerden tahmin edebiliriz ancak burda veriler arasında polinomal bir artış olduğu durumlarda polinomal regresyon kullanırız.
     PolynomialFeatures: Herhangi bir sayıyı polinomal olarak ifade etmeye yarar
     
     degree mantığı
         X verilerimiz 1-10 arasındaki sayılardan oluşuyordu, bizim verdiğimiz dereceye göre X'in formülünün çarpanları arasında bir bağlantı kurulur
         örneğin 2 verdiysek dereceye: 2 4  gibi her değerin 1 ve 2.derecesi alınır ve böyle formülün çarpanları arasındaki bağlantı bulunur
        dereceyi arttırınca elde edeceğimiz veriler çoğalacağı çok olacağı için aradaki bağlantı daha iyi olur ancak çok fazla derece verirsek de
        ezberleme olayı ortaya çıkar, genelde 3-5 arası derece vermek idealdir.
     
  
"""