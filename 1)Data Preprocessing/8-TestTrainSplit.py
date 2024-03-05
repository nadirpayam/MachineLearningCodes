#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#veri yükleme
datas = pd.read_csv('veriler.csv')

#ön işleme
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

#verileri test ve train olarak bölme aşaması
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,result3, test_size=0.33, random_state=0)


"""
ülke,boy,kilo ve yaştan cinsiyeti tahmin etmeye çalıştığımız için ülke,boy,kilo,yas'ı ayrı cinsiyet'i ayrı bir frame'de tutmalıyız
x bağımsız değişkenler, y bağımlı değişkenlerdir
belirli bir satıra kadar olan veriler train, sonrası test olarak bölünür
x ve y bölünme mantığıysa bağımlı ve bağımsız olarak bölünmesidir verilerin

train_test_split metotu parametreleri
    ilk parametre s= tahmin ederken kullanacağımız sütünların frame'i
    ikinci parametre result3= tahmin edilecek olan cinsiyet verilerinin tutulduğu frame
    test_size parametresi = test ve train olarka verinin bölünme oranı %67 test, %33 test olarak bölünür 0.33 ile
    random_state parametresi= verinin test train olarak random şekilde bölünmesini sağlar 0 verdiğimiz için rastsal olarak sıfırdan bölmeye başlar

veriyi öncelikle dikey eksende bağımlı ve bağımsız olarak ayırıyoruz (x,y)
sonra yatay eksende train ve test kümeleri olarak ayırıyoruz böylelikle toplamda 4 kümemiz çıkmış oluyor

x_train ve y_train - x_test  ve y_test ikilileri
-x_train'te verileri öğrenmesi için bağımsız değişkenleri verdik
-y_train'te verileri öğrenmesi için bağımsız değişkenlere göre belirlenen bağımlı değişkeni verdik

-x_test'te tahmin edebilmesi için bağımsız değişkenleri verdik
-y_test'te tahmin ettiği sonuç bizim verdiğimiz y_test ile aynı mı diye kontrol ettik
"""