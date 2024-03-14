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

y_pred = regressor.predict(x_test) 

#boy tahmini için model kuralım
boy = s2.iloc[:,3:4].values 
sol = s2.iloc[:,:3] 
sag = s2.iloc[:,4:] 
veri = pd.concat([sol,sag],axis=1) 

x_train, x_test, y_train, y_test = train_test_split(veri,boy, test_size=0.33, random_state=0)

r2 = LinearRegression()
r2.fit(x_train,y_train)
y_pred2 = r2.predict(x_test)

#backward elimination
import statsmodels.api as sm

X = np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1)
X_l = veri.iloc[:,[0,1,2,3,4,5]].values #tüm kolonlarımızı aldık 6 kolonumuz vardı burda indeks 0'dan başlıyor
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit() # bağımlı değişken boy ile bağımsız olan değişkenler arasında modeli kurduk
print(model.summary()) #modelin çıktısı

#4.sütunu eleyerek yolumuzda devam
X = np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1)
X_l = veri.iloc[:,[0,1,2,3,5]].values #tüm kolonlarımızı aldık 6 kolonumuz vardı burda indeks 0'dan başlıyor
X_l = np.array(X_l,dtype=float)
model = sm.OLS(boy,X_l).fit() # bağımlı değişken boy ile bağımsız olan değişkenler arasında modeli kurduk
print(model.summary()) #modelin çıktısı



"""
    Backward elimination
        1)Significance Level (SL) seçilir (genelde 0.05)
        2)Bütün değişkenler kullanılarak bir model inşa edilir (66.satır)
        3)En yüksek p-value değerine sahip olan değişken ele alınır ve şayet P>SL ise 4.adıma değilse 6.adıma gidilir
        4)Bu aşamda 3.adımda seçilen ve en yüksek p-value'ya sahip olan değişken sistemden kaldırılır
        5)Makine öğrenmesi güncellenir ve 3.adıma geri dönülür
        6)Makine öğrenmesi sonlandırılır
   
    np.append'te ne yaptık?
        şimdi çoklu doğrusal regresyonda y = b + b1x1 + b2X2 .... diye bir b sabiti var ancak bu tüm sayıların sabit olduğu bir sütun yok verilerimizde
        biz append ile 22 satırlık 1'lerden oluşan bir kolonu verilerimize dahil ettik 
    model.summary ile OLS Regression Result'a eriştik ve burda p_value değeri en büyük olan veri 4.kolonmuş görmüş olduk bu veriyi eledik
    
"""

