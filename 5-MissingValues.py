#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#veri yükleme
datas = pd.read_csv('eksikveriler.csv')

#SımpleImputer metodunu çağırdık
from sklearn.impute import SimpleImputer

#metodun parametrelerini verip bir değişkene atadık
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

#değişiklik yapacağımız sütunu verilerden çektik
age = datas.iloc[:,3:4].values

#fit ile modelin veriyi öğrenmesini sağladık
imputer = imputer.fit(age[:,0:1])

#transform ile değişiklikleri gerçekleştirdik
age[:,0:1] = imputer.transform(age[:,0:1])

print(age)

"""
SimpleImputer parametreleri:
    missing_values: verideki kayıp olarak belirlenecek değerleri seçeriz np.nan ile nan olan verileri kayıp veri olarak işaretledik
    strategy: kayıp verinin hangi stratejiyle doldurulacağını belirleriz, mean ile  sütunun ortalamasını kayıp verilere atamış olduk
    
iloc: iloc veri seçmede işe yarar 
    virgülden önceki kısım: hangi satırların seçileceğini belirler
    virgülden sonraki kısım: hangi sütunların seçileceğini belirler
    index sıfırdan başlar
    [1,2) mantığındadır yani 2.değer dahil değildir
"""