#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#veri yükleme
datas = pd.read_csv('veriler.csv')

ulke = datas.iloc[:,0:1].values


from sklearn import preprocessing

#labelEncoder ile ülkeleri 0,1,2 diye sıraladık
le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(datas.iloc[:,0])
"""
#oneHotEncoder her ülke için bir sütun oluşturdu kişinin ülkesine 1 diğerlerine 0 atadı
ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()
"""

"""
LabelEncoder ve OneHotEncoder, kategorik verilerin makine öğrenimi algoritmalarına uygun hale getirilmesinde sıklıkla kullanılan araçlardır. 
İkisi de kategorik verileri sayısal değerlere dönüştürmek için kullanılır, ancak farklı amaçlar için tasarlanmışlardır ve farklı sonuçlar 
üretirler.

LabelEncoder:

LabelEncoder, kategorik verileri sıralı veya ordinal sayısal değerlere dönüştürmek için kullanılır.
Örneğin, "kırmızı", "mavi", "yeşil" gibi sıralı kategorik değerlerinizi sırasıyla 0, 1, 2 gibi sayısal değerlere dönüştürebilir.
LabelEncoder, tek bir sütunu (değişkeni) işler ve her kategoriye tek bir sayısal değer atar.
Bu nedenle, sıralı olmayan (ordinal olmayan) kategorik verilerle kullanıldığında, modelin yanlış bir eğitim almasına neden olabilir, 
çünkü kategori arasındaki ilişkiyi ifade etmez.
OneHotEncoder:

OneHotEncoder, kategorik verileri ikili (binary) formatına dönüştürmek için kullanılır.
Her kategori için yeni bir sütun oluşturur ve bu sütunlardan sadece biri 1 (varlık) diğerleri 0 (yokluk) değerini alır.
Bu, kategorik verilerin arasında hiyerarşik veya sıralı bir ilişki olup olmadığına bakılmaksızın kullanılabilir.
Her kategori, birbirinden bağımsız olarak ele alınır.
Bu dönüşüm, özellikle kategorik verilerin modelin doğru bir şekilde eğitilmesi için daha uygun olduğu durumlarda tercih edilir.
Özetle, LabelEncoder kategorik verileri tek bir sütunda ordinal sayısal değerlere dönüştürmek için kullanılırken,
 OneHotEncoder kategorik verileri ikili (binary) formatına dönüştürmek için kullanılır ve her kategoriye ayrı bir sütun atar.
 
"""