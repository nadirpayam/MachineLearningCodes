#kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

#veri yükleme
datas = pd.read_csv('veriler.csv')

#veriden belirli bir sütun almak için
boy = datas[['boy']]
boyveKilo = datas[['boy','kilo']]



"""

-Kodu çalıştırdığımız yer ile dosya aynı klasörde olmalı.

"""