class insan:
    boy = 180
    def kosmak(self,b):
        return b - 10
    
ali = insan()
print(ali.boy)
print(ali.kosmak(32))
    
l = [1,2,3]
print(l[1])



"""
eğer class içerisindeki bir fonksiyona parametre veriyorsanız default olarak self parametresini de eklemelisiniz
fonksiyon bir class içerisinde değilse parametre olarak self vermenize gerek yok

"""