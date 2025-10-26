
# Titanic: Hayatta Kalma Tahmini Projesi

Bu proje, Kaggle'ın "Titanic - Machine Learning from Disaster" veri seti
kullanılarak gerçekleştirilmiş bir veri analizi ve makine öğrenmesi
çalışmasıdır. Projenin amacı, yolcu verilerinden yola çıkarak bir
yolcunun hayatta kalıp kalmayacağını tahmin eden bir model
geliştirmektir.

## Veri Setinin Hikayesi

Bu proje, 15 Nisan 1912'de bir buzdağına çarparak batan RMS Titanic
gemisinin yolcu verilerini kullanmaktadır. Veri seti, hangi yolcuların
bu trajik kazadan sağ kurtulduğunu ve hangilerinin kurtulamadığını
içermektedir. Tarihin en bilinen denizcilik facialarından biri olan bu
olay, "kadınlar ve çocuklar önce" gibi sosyal protokollerin ne ölçüde
uygulandığını ve sosyal statü gibi faktörlerin hayatta kalma üzerindeki
etkisini analiz etmek için zengin bir zemin sunar. Bu veri seti,
genellikle sınıflandırma problemlerine giriş yapmak ve temel veri bilimi
adımlarını uygulamak için bir standart olarak kabul edilir.

## Veri Setindeki Değişkenler

-   Survived: Hedef değişken. Yolcunun hayatta kalıp kalmadığını
    belirtir. (**0** = Hayır, **1** = Evet)

-   Pclass: Yolcu sınıfı. Sosyo-ekonomik durumu temsil eder. (**1** = 1.
    Sınıf, **2** = 2. Sınıf, **3** = 3. Sınıf)

-   Sex: Yolcunun cinsiyeti.

-   Age: Yolcunun yaşı.

-   SibSp: Gemideki kardeş veya eş sayısı.

-   Parch: Gemideki ebeveyn veya çocuk sayısı.

-   Fare: Yolcunun ödediği bilet ücreti.

-   Embarked: Yolcunun gemiye bindiği liman. (**C** = Cherbourg, **Q** =
    Queenstown, **S** = Southampton)

-   Cabin: Yolcunun kabin numarası.

## 🔍 Veri Önişleme ve Özellik Mühendisliği

**1.Eksik Değerler**
- 'Cabin' sütununda çok fazla eksik değer olduğu için modellemeye dahil edilmedi.
-   Veri setinde 'Embarked' sütunundaki eksik değerler için en çok tekrar eden veriyle değiştirildi(mod işlemi).
-   'Age' sütunundaki eksik değerleri doldurmak için tüm yolcuların ortalama yaşını kullanmak yerine 3 farklı 'pclass' için yaş ortalaması alıp yolcu hangi böümdeyse o bölümün yaş ortalaması kullanıldı.Böylece eksik veriler daha doğru bir biçimde dolduruldu.  




**2.Aykırı Değerler**

-   'Age' ve 'Fare' sütunlarındaki değerlerin aralığı geniş , aykırı değerlerin fazla olduğu tespit edildi. Bu yüzden **capping(baskılama)** metodu kullanılarak alt ve üst sınırlar belirlendi.
-   'Age' sütunu için aykırı değer sınırları: Alt=-0.50, Üst=59.50
-   'Fare' sütunu için aykırı değer sınırları: Alt=-26.72, Üst=65.63
-   **Not** : Alt değerin negatif bir değer çıkması, her iki verinin de minimum *0* değerini alabileceği için herhangi bir yanlış hesaplamaya yol açmaz.

**3.Özellik Mühendsiliği**
-    'SibSp' ve 'Parch' sütunlarını birleştirerek 'FamilySize' (Aile Büyüklüğü) adında yeni bir sütun oluşturarak *Multicollinearity* probleminin önüne geçmeye çalışıldı.'FamilySize' sütunundaki verilerden yola çıkarak yolcuların yalnız seyahat etme durumunu gösteren 'IsAlone' sütunu oluşturuldu. 

**4.Kategorik Değişkenleri Sayısal Değerlere Çevirme**
- Makine öğrenmesi modelleri sadece sayısal değerlerle hesaplama yapabildiği için hesaplamalarda kullanacağımız tüm sütunların sayısal veriden oluşması gerekir.
- Bu yüzden 'Sex' sütunundaki değerler("female": *0*, "male": *1*) ve 'Embarked' sütunundaki değerler sayısal değerlere çevrildi("C": *0*, "S": *1*, "Q": *2*).


## 🤖 Model Eğitimi

-Bu sınıflandırma problemi için temel bir başlangıç modeli olarak **Lojistik Regresyon (LogisticRegression)** tercih edildi.
- Veri seti kaggle'dan indirdiğimiz haliyle zaten train ve test olarak iki ayrı csv dosyasına sahip olmasına rağmen bu haliyle kullanılmamış, doğrudan **train.csv** dosaysındaki veriler python kodunda 60/40 oranında manuel olarak eğitim ve test verisi olarak ayrılmıştır.

### Modelin Değerlendirilmesi
**Hata Matrisi**

$$
A = \begin{bmatrix}
93 & 12 \\
19 & 55 \\
\end{bmatrix}
$$

-   Model, hayatta kalamayan 93 kişiyi ve hayatta kalan 55 kişiyi doğru sınıflandırmıştır.

-   Hayatta kalan 19 kişiyi yanlışlıkla "hayatta kalamaz" olarak  tahmin etmiştir.

**Doğruluk Oranı(Accuracy)** : **0.83**
- Model, test verisindeki yolcuların %83'ünün hayatta kalıp kalmayacağını doğru tahmin etmiştir.

**Sınıflandırma Raporu:**

| Survived| precision | recall | f1-score | support |
|---|---|---|---|---|
| 0| 0.83 |  0.89 |0.86  |105 |
| 1 | 0.82 | 0.74|0.78 |74 |


-   Rapor, modelin hayatta kalamayanları tespit etmede (recall=0.89)
        daha başarılı olduğunu, ancak hayatta kalanları tespit etmede
        (recall=0.74) biraz daha zayıf kaldığını göstermektedir.

**Modelin Değerlendirmesi**

Elde edilen **%83 doğruluk oranı**, Lojistik Regresyon gibi basit bir
model için oldukça başarılı bir sonuçtur. Model, özellikle bir yolcunun
sosyo-ekonomik durumu (Pclass) ve cinsiyeti (Sex) gibi güçlü
göstergelere dayanarak tutarlı tahminler yapabilmektedir. Modelin en
büyük zayıflığı, hayatta kalan bazı yolcuları tespit edememesidir (düşük
recall değeri). Bu durum, hayatta kalmanın daha karmaşık ve modelin
yakalayamadığı başka faktörlere de bağlı olabileceğini düşündürmektedir.



**📊 Veri Görselleştirme**

Proje kapsamında **visualization.py** dosyasında çeşitli görselleştirmeler yapıldı:

-   **Çok Değişkenli Analiz:**

    -   **Cinsiyet ve Hayatta Kalma:** Cinsiyete göre hayatta kalma
        oranlarını gösteren grafik, kadınların hayatta kalma şansının
        erkeklere göre çok daha yüksek olduğunu net bir şekilde ortaya
        koydu.

    -   **Sınıf ve Hayatta Kalma:** Yolcu sınıfına göre hayatta kalma
        oranları incelendiğinde, 1. sınıf yolcuların hayatta kalma
        oranının 3. sınıf yolculara kıyasla belirgin şekilde daha yüksek
        olduğu gözlemlendi.

    -   **Yaş ve Hayatta Kalma:** Yaş dağılımı grafiği, özellikle küçük
        çocukların hayatta kalma oranının diğer yaş gruplarına göre daha
        yüksek olduğunu gösterdi.


**Projenin bana kattıkları**
- Bu projede veri önişleme adımlarını doğru sırayla uygulayarak veri setini adeta *modelin anlayacağı dile* dönüştürmeyi öğrendim.
- Yaş,cinsyet gibi kategorik verilerin hayatta kalma oranına nasıl bir etkisi olduğunu görselleştirme yaparak analiz etmeyi kavradım.
- Lineer regresyon modeliyle yolcuların *hayatta kalma* durumunu tahmin etmeye çalıştım.
- Genel olarak baktığımda veri bilimi alanında ilk mini projemi gerçekleştirdim diyebilirm.
