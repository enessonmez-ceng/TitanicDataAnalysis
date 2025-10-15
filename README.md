
**Titanic: Hayatta Kalma Tahmini Projesi**

Bu proje, Kaggle'ın "Titanic - Machine Learning from Disaster" veri seti
kullanılarak gerçekleştirilmiş bir veri analizi ve makine öğrenmesi
çalışmasıdır. Projenin amacı, yolcu verilerinden yola çıkarak bir
yolcunun hayatta kalıp kalmayacağını tahmin eden bir model
geliştirmektir.

**Veri Setinin Hikayesi**

Bu proje, 15 Nisan 1912'de bir buzdağına çarparak batan RMS Titanic
gemisinin yolcu verilerini kullanmaktadır. Veri seti, hangi yolcuların
bu trajik kazadan sağ kurtulduğunu ve hangilerinin kurtulamadığını
içermektedir. Tarihin en bilinen denizcilik facialarından biri olan bu
olay, "kadınlar ve çocuklar önce" gibi sosyal protokollerin ne ölçüde
uygulandığını ve sosyal statü gibi faktörlerin hayatta kalma üzerindeki
etkisini analiz etmek için zengin bir zemin sunar. Bu veri seti,
genellikle sınıflandırma problemlerine giriş yapmak ve temel veri bilimi
adımlarını uygulamak için bir standart olarak kabul edilir.

**Veri Setindeki Değişkenler**

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

**🔍 Temel Veri Analizi (EDA)**

**Özet İstatistikler ve Eksik Değer Analizi**

Veri seti ilk yüklendiğinde info() ve describe() metodları ile temel bir
inceleme yapıldı.

-   **Özet İstatistikler:** Yolcuların yaş ortalamasının yaklaşık 29.7
    olduğu, ancak yaş verilerinde standart sapmanın yüksek olduğu
    görüldü. Bilet ücretleri (Fare) arasında da ciddi bir dağılım farkı
    mevcuttu, bu da farklı ekonomik sınıflardaki yolcuları
    yansıtmaktadır.

-   **Eksik Değerler:** En belirgin eksiklikler Age (Yaş) ve Cabin
    (Kabin) sütunlarındaydı. Cabin sütunundaki eksiklik oranı çok yüksek
    olduğu için modellemede kullanımı zor olarak değerlendirildi.
    Embarked sütununda ise çok az sayıda eksik veri tespit edildi.

**Değişken Dağılımları ve Aykırı Değerler**

-   **Kategorik Değişkenler:** Yolcuların çoğunluğunun erkek, 3. sınıfta
    seyahat eden ve Southampton limanından binen kişilerden oluştuğu
    görüldü.

-   **Sayısal Değişkenler:** Age dağılımı, en yoğun yolcu grubunun 20-35
    yaş arası genç yetişkinler olduğunu gösterdi. Fare dağılımı ise sağa
    çarpık bir yapıdaydı; yani yolcuların büyük çoğunluğu düşük ücretler
    öderken, çok az sayıda yolcu aşırı yüksek ücretler ödemişti. Bu
    yüksek ücretler, aykırı değer (outlier) olarak değerlendirilebilecek
    potansiyele sahipti.

**📊 Veri Görselleştirme**

Proje kapsamında veri içindeki desenleri ve ilişkileri daha iyi anlamak
için çeşitli görselleştirmeler yapıldı:

-   **Tek Değişkenli Analiz:** Age ve Fare için **histogramlar**
    kullanılarak dağılımları incelendi. Sex, Pclass ve Embarked için ise
    **bar grafikleri** ile yolcu sayıları görselleştirildi.

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

**🧹 Veri Ön İşleme**

Modeli eğitmeden önce veri seti üzerinde aşağıdaki ön işleme adımları
uygulandı:

-   **Eksik Verileri Doldurma:**

    -   Age sütunundaki eksik değerler, yolcu sınıflarına göre medyan
        yaş değeri ile dolduruldu.

    -   Embarked sütunundaki çok az sayıdaki eksik veri, en sık görülen
        liman (mod) ile dolduruldu.

    -   Cabin sütunu, aşırı eksik veri içerdiği için modelden çıkarıldı.

-   **Kategorik Değişkenlerin Dönüştürülmesi (Encoding):**

    -   Sex ve Embarked gibi kategorik değişkenler, modelin
        anlayabileceği sayısal formata dönüştürmek için **One-Hot
        Encoding** tekniği ile işlendi.

-   **Özellik Mühendisliği (Feature Engineering):**

    -   SibSp ve Parch sütunları birleştirilerek FamilySize adında yeni
        bir özellik türetildi.

**🤖 Basit Bir Modelleme**

**Model ve Değerlendirme Yöntemi**

Bu sınıflandırma problemi için temel bir başlangıç modeli olarak
**Lojistik Regresyon (LogisticRegression)** tercih edildi. Modelin
performansını objektif bir şekilde ölçmek için veri seti, **%80 eğitim**
ve **%20 test** seti olacak şekilde ikiye ayrıldı. Model eğitim
verileriyle eğitildi ve daha önce görmediği test verileri üzerinde
değerlendirildi.

**Başarı Metrikleri**

Test seti üzerinde elde edilen sonuçlar aşağıdaki gibidir:

-   **Doğruluk Oranı (Accuracy):** **0.80**

    -   Model, test verisindeki yolcuların %80'inin hayatta kalıp
        kalmayacağını doğru tahmin etmiştir.

-   **Hata Matrisi (Confusion Matrix):**

-   \[\[91 14\]

-   \[21 53\]\]

    -   Model, hayatta kalamayan 91 kişiyi ve hayatta kalan 53 kişiyi
        doğru sınıflandırmıştır.

    -   Hayatta kalan 21 kişiyi yanlışlıkla "hayatta kalamaz" olarak
        tahmin etmiştir.

-   **Sınıflandırma Raporu:**

-   precision recall f1-score support

-   0 (Kalamadı) 0.81 0.87 0.84 105

-   1 (Kaldı) 0.79 0.72 0.75 74

    -   Rapor, modelin hayatta kalamayanları tespit etmede (recall=0.87)
        daha başarılı olduğunu, ancak hayatta kalanları tespit etmede
        (recall=0.72) biraz daha zayıf kaldığını göstermektedir.

**📈 Sonuçların Yorumlanması**

**Modelin Değerlendirmesi**

Elde edilen **%80 doğruluk oranı**, Lojistik Regresyon gibi basit bir
model için oldukça başarılı bir sonuçtur. Model, özellikle bir yolcunun
sosyo-ekonomik durumu (Pclass) ve cinsiyeti (Sex) gibi güçlü
göstergelere dayanarak tutarlı tahminler yapabilmektedir. Modelin en
büyük zayıflığı, hayatta kalan bazı yolcuları tespit edememesidir (düşük
recall değeri). Bu durum, hayatta kalmanın daha karmaşık ve modelin
yakalayamadığı başka faktörlere de bağlı olabileceğini düşündürmektedir.

**Ne Öğrenildi?**

Bu proje, bir veri bilimi projesinin temel yaşam döngüsünü deneyimlemek
için harika bir fırsat sundu.

1.  **Veri Keşfinin Gücü:** Veriyi görselleştirmenin, ham sayılarda
    gizli olan sosyal dinamikleri (örn: sınıf ve cinsiyetin önemi) nasıl
    ortaya çıkardığını öğrendim.

2.  **Ön İşlemenin Önemi:** Bir makine öğrenmesi modelinin başarısının,
    büyük ölçüde verinin ne kadar iyi temizlendiği ve hazırlandığına
    bağlı olduğunu anladım.

3.  **Metriklerin Dili:** Doğruluk oranının ötesinde precision ve recall
    gibi metriklerin, bir modelin güçlü ve zayıf yönlerini nasıl daha
    detaylı anlattığını tecrübe ettim.

4.  **Tarihsel Veriden Anlam Çıkarma:** Son olarak, bu çalışma, tarihsel
    bir veri setinin modern analitik tekniklerle nasıl
    incelenebileceğini ve insan davranışları hakkında nasıl anlamlı
    sonuçlar çıkarılabileceğini gösterdi.
