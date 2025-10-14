import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("Data/titanic/train.csv")


# Grafiklerin daha estetik görünmesi için bir stil ayarı
sns.set_style('whitegrid')

# Grafiğin boyutunu ayarlayalım
plt.figure(figsize=(8, 6))

# Cinsiyete göre hayatta kalma durumunu gösteren bir countplot çizdirelim
# hue='Survived' parametresi, her cinsiyet çubuğunu 'Hayatta Kaldı' (1) ve 'Kalamadı' (0) olarak ikiye ayırır.
sns.countplot(x='Sex', hue='Survived', data=df)

# Grafiğe başlık ve etiketler ekleyelim
plt.title('Cinsiyete Göre Hayatta Kalma Dağılımı', fontsize=16)
plt.xlabel('Cinsiyet', fontsize=12)
plt.ylabel('Kişi Sayısı', fontsize=12)

# Grafiği göster
plt.show()


plt.figure(figsize=(10, 6))

# Pclass'a göre hayatta kalma durumunu gösteren bir countplot
sns.countplot(x='Pclass', hue='Survived', data=df)

plt.title('Yolcu Sınıfına Göre Hayatta Kalma Dağılımı', fontsize=16)
plt.xlabel('Yolcu Sınıfı', fontsize=12)
plt.ylabel('Kişi Sayısı', fontsize=12)
plt.legend(title='Hayatta Kalma Durumu', labels=['Kalamadı', 'Kaldı']) # Etiketleri daha anlaşılır yapalım

plt.show()


plt.figure(figsize=(12, 7))

# Yaş dağılımını hayatta kalma durumuna göre ayıran bir histogram
# kde=True, dağılımın üzerine yumuşak bir çizgi (yoğunluk eğrisi) ekler.
sns.histplot(data=df, x='Age', hue='Survived', kde=True, multiple="stack")

plt.title('Yaşa Göre Hayatta Kalma Dağılımı', fontsize=16)
plt.xlabel('Yaş', fontsize=12)
plt.ylabel('Kişi Sayısı', fontsize=12)

plt.show()

# Yalnızca ilgili sayısal sütunları ve hedef değişkeni seçelim
pairplot_df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]

# Seçtiğimiz sütunlar için bir pairplot çizdirelim
# hue='Survived' ile noktaları hayatta kalma durumuna göre renklendiriyoruz.
sns.pairplot(pairplot_df, hue='Survived', diag_kind='kde')

plt.suptitle('Değişkenlerin İkili İlişkileri', y=1.02, fontsize=18) # Ana başlık ekleyelim

plt.show()