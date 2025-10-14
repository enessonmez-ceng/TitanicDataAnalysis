import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("Data/titanic/train.csv")



sns.set_style('whitegrid')

plt.figure(figsize=(8, 6))

sns.countplot(x='Sex', hue='Survived', data=df)


plt.title('Cinsiyete Göre Hayatta Kalma Dağılımı', fontsize=16)
plt.xlabel('Cinsiyet', fontsize=12)
plt.ylabel('Kişi Sayısı', fontsize=12)


plt.show()


plt.figure(figsize=(10, 6))


sns.countplot(x='Pclass', hue='Survived', data=df)

plt.title('Yolcu Sınıfına Göre Hayatta Kalma Dağılımı', fontsize=16)
plt.xlabel('Yolcu Sınıfı', fontsize=12)
plt.ylabel('Kişi Sayısı', fontsize=12)
plt.legend(title='Hayatta Kalma Durumu', labels=['Kalamadı', 'Kaldı']) 

plt.show()


plt.figure(figsize=(12, 7))


sns.histplot(data=df, x='Age', hue='Survived', kde=True, multiple="stack")

plt.title('Yaşa Göre Hayatta Kalma Dağılımı', fontsize=16)
plt.xlabel('Yaş', fontsize=12)
plt.ylabel('Kişi Sayısı', fontsize=12)

plt.show()


pairplot_df = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]


sns.pairplot(pairplot_df, hue='Survived', diag_kind='kde')

plt.suptitle('Değişkenlerin İkili İlişkileri', y=1.02, fontsize=18)

plt.show()
