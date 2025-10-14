from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from main import get_data

data = get_data()

y = data["Survived"]

X = data.drop(["Survived"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Eğitim seti boyutu (X_train):", X_train.shape)
print("Test seti boyutu (X_test):", X_test.shape)

# Lojistik Regresyon modelini oluşturuyoruz.
# max_iter=1000, modelin en iyi sonuca ulaşmak için yapacağı deneme sayısını artırır, bu genellikle hataları önler.
model = LogisticRegression(max_iter=1000)

# Modeli eğitim verileri ile eğitiyoruz. '.fit()' öğrenme işleminin gerçekleştiği yerdir.
model.fit(X_train, y_train)

print("\nModel başarıyla eğitildi!")

# Eğitilmiş modelden test verilerini kullanarak tahmin yapmasını istiyoruz.
y_pred = model.predict(X_test)

# 1. Doğruluk Oranı (Accuracy)
# Modelin yaptığı tahminlerin yüzde kaçının doğru olduğunu gösterir.
accuracy = accuracy_score(y_test, y_pred)
print(f"\nDoğruluk Oranı: {accuracy:.2f}") # Sonucu 2 ondalık basamakla yazdırır

# 2. Hata Matrisi (Confusion Matrix)
# Modelin hangi sınıfları birbiriyle karıştırdığını detaylıca gösterir.
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
# Matrisi okumak için:
# Sol-Üst: Gerçekte 0 olanları doğru tahmin etme sayısı (Doğru Negatif)
# Sağ-Alt: Gerçekte 1 olanları doğru tahmin etme sayısı (Doğru Pozitif)
# Sol-Alt: Gerçekte 1 olup 0 tahmin edilenler (Yanlış Negatif)
# Sağ-Üst: Gerçekte 0 olup 1 tahmin edilenler (Yanlış Pozitif)


# 3. Sınıflandırma Raporu (Classification Report)
# Precision, Recall ve F1-score gibi daha detaylı metrikler sunar.
report = classification_report(y_test, y_pred)
print("\nSınıflandırma Raporu:")
print(report)
