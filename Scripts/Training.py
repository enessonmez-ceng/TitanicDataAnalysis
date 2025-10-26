from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from main import get_data

data = get_data()

y = data["Survived"]

X = data.drop(["Survived"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Eğitim seti boyutu (X_train):", X_train.shape)
print("Test seti boyutu (X_test):", X_test.shape)

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel başarıyla eğitildi!")


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"\nDoğruluk Oranı: {accuracy:.2f}") 


cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

report = classification_report(y_test, y_pred)
print("\nSınıflandırma Raporu:")
print(report)
