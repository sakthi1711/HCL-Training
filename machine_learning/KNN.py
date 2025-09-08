import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer  
from sklearn.preprocessing import StandardScaler 

cancer = load_breast_cancer()

x = pd.DataFrame(cancer.data, columns=cancer.feature_names) 

y = pd.Series(cancer.target) 

print("Feature names:", cancer.feature_names)
print("Target names:", cancer.target_names) # 0: Malignant, 1: Benign
print("Shape of features (x):", x.shape)
print("Shape of target (y):", y.shape)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x) 

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)
print(y_pred)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

