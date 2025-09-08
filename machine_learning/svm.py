from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt


iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target

x = x[y < 2]
y = y[y < 2]

x_train, x_test, y_train, y_test =  train_test_split(x, y,test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print(f"Accuracy: {accuracy}")


plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='bwr', label='Train')
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap='bwr', marker='x', label='Test')

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("SVM Scatter Plot (Iris, 2 classes, linear kernel)")
plt.legend()
plt.show()