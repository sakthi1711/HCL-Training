import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {'Area':[1000,1500,2000,2500,3000],
        'Bedrooms':[2,3,4,4,5],
        'Price':[200000,250000,300000,350000,400000]}
df = pd.DataFrame(data)

X = df[['Area']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

# Plot regression line
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()

model = LinearRegression()
model.fit(X_train,y_train)

print("Predicted price:", model.predict([[2200,3]]))
