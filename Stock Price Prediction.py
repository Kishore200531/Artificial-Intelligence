import pandas as pd
import matplotlib.pyplot as plt
#Loading Dataset
data = pd.read_csv("stock_data.csv")
df = pd.DataFrame(data)
print(df.head())
print(df.shape)
#Plotting Open & Close Price's
plt.plot(df['Open'])
plt.plot(df['Close'])
plt.title('Open Vs Close Price')
plt.legend(['Open', 'Close'])
plt.show()

#Checking that dataset has a Date column
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

#Splitting the dataset
from sklearn.model_selection import train_test_split
#Setting target price as predicted closing price
df['Target'] = df['Close'].shift(-1)
df = df.dropna()
x = df[['Open', 'High', 'Low']]
y = df['Target']

X_train,X_test,y_train,y_test = train_test_split(x, y,test_size=0.2, random_state= 42)
#print("X_train:",X_train.shape)
#print("y_train:",y_train.shape)
#print("X_test:",X_test.shape)
#print("y_test:",y_test.shape)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
#Model Creation
model = LinearRegression()
#Model Training
model.fit(X_train,y_train)
#Model Evaluation
prediction = model.predict(X_test)
print("MSE:",mean_squared_error(y_test,prediction))
print("R2 Score:",r2_score(y_test,prediction))
#Comparing Actual Price & Predicted Price
comparison = pd.DataFrame({'Date': y_test.index, 'Actual Price': y_test.values, 'Predicted Price': prediction })
comparison = comparison.sort_values(by='Date')
print(comparison.head())
#Plotting Actual Price Vs Predicted Price
plt.plot(y_test.values, label='Actual Price')
plt.plot(prediction, label='Predicted Price')
plt.legend()
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Samples")
plt.ylabel("Price")
plt.show()