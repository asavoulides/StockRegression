import requests
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import numpy as np

# Replace YOUR_API_KEY with your actual API key
api_key = "ZOFF8JEDW0TCPKSS"

# Specify the symbols for the two stocks you want to compare
symbol1 = input("Stock 1: ")
symbol2 = input("Stock 2: ")

# Send a GET request to the AlphaVantage API
response = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol1}&apikey={api_key}")

# Extract the time series data for each stock from the response
data1 = response.json()["Time Series (Daily)"]
data2 = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol2}&apikey={api_key}").json()["Time Series (Daily)"]

# Convert the time series data into lists of close prices
close_prices1 = [float(data1[date]["4. close"]) for date in data1]
close_prices2 = [float(data2[date]["4. close"]) for date in data2]

# Calculate the returns for each stock
returns1 = []
for i in range(1, len(close_prices1)):
  return1 = (close_prices1[i] - close_prices1[i - 1]) / close_prices1[i - 1]
  returns1.append(return1)

returns2 = []
for i in range(1, len(close_prices2)):
  return2 = (close_prices2[i] - close_prices2[i - 1]) / close_prices2[i - 1]
  returns2.append(return2)

# Calculate the Pearson correlation coefficient and the p-value
r, p = pearsonr(returns1, returns2)

# Convert the returns to NumPy arrays
X = np.array(returns1).reshape(-1, 1)
y = np.array(returns2)

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the R^2 value and the slope of the line
r_squared = model.score(X, y)
slope = model.coef_[0]

# Predict the returns of the second stock based on the returns of the first stock
y_pred = model.predict(X)

# Plot the data and the line of best fit
plt.scatter(returns1, returns2)
plt.plot(returns1, y_pred, color='red')
plt.title(f"{symbol1} vs. {symbol2} (R^2 = {r_squared:.2f}, slope = {slope:.2f})")
plt.xlabel(symbol1)
plt.ylabel(symbol2)
#plt.legend([symbol1, symbol2])
plt.show()
