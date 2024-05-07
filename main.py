import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
from bs4 import BeautifulSoup # Web Scraping
import pandas as pd
from pandas_datareader import data as pdr
from selenium import webdriver
import time
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import copy
import random

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

#get the URL using response variable
financials_url = "https://finance.yahoo.com/most-active?count=100"

response = requests.get(financials_url)
url_contents = response.text
with open('yahoo-finance-most-active.html', 'w', encoding="utf-8") as file:
    file.write(url_contents)

with open('yahoo-finance-most-active.html', 'r') as f:
    html_source = f.read()

#Generalize this task with a function so that it can be used for multiple types of securities
def download_web_page(url):
    response = requests.get(url)
    status_code = response.status_code
    url_contents = response.text
    if status_code in range(200,300):
        url_contents = response.text
        with open('new-downloaded-web-page.html', 'w', encoding="utf-8") as file:
            file.write(url_contents)
        print('Status code is within an okay range of {}.'.format(status_code))
        
    else: 
        return

doc = BeautifulSoup(html_source, 'html.parser')


title = doc.title
print(title.text)

tr_class_tags = doc.find_all('tr',class_='simpTblRow')
tr_class_tags[:2]
tr_tag_amount = len(tr_class_tags)

tr_class_tag1 = tr_class_tags[0]
td_tag = tr_class_tag1.find_all('td')
a_tag = td_tag[0].find_all('a', recursive=False)
ticker_name = a_tag[0].text.strip()

    

def parse_stocks(tr_class_tag):
    # <td> tags contain all of the stock info, <tr tags contain all of the individual details, <a> tags contain ticker name
    td_tag = tr_class_tag.find_all('td')
    a_tag = td_tag[0].find('a', recursive=False)
    # Stock ticker
    ticker_name = a_tag.text.strip()

    
    # Return a dictionary
    return {
        'Stock ticker': ticker_name,
    }

def list_tickers(tr_class_tag):
    td_tag = tr_class_tag.find_all('td')
    a_tag = td_tag[0].find('a', recursive=False)
    # Stock ticker
    ticker_name = a_tag.text.strip()
    return ticker_name

stock_tickers = [list_tickers(x) for x in tr_class_tags]

yFinanceData = []
print(yFinanceData)
for x in tr_class_tags:  
    yFinanceData.append(list_tickers(x))

#Load the Data
unprocessedData = yf.download(yFinanceData, period="60mo")
unprocessedAdjCloseData = unprocessedData["Adj Close"]
#Preprocess Data
data = unprocessedAdjCloseData.dropna()

# Normalize numerical features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split data into X (features) and y (target)
X = scaled_data[:, :-1]  # Input features (excluding 'Close')
y = scaled_data[:, -1]   # Target variable ('Close')

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Define the deep learning model
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, activation='relu'))
model.add(Dense(units=25))
model.add(Dense(units=1))

# 5. Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train the model
model.fit(X_train, y_train, epochs=100, batch_size=100)

# 7. Evaluate the model
loss = model.evaluate(X_test, y_test)

# 8.Make predictions
predictions = model.predict(X_test)

# 9. Calculate R-squared
from sklearn.metrics import r2_score
r_squared = r2_score(y_test, predictions)

# Reshape predictions to match the shape of y_test
predictions = predictions.reshape(-1, 1)

# Now both y_test and predictions have the same shape
print("Shape of y_test:", y_test.shape)
print("Shape of predictions:", predictions.shape)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

#10. Calculate R-squared
r_squared = r2_score(y_test, predictions)
print("R-squared:", r_squared)

## Data Visualization
# Plotting real vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual', color='blue')
plt.plot(predictions, label='Predicted', color='orange')
plt.title('Actual vs Predicted Values')
plt.xlabel('Data Point')
plt.ylabel('Scaled Value')  # Remember, you're working with scaled values
plt.legend()
plt.show()
closing_df = data
# Make a new tech returns DataFrame
tech_rets1 = closing_df.pct_change()
tech_rets = tech_rets1.dropna()



plt.figure(figsize=(25, 15))
plt.subplot(2, 2, 1)
sns.heatmap(tech_rets.corr(), annot=False, cmap='summer')
plt.title('Correlation of stock return')
plt.show()

closing_df.plot()
plt.show()


    
