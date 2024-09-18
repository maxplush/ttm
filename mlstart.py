import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Fetch and process data
def get_market_data(ticker='VOO', start='2010-01-01', end='2023-01-01'):
    data = yf.download(ticker, start=start, end=end)
    data['50_SMA'] = data['Close'].rolling(window=50).mean()
    data['200_SMA'] = data['Close'].rolling(window=200).mean()
    data['Volatility'] = data['Close'].rolling(window=10).std()  # Example for volatility
    data['Volume_Trend'] = data['Volume'].rolling(window=5).mean()
    data['Price_vs_200_SMA'] = (data['Close'] - data['200_SMA']) / data['Close']
    return data.dropna()

# Step 2: Generate deposit dates (last trading day of the month)
def generate_regular_deposit_dates(data, amount=1000):
    deposit_dates = data.resample('M').last().index
    deposits = pd.DataFrame(index=deposit_dates, columns=['Deposit'], data=amount)
    return deposits

# Step 3: Create target labels (1 = Buy, 0 = Hold)
def prepare_features_and_labels(data):
    features = data[['50_SMA', '200_SMA', 'Volatility', 'Volume_Trend', 'Price_vs_200_SMA']].copy()
    labels = (data['Close'].shift(-1) > data['Close']).astype(int)  # If tomorrow's price is higher, it's a buy signal
    return features, labels

# Step 4: Train a simple classification model
def train_ml_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print(f"Model accuracy: {model.score(X_test, y_test)}")
    return model

# Step 5: Simulate ML-based strategy
def simulate_ml_strategy(data, model, initial_cash=10000):
    portfolio = pd.DataFrame(index=data.index, columns=['Portfolio Value'])
    cash = initial_cash
    units_held = 0

    for date in data.index:
        feature_vector = data.loc[date, ['50_SMA', '200_SMA', 'Volatility', 'Volume_Trend', 'Price_vs_200_SMA']].values.reshape(1, -1)
        buy_signal = model.predict(feature_vector)[0]
        
        # If buy signal is 1 and there's cash available, buy
        if buy_signal == 1 and cash > 0:
            units_held += cash / data.loc[date, 'Close']
            cash = 0  # Invest all cash

        portfolio.loc[date, 'Portfolio Value'] = units_held * data.loc[date, 'Close'] if units_held > 0 else cash

    return portfolio

# Step 6: Simulate regular deposits for comparison
def simulate_regular_deposits(data, deposits):
    portfolio = pd.DataFrame(index=data.index, columns=['Portfolio Value'])
    cash = 0
    units_held = 0

    for date in data.index:
        if date in deposits.index:
            cash += deposits.loc[date, 'Deposit']
            units_held += cash / data.loc[date, 'Close']
            cash = 0  # Reset cash after buying shares

        portfolio.loc[date, 'Portfolio Value'] = units_held * data.loc[date, 'Close']

    return portfolio

# Step 7: Run simulation
def run_simulation():
    data = get_market_data()
    deposits = generate_regular_deposit_dates(data)

    # Prepare features and labels for ML model
    features, labels = prepare_features_and_labels(data)

    # Train ML model
    model = train_ml_model(features, labels)

    # Simulate both strategies
    ml_portfolio = simulate_ml_strategy(data, model)
    regular_portfolio = simulate_regular_deposits(data, deposits)

    # Compare results
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, ml_portfolio['Portfolio Value'], label="ML Strategy")
    plt.plot(data.index, regular_portfolio['Portfolio Value'], label="Regular Deposits")
    plt.legend()
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.show()

# Execute the simulation
run_simulation()
