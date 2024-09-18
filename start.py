import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Fetch VOO data from yfinance (daily prices)
def get_voo_data(start_date: str, end_date: str):
    voo = yf.Ticker('VOO')
    data = voo.history(start=start_date, end=end_date, interval='1d')
    data['SMA_50'] = data['Close'].rolling(window=50).mean()  # 50-day SMA
    data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day SMA
    return data

# Generate a list of deposit dates (last trading day of each month)
def generate_deposit_dates(data, monthly_amount=1000):
    deposit_dates = data.resample('M').last().index
    deposits = pd.DataFrame(index=deposit_dates, columns=['Deposit'], data=monthly_amount)
    return deposits

# Calculate portfolio value based on regular deposits
def simulate_regular_deposits(data, deposits):
    portfolio = pd.DataFrame(index=data.index, columns=['Portfolio Value'])
    cash = 0
    units_held = 0

    for date in data.index:
        if date in deposits.index:
            cash += deposits.loc[date, 'Deposit']
            units_held += cash / data.loc[date, 'Close']
            cash = 0  # Reset cash to 0 after buying shares

        portfolio.loc[date, 'Portfolio Value'] = units_held * data.loc[date, 'Close']

    return portfolio

# Alternative buy strategy based on SMA crossover (example)
def simulate_sma_strategy(data, initial_cash=10000):
    portfolio = pd.DataFrame(index=data.index, columns=['Portfolio Value'])
    cash = initial_cash
    units_held = 0
    for date in data.index:
        # Buy signal: when 50-day SMA crosses above 200-day SMA
        if data.loc[date, 'SMA_50'] > data.loc[date, 'SMA_200'] and cash > 0:
            units_held += cash / data.loc[date, 'Close']
            cash = 0  # All in
        portfolio.loc[date, 'Portfolio Value'] = units_held * data.loc[date, 'Close'] if units_held > 0 else cash
    return portfolio


# Main function to run the simulation
def run_simulation(start_date, end_date):
    data = get_voo_data(start_date, end_date)
    deposits = generate_deposit_dates(data)
    
    # Simulate regular deposits
    regular_portfolio = simulate_regular_deposits(data, deposits)

    # Simulate SMA strategy
    sma_portfolio = simulate_sma_strategy(data)

    # Plot results
    plt.figure(figsize=(10,6))
    plt.plot(data.index, regular_portfolio['Portfolio Value'], label="Regular Deposits")
    plt.plot(data.index, sma_portfolio['Portfolio Value'], label="SMA Strategy")
    plt.legend()
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.show()

# Run simulation (example dates)
run_simulation('2015-01-01', '2024-01-01')
