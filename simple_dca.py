import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Fetch VOO data from yfinance (daily prices)
def get_voo_data(start_date: str, end_date: str):
    voo = yf.Ticker('VOO')
    data = voo.history(start=start_date, end=end_date, interval='1d')
    data['SMA_200'] = data['Close'].rolling(window=200).mean()  # 200-day SMA
    return data

# Generate a list of deposit dates (first market day of each month)
def generate_deposit_dates(data, monthly_amount=1000):
    first_market_days = data.resample('M').first().index
    deposits = pd.DataFrame(index=first_market_days, columns=['Deposit'], data=monthly_amount)
    return deposits

# Simulate the optimized recurring deposit strategy
def simulate_optimized_deposits(data, deposits):
    portfolio = pd.DataFrame(index=data.index, columns=['Portfolio Value'])
    cash_reserve = 0
    units_held = 0

    for date in data.index:
        if date in deposits.index:
            # Check if the price is below the 200-day SMA
            if data.loc[date, 'Close'] < data.loc[date, 'SMA_200']:
                cash_reserve += deposits.loc[date, 'Deposit']
            else:
                # Deploy all cash reserves if the price is above the 200-day SMA or the price gap condition is met
                if cash_reserve > 0:
                    price_gap = (data.loc[date, 'Close'] - data.loc[date, 'SMA_200']) / data.loc[date, 'Close']
                    if price_gap > 0 or price_gap < -0.15:  # Price above SMA or gap > 15%
                        units_held += cash_reserve / data.loc[date, 'Close']
                        cash_reserve = 0
                # Regular deposit for the month
                units_held += deposits.loc[date, 'Deposit'] / data.loc[date, 'Close']

        portfolio.loc[date, 'Portfolio Value'] = units_held * data.loc[date, 'Close'] + cash_reserve

    return portfolio

# Simulate the regular recurring deposit strategy
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

# Function to calculate and print the comparison metrics
def compare_strategies(optimized_portfolio, regular_portfolio, data):
    # Final values
    final_optimized_value = optimized_portfolio['Portfolio Value'].iloc[-1]
    final_regular_value = regular_portfolio['Portfolio Value'].iloc[-1]
    
    # Difference in final values
    difference = final_optimized_value - final_regular_value
    percentage_diff = (difference / final_regular_value) * 100

    print(f"Final Portfolio Value (Optimized): ${final_optimized_value:,.2f}")
    print(f"Final Portfolio Value (Regular): ${final_regular_value:,.2f}")
    print(f"Difference in Final Portfolio Value: ${difference:,.2f}")
    print(f"Percentage Difference: {percentage_diff:.2f}%")

    # Plot percentage difference over time
    percentage_diff_over_time = (
        (optimized_portfolio['Portfolio Value'] - regular_portfolio['Portfolio Value']) 
        / regular_portfolio['Portfolio Value']
    ) * 100

    plt.figure(figsize=(10,6))
    plt.plot(data.index, percentage_diff_over_time, label="Percentage Difference Over Time")
    plt.axhline(0, color='gray', linestyle='--')  # Zero difference line
    plt.title("Percentage Difference Between Optimized and Regular Deposits Over Time")
    plt.xlabel("Date")
    plt.ylabel("Percentage Difference (%)")
    plt.legend()
    plt.show()

# Main function to run the simulation
def run_simulation(start_date, end_date):
    data = get_voo_data(start_date, end_date)
    deposits = generate_deposit_dates(data)
    
    # Simulate optimized deposits
    optimized_portfolio = simulate_optimized_deposits(data, deposits)

    # Simulate regular deposits for comparison
    regular_portfolio = simulate_regular_deposits(data, deposits)

    # Plot results
    plt.figure(figsize=(10,6))
    plt.plot(data.index, regular_portfolio['Portfolio Value'], label="Regular Deposits")
    plt.plot(data.index, optimized_portfolio['Portfolio Value'], label="Optimized Recurring Deposits")
    plt.legend()
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.show()

    # Output comparison metrics
    compare_strategies(optimized_portfolio, regular_portfolio, data)

# Run simulation (example dates)
run_simulation('2015-01-01', '2024-01-01')
