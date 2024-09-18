import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fetch VOO data from yfinance (daily prices)
def get_voo_data(start_date: str, end_date: str):
    voo = yf.Ticker('SPY')
    data = voo.history(start=start_date, end=end_date, interval='1d')
    data['SMA_204'] = data['Close'].rolling(window=204).mean()
    return data

# Generate a list of deposit dates (first market day of each month)
def generate_deposit_dates(data, monthly_amount=1000):
    first_market_days = data.resample('B').first().resample('ME').first().index
    deposits = pd.DataFrame(index=first_market_days, columns=['Deposit'], data=monthly_amount)
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

# Simulate the optimized recurring deposit strategy
def simulate_optimized_deposits(data, deposits):
    portfolio = pd.DataFrame(index=data.index, columns=['Portfolio Value'])
    cash_reserve = 0
    units_held = 0
    prev_invest_date = None

    for date in data.index:
        if date in deposits.index:
            if prev_invest_date is not None:
                prev_price = data.loc[prev_invest_date, 'Close']
                sma_204 = data.loc[date, 'SMA_204']
                if data.loc[date, 'Close'] < sma_204:
                    cash_reserve += deposits.loc[date, 'Deposit']
                else:
                    price_gap = (data.loc[date, 'Close'] - sma_204) / data.loc[date, 'Close']
                    if price_gap > 0 or price_gap < -0.15:
                        units_held += cash_reserve / data.loc[date, 'Close']
                        cash_reserve = 0
                    units_held += deposits.loc[date, 'Deposit'] / data.loc[date, 'Close']
            else:
                units_held += deposits.loc[date, 'Deposit'] / data.loc[date, 'Close']
            prev_invest_date = date

        portfolio.loc[date, 'Portfolio Value'] = units_held * data.loc[date, 'Close'] + cash_reserve

    return portfolio

# Compare the two strategies and output the difference
def compare_strategies(optimized_portfolio, regular_portfolio):
    final_optimized_value = optimized_portfolio['Portfolio Value'].dropna().iloc[-1]
    final_regular_value = regular_portfolio['Portfolio Value'].dropna().iloc[-1]
    difference = final_optimized_value - final_regular_value
    pct_difference = (difference / final_regular_value) * 100

    print(f"Final Portfolio Value (Optimized): ${final_optimized_value:,.2f}")
    print(f"Final Portfolio Value (Regular): ${final_regular_value:,.2f}")
    print(f"Difference in Final Portfolio Value: ${difference:,.2f}")
    print(f"Percentage Difference: {pct_difference:.2f}%")

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
    #plt.show()

    # Output comparison metrics
    compare_strategies(optimized_portfolio, regular_portfolio)

# Run simulation (example dates)
run_simulation('1993-01-01', '2024-01-01')
