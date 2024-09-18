# Optimized Recurring Deposits Strategy

This project presents an optimized approach to recurring deposits, aiming to enhance returns by strategically timing investments based on market conditions. The focus is on a rules-based strategy that utilizes the Simple Moving Average (SMA) as a key indicator for decision-making.

## Strategy Overview

The rules-based strategy involves the following steps:

1. **Monthly Investment**: Allocate $1,000 for investment on the first market day of each month.

2. **SMA Comparison**:
   - **If Current Price ≥ 200-day SMA**: Proceed with the investment.
   - **If Current Price < 200-day SMA**: Postpone the investment until the price surpasses the 200-day SMA or the gap between the price and the SMA exceeds 15%.

This method ensures that investments are made when the market is above its long-term average, potentially capturing upward momentum.

## Performance Insights

Backtesting this strategy over a 30-year period revealed a 1.4% outperformance compared to a standard recurring deposit approach. This indicates that the rules-based strategy can lead to higher returns by optimizing the timing of investments.


## Getting Started

- Before running the script, make sure you have the packages listed in the requirements.txt file.
