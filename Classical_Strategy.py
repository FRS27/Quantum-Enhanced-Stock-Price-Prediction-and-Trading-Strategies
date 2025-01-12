import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the combined data
data = pd.read_csv('combined_financial_data.csv', index_col=0, parse_dates=True)

def classical_strategy(data, stock_column, short_window=50, long_window=200):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0

    # Create short and long moving averages
    signals['Short_MA'] = data[stock_column].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['Long_MA'] = data[stock_column].rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals['Signal'][short_window:] = np.where(signals['Short_MA'][short_window:] 
                                                > signals['Long_MA'][short_window:], 1.0, 0.0)

    # Generate trading orders
    signals['Position'] = signals['Signal'].diff()

    return signals

# Apply the strategy
stock_column = 'Stock_AAPL'
classical_signals = classical_strategy(data, stock_column)

# Plot the trading signals
plt.figure(figsize=(14,7))
plt.plot(data.index, data[stock_column], label='Stock Price')
plt.plot(classical_signals.index, classical_signals['Short_MA'], label='Short MA')
plt.plot(classical_signals.index, classical_signals['Long_MA'], label='Long MA')
plt.scatter(classical_signals.loc[classical_signals['Position'] == 1.0].index, 
            data[stock_column][classical_signals['Position'] == 1.0],
            label='Buy', marker='^', color='g')
plt.scatter(classical_signals.loc[classical_signals['Position'] == -1.0].index, 
            data[stock_column][classical_signals['Position'] == -1.0],
            label='Sell', marker='v', color='r')
plt.title('Classical Trading Strategy Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend(loc='best')
plt.show()

print(classical_signals.head())




