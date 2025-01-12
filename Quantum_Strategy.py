import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the combined data
data = pd.read_csv('combined_financial_data.csv', index_col=0, parse_dates=True)

def quantum_inspired_strategy(data, stock_column, window=50, quantum_factor=0.01):
    signals = pd.DataFrame(index=data.index)
    signals['Signal'] = 0.0


    # Create a "quantum-enhanced" prediction
    np.random.seed(42)  # for reproducibility
    quantum_noise = np.random.normal(0, quantum_factor, len(data))
    signals['Quantum_Prediction'] = data[stock_column].rolling(window=window, min_periods=1, center=False).mean() + \
                                    data[stock_column] * quantum_noise

    # Generate trading signals
    signals['Signal'] = np.where(signals['Quantum_Prediction'] > data[stock_column], 1.0, 0.0)

    # Generate trading orders
    signals['Position'] = signals['Signal'].diff()

    return signals

# Apply the quantum-inspired strategy to the same stock
stock_column = 'Stock_AAPL'
quantum_signals = quantum_inspired_strategy(data, stock_column)

# Plot the quantum-inspired trading signals
plt.figure(figsize=(14,7))
plt.plot(data.index, data[stock_column], label='Stock Price')
plt.plot(quantum_signals.index, quantum_signals['Quantum_Prediction'], label='Quantum Prediction', alpha=0.7)
plt.scatter(quantum_signals.loc[quantum_signals['Position'] == 1.0].index, 
            data[stock_column][quantum_signals['Position'] == 1.0],
            label='Buy', marker='^', color='g')
plt.scatter(quantum_signals.loc[quantum_signals['Position'] == -1.0].index, 
            data[stock_column][quantum_signals['Position'] == -1.0],
            label='Sell', marker='v', color='r')
plt.title('Quantum-Inspired Trading Strategy Signals')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend(loc='best')
plt.show()

print(quantum_signals.head())