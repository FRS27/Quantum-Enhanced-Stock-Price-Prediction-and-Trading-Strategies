import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the saved model
saved_model = keras.models.load_model('quantum_enhanced_lstm_model_g.h5')
print("Loaded saved model.")

# Load and preprocess the test data
test_df = pd.read_csv('Book2.csv', parse_dates=['Date'], index_col='Date')
test_stock_columns = ['Stock_Apple']

# Apply the same MinMaxScaler that was used for training data
scaler = MinMaxScaler()
test_data_normalized = scaler.fit_transform(test_df[test_stock_columns])

# Apply PCA 
pca = PCA(n_components=1)
test_data_reduced = pca.fit_transform(test_data_normalized)

# Prepare the test data in the same way (sliding window)
look_back = 30
X_test = []
for i in range(look_back, len(test_data_reduced)):
    X_test.append(test_data_reduced[i-look_back:i])

X_test = np.array(X_test)

# Make predictions using the loaded model
test_predictions = saved_model.predict(X_test)

# Inverse transform the predictions to get them back to original scale
test_predictions_scaled = scaler.inverse_transform(test_predictions)

# Calculate and print performance metrics
rmse_scores = []
r2_scores = []
mae_scores = []

print("Performance metrics:")

for i, stock in enumerate(test_stock_columns):
    # Extract actual test values and predictions
    actual_values = test_df[stock].values[-len(test_predictions_scaled):]
    predicted_values = test_predictions_scaled[:, i]
    
    # Calculate RMSE, R², and MAE
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    r2 = r2_score(actual_values, predicted_values)
    mae = mean_absolute_error(actual_values, predicted_values)
    
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    mae_scores.append(mae)
    
    print(f'{stock}: RMSE = {rmse:.4f}, R² = {r2:.4f}, MAE = {mae:.4f}')

# Plot the original vs predicted stock prices
plt.figure(figsize=(14, 8))
for i, stock in enumerate(test_stock_columns):
    plt.subplot(2, 2, i+1)
    
    # Plot original test data
    plt.plot(test_df.index[-len(test_predictions_scaled):], test_df[stock].values[-len(test_predictions_scaled):], label='Original Data')
    
    # Plot predictions
    plt.plot(test_df.index[-len(test_predictions_scaled):], test_predictions_scaled[:, i], 
             label='Quantum-Enhanced Bidirectional LSTM Forecast', alpha=0.7)
    
    plt.title(stock)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Print average performance metrics
average_rmse = np.mean(rmse_scores)
average_r2 = np.mean(r2_scores)
average_mae = np.mean(mae_scores)

print(f'\nAverage RMSE: {average_rmse:.4f}')
print(f'Average R²: {average_r2:.4f}')
print(f'Average MAE: {average_mae:.4f}')
