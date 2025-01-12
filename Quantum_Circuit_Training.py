import os
import pennylane as qml
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA


# Load and normalize the stock data
df = pd.read_csv('Book1.csv', parse_dates=['Date'], index_col='Date')
stock_columns = ['Apple']
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(df[stock_columns])

# Apply PCA to reduce dimensionality
pca = PCA(n_components=1)
data_reduced = pca.fit_transform(data_normalized)

# Prepare the data for LSTM
look_back = 30
X = []
Y = []
for i in range(look_back, len(data_reduced)):
    X.append(data_reduced[i - look_back:i])
    Y.append(data_normalized[i])  # Predicting all stock prices

X = np.array(X)
Y = np.array(Y)

# Define a quantum circuit
n_qubits = 1
n_layers = 1
dev = qml.device('default.qubit', wires=n_qubits)

@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# weight initialization
weights = np.random.randn(n_layers, n_qubits)

# Process data through the quantum circuit
def process_quantum_data(data):
    quantum_data = []
    for x in data:
        batch_data = np.array_split(x.flatten(), look_back // n_qubits)
        batch_output = [quantum_circuit(batch, weights) for batch in batch_data]
        quantum_data.append(np.concatenate(batch_output))
    return np.array(quantum_data)

quantum_data = process_quantum_data(X)
quantum_data_mean = quantum_data.mean(axis=1)

# Define classical Bidirectional LSTM model with quantum-enhanced input
def create_bidirectional_lstm_model(input_shape):
    combined_regularizer = tf.keras.regularizers.l1_l2(l1=0.002, l2=0.01)

    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Bidirectional(keras.layers.LSTM(32, return_sequences=False, 
                                                     kernel_regularizer=combined_regularizer))(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(20, activation='relu', kernel_regularizer=combined_regularizer)(x)
    outputs = keras.layers.Dense(len(stock_columns))(x)  # Predicting all stocks
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Evaluate performance metrics
def evaluate_performance(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# Perform k-fold cross-validation and calculate metrics
kf = KFold(n_splits=5, shuffle=True)
mse_scores, rmse_scores, mae_scores, r2_scores = [], [], [], []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    lstm_model = create_bidirectional_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    lstm_model.fit(X_train, Y_train, validation_split=0.2, batch_size=32, epochs=50, 
                   callbacks=[early_stopping], verbose=0)  # Use verbose=0 for clean logs
    
    Y_pred = lstm_model.predict(X_test)
    
    mse, rmse, mae, r2 = evaluate_performance(Y_test, Y_pred)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r2_scores.append(r2)

# Calculate average metrics across all folds
average_mse = np.mean(mse_scores)
average_rmse = np.mean(rmse_scores)
average_mae = np.mean(mae_scores)
average_r2 = np.mean(r2_scores)

print(f'Average MSE: {average_mse:.4f}')
print(f'Average RMSE: {average_rmse:.4f}')
print(f'Average MAE: {average_mae:.4f}')
print(f'Average R²: {average_r2:.4f}')

# Save the model
lstm_model.save('quantum_enhanced_lstm_model_g.h5')
print("Model saved as quantum_enhanced_lstm_model_g.h5")

# Make final predictions using the full dataset
lstm_model = create_bidirectional_lstm_model((X.shape[1], X.shape[2]))
lstm_model.fit(X, Y, batch_size=32, epochs=50)
predictions = lstm_model.predict(X)
predictions_scaled = scaler.inverse_transform(predictions)

# Plot original and forecasted data
plt.figure(figsize=(14, 8))
for i, stock in enumerate(stock_columns):
    plt.subplot(2, 2, i + 1)
    
    # Plot original data for the entire dataset
    plt.plot(df.index, df[stock], label='Original Data')
    
    # Plot the portion of predictions 
    plt.plot(df.index[-len(predictions_scaled):], predictions_scaled[:, i], 
             label='Quantum-Enhanced Bidirectional LSTM Forecast', alpha=0.7)
    
    plt.title(stock)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
