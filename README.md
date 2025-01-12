**﻿# Quantum-Enhanced-Stock-Price-Prediction-and-Trading-Strategies**


This repository contains a comprehensive project that leverages quantum-inspired techniques to enhance stock price prediction and trading strategies. By integrating classical machine learning, quantum-inspired features, and financial analysis, this project demonstrates innovative approaches to solving time-series forecasting problems.

**🚀 Features**

**1. Stock Price Prediction**

The stock price prediction pipeline integrates classical machine learning and quantum-inspired techniques for time-series forecasting.

**Quantum-Enhanced Bidirectional LSTM**

**Architecture:** A Bidirectional Long Short-Term Memory (LSTM) neural network augmented with quantum-inspired features.

**Quantum Circuit:** Utilizes PennyLane to implement a single-qubit quantum circuit with the following components:

**Angle Embedding:** Encodes classical data as quantum states using rotation gates.

**Basic Entangling Layers:** Enhances feature interaction through parameterized quantum gates.

**Performance Metrics (evaluated using scikit-learn):**

**Root Mean Squared Error (RMSE):** 9.8041

**Coefficient of Determination (R2):** 0.8053

**Mean Absolute Error (MAE):** 7.4524

**Preprocessing:**

MinMaxScaler from scikit-learn is used for normalizing stock price data to improve model performance.
Principal Component Analysis (PCA) from scikit-learn is applied to reduce dimensionality and capture the most significant features.

**Results:**

Accurately predicts the stock prices for Apple (AAPL) using a quantum-enhanced pipeline.
Provides improved forecasting accuracy compared to purely classical models.

**Key Technologies:**

**TensorFlow:** For building and training the Bidirectional LSTM.

**PennyLane:** For quantum circuit integration.

**scikit-learn:** For feature preprocessing (e.g., normalization, PCA) and evaluation metrics.

**NumPy and Pandas:** For preprocessing and numerical computations.

**Matplotlib:** For visualization of prediction results.

**2. Trading Strategies**

The project includes two trading strategies for generating buy/sell signals based on stock price data: Classical and Quantum-Inspired.

**Classical Trading Strategy**

**Method:**
Implements a traditional moving average crossover strategy.

**Uses two moving averages:**
Short-Term Moving Average (e.g., 50-day): Captures recent price trends.
Long-Term Moving Average (e.g., 200-day): Represents overall market trends.
Generates buy signals when the short-term average crosses above the long-term average.
Generates sell signals when the short-term average crosses below the long-term average.

**Output:**
Buy/Sell signals visualized alongside stock price and moving averages.

**Advantages:**
Simplicity and interpretability.
Proven effectiveness in identifying trends.

**Quantum-Inspired Trading Strategy**

**Method:**
Enhances the classical strategy by adding quantum-inspired noise to the moving averages:
Quantum Noise: Introduced via random noise generation using NumPy.
Adjusts moving averages dynamically to improve sensitivity to price changes.
Generates buy/sell signals based on the adjusted "Quantum Prediction."

**Output:**
Visualizations showing quantum-enhanced predictions, stock price, and trading signals.

**Advantages:**
Captures subtle patterns in stock price movements that may be missed by classical strategies.
Improves decision-making by incorporating probabilistic components.

**Comparison:**
**Evaluation Metrics (using scikit-learn):**
**RMSE:** Evaluates the accuracy of trading signals.
**R2:** Measures the variance explained by the model.
**Cumulative Returns:** Can be computed to assess financial performance.

Both strategies generate actionable buy/sell signals.

_The quantum-inspired strategy demonstrates enhanced adaptability and precision in dynamic markets.
By integrating scikit-learn for preprocessing, evaluation, and dimensionality reduction, this project highlights the synergy between classical machine learning and quantum-inspired methodologies in financial modeling._

**🧑‍💻 Contributing**

Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.
