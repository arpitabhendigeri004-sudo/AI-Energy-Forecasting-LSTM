import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load dataset
def load_data():
    data = pd.read_csv('data/energy.csv')
    
    # Rename columns correctly
    data.columns = ['Datetime', 'Energy']
    
    # Convert to datetime
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    
    # Set index
    data.set_index('Datetime', inplace=True)

    data = data.iloc[:1000]
    
    # Resample hourly
    data = data.resample('h').mean()
    
    # Fill missing values
    data = data.ffill()
    
    return data

# Scale data (IMPORTANT for LSTM)
def scale_data(data):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[['Energy']])
    return scaled, scaler

# Create sequences
def create_sequences(data, seq_length=24):
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
        
    X = np.array(X)
    y = np.array(y)
    print("Shape before reshape:", X.shape)
    # reshape for LSTM [samples, time_steps, features]
    if len(X.shape) == 2:
       X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y