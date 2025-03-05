import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(file_path="pca_final.csv"):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    return X, y

def create_dataloaders(X, y, num_clients=3, batch_size=32):
    client_data = []
    X_splits = np.array_split(X, num_clients)
    y_splits = np.array_split(y, num_clients)
    
    for X_client, y_client in zip(X_splits, y_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client, test_size=0.2, random_state=42
        )
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        # Create dataloaders
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), 
            batch_size=batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test), 
            batch_size=batch_size, 
            shuffle=False
        )
        
        client_data.append({
            'train': train_loader,
            'test': test_loader
        })
    
    return client_data

def calculate_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }