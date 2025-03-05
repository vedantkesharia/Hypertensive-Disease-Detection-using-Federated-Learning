import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import flwr as fl
from typing import List, Tuple, Dict
from LSTM_model import LSTMModel

# Load and preprocess data
def load_data():
    df = pd.read_csv("pca_final.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
    return train_loader, test_loader, X.shape[2], len(np.unique(y))

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for inputs, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        y_pred, y_true = [], []
        
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                outputs = self.model(inputs)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.tolist())
                y_true.extend(labels.tolist())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")

        return float(loss), len(self.test_loader.dataset), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

if __name__ == "__main__":
    train_loader, test_loader, input_size, num_classes = load_data()
    model = LSTMModel(
        input_size=input_size,
        hidden_size=128,
        num_layers=1,
        dropout=0.0,
        num_classes=num_classes
    )
    
    client = FlowerClient(model, train_loader, test_loader)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)