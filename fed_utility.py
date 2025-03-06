import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class FederatedBaseOptimizer:
    def __init__(self, X, y, num_clients=3, num_rounds=5):
        self.X = X
        self.y = y
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.best_metrics = None
        
    def split_data(self):
        """Split data among clients"""
        client_data = []
        X_splits = np.array_split(self.X, self.num_clients)
        y_splits = np.array_split(self.y, self.num_clients)
        
        for X_client, y_client in zip(X_splits, y_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X_client, y_client, test_size=0.2, 
                random_state=42, stratify=y_client
            )
            client_data.append({
                'train': (X_train, y_train),
                'test': (X_test, y_test)
            })
        return client_data
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }