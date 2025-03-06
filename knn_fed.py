import numpy as np
import optuna
import joblib
import pandas as pd  # Add pandas import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler  # Fix StandardScaler import
from sklearn.base import clone  # Add clone import
from fed_utility import FederatedBaseOptimizer
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

class FederatedKNNOptimizer(FederatedBaseOptimizer):
    def __init__(self, X, y, num_clients=3, num_rounds=5):
        super().__init__(X, y, num_clients, num_rounds)
        self.best_model = None
        self.scaler = StandardScaler()
        
    def split_data(self):
        """Improved data splitting with stratification"""
        client_data = []
        # Use StratifiedKFold to ensure balanced class distribution
        skf = StratifiedKFold(n_splits=self.num_clients, shuffle=True, random_state=42)
        
        for train_idx, test_idx in skf.split(self.X, self.y):
            X_client, y_client = self.X[train_idx], self.y[train_idx]
            X_train, X_test, y_train, y_test = train_test_split(
                X_client, y_client, 
                test_size=0.2, 
                random_state=42,
                stratify=y_client
            )
            client_data.append({
                'train': (X_train, y_train),
                'test': (X_test, y_test)
            })
        return client_data

    def federated_averaging(self, models, client_data):
  
        X_combined = []
        y_combined = []
        weights = []
    
        for model, data in zip(models, client_data):
            X_train, y_train = data['train']
            X_test, y_test = data['test']
        
        # Calculate weight based on validation performance
            score = model.score(X_test, y_test)
            weights.append(score)
            X_combined.append(X_train)
            y_combined.append(y_train)
    
    # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
    
    # Weighted combination of data
        X_combined = np.vstack(X_combined)
        y_combined = np.concatenate(y_combined)
    
    # Get parameters from best performing model
        best_idx = np.argmax(weights)
        best_params = models[best_idx].get_params()
    
    # Remove 'weights' from best_params if it exists
        best_params.pop('weights', None)
    
    # Create federated model with best parameters
        fed_model = KNeighborsClassifier(
        weights='distance',  # Set weights parameter explicitly
        **best_params  # Include remaining parameters
        )
        fed_model.fit(X_combined, y_combined)
    
        return fed_model

    def objective(self, trial):
        params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 5, 20),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
        'algorithm': 'auto',
        'p': trial.suggest_int('p', 1, 3),  # Minkowski parameter
        'weights': 'distance'  # Set weights parameter directly
       }
        
        client_data = self.split_data()
        models = []
        
        # Initialize models
        for _ in range(self.num_clients):
            model = KNeighborsClassifier(**params)
            models.append(model)
        
        best_f1 = 0.0
        best_model = None
        patience = 3
        no_improve_count = 0
        prev_f1 = 0
        
        for round_num in range(self.num_rounds):
            print(f"\nFederated Round {round_num + 1}/{self.num_rounds}")
            
            # Train and evaluate client models
            client_metrics = []
            for client_idx, (model, data) in enumerate(zip(models, client_data)):
                X_train, y_train = data['train']
                X_test, y_test = data['test']
                
                model.fit(X_train, y_train)
                metrics = self.evaluate_model(model, X_test, y_test)
                client_metrics.append(metrics)
                print(f"Client {client_idx + 1} metrics:", metrics)
            
            # Federation
            fed_model = self.federated_averaging(models, client_data)
            
            # Evaluate federated model
            all_metrics = []
            for data in client_data:
                X_test, y_test = data['test']
                metrics = self.evaluate_model(fed_model, X_test, y_test)
                all_metrics.append(metrics)
            
            avg_metrics = {
                metric: np.mean([m[metric] for m in all_metrics])
                for metric in all_metrics[0].keys()
            }
            
            print(f"Global metrics for round {round_num + 1}:", avg_metrics)
            
            # Early stopping check
            current_f1 = avg_metrics['f1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                self.best_metrics = avg_metrics
                best_model = fed_model
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= patience:
                print("Early stopping triggered")
                break
            
            # Update client models
            models = [clone(fed_model) for _ in range(self.num_clients)]
        
        return best_f1

def main():
    print("Loading and preprocessing data...")
    df = pd.read_csv("pca_final.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create study with improved settings
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="federated_knn_optimization"
    )
    
    # Initialize optimizer with scaled data
    fed_optimizer = FederatedKNNOptimizer(X, y, num_rounds=5)
    
    # Run optimization
    n_trials = 20
    print(f"\nStarting hyperparameter optimization with {n_trials} trials...")
    study.optimize(fed_optimizer.objective, n_trials=n_trials)
    
    # Print results
    print("\n=== Optimization Results ===")
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    print("\nBest model metrics:")
    for metric, value in fed_optimizer.best_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study.trials_dataframe().to_csv(f'results/knn_optimization_results_{timestamp}.csv')

if __name__ == "__main__":
    main()