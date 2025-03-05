import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from torch.utils.data import DataLoader, TensorDataset
import os
from datetime import datetime

from MLP_model import MLPClassifier

class FederatedMLPOptimizer:
    def __init__(self, X, y, num_clients=3, num_rounds=5):  # Increased rounds
        self.X = X
        self.y = y
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.num_classes = len(np.unique(y))
        self.input_size = X.shape[1]
        self.best_metrics = None
        self.best_trial_metrics = None
        
    def create_dataloaders(self, batch_size):
        client_data = []
        # Stratified split for better distribution
        X_splits = np.array_split(self.X, self.num_clients)
        y_splits = np.array_split(self.y, self.num_clients)
        
        for X_client, y_client in zip(X_splits, y_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                X_client, y_client, test_size=0.2, 
                random_state=42, stratify=y_client  # Added stratification
            )
            
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            
            train_loader = DataLoader(
                TensorDataset(X_train, y_train), 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=0  # Avoid multiprocessing issues
            )
            test_loader = DataLoader(
                TensorDataset(X_test, y_test), 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=0
            )
            
            client_data.append({
                'train': train_loader,
                'test': test_loader
            })
        
        return client_data

    def train_client(self, model, train_loader, optimizer, criterion, num_epochs=5):
        best_loss = float('inf')
        patience = 3
        patience_counter = 0
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = total_loss / len(train_loader)
            epoch_acc = 100 * correct / total
            
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        return epoch_loss

    def evaluate_model(self, model, test_loader):
        model.eval()
        y_pred = []
        y_true = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_pred.extend(predicted.numpy())
                y_true.extend(labels.numpy())
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted')
        }

    def federated_averaging(self, models):
        state_dict = models[0].state_dict()
        for key in state_dict.keys():
            stacked_weights = torch.stack([model.state_dict()[key] for model in models])
            if stacked_weights.dtype == torch.long:
            # For integer tensors, convert to float for averaging then back to long
                state_dict[key] = stacked_weights.float().mean(0).long()
            else:
                state_dict[key] = stacked_weights.mean(0)
        return state_dict

    def objective(self, trial):
        params = {
            'hidden_size1': trial.suggest_int('hidden_size1', 128, 1024),  # Increased range
            'hidden_size2': trial.suggest_int('hidden_size2', 64, 512),    # Increased range
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.4),  # Narrowed range
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])  # Removed small batch size
        }
        
        client_data = self.create_dataloaders(params['batch_size'])
        
        models = []
        optimizers = []
        for _ in range(self.num_clients):
            model = MLPClassifier(
                input_size=self.input_size,
                hidden_size1=params['hidden_size1'],
                hidden_size2=params['hidden_size2'],
                dropout_rate=params['dropout_rate'],
                num_classes=self.num_classes
            )
            # Added weight decay for regularization
            optimizer = optim.Adam(
                model.parameters(), 
                lr=params['learning_rate'],
                weight_decay=1e-5
            )
            models.append(model)
            optimizers.append(optimizer)
        
        criterion = nn.CrossEntropyLoss()
        best_f1 = 0.0
        
        # Pre-federation training
        print(f"\nPre-federation training for trial {trial.number}")
        for client_idx, (model, optimizer, data) in enumerate(zip(models, optimizers, client_data)):
            loss = self.train_client(model, data['train'], optimizer, criterion)
            metrics = self.evaluate_model(model, data['test'])
            print(f"Client {client_idx} initial metrics:", metrics)
        
        # Federated learning rounds
        for round_num in range(self.num_rounds):
            print(f"\nFederated Round {round_num + 1}/{self.num_rounds}")
            
            # Train each client
            for client_idx, (model, optimizer, data) in enumerate(zip(models, optimizers, client_data)):
                loss = self.train_client(model, data['train'], optimizer, criterion)
                metrics = self.evaluate_model(model, data['test'])
                trial.report(metrics['f1'], round_num * self.num_clients + client_idx)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Federated averaging
            averaged_state_dict = self.federated_averaging(models)
            
            # Update all models
            for model in models:
                model.load_state_dict(averaged_state_dict)
            
            # Evaluate federated model
            all_metrics = []
            for model, data in zip(models, client_data):
                metrics = self.evaluate_model(model, data['test'])
                all_metrics.append(metrics)
            
            avg_metrics = {
                metric: np.mean([m[metric] for m in all_metrics])
                for metric in all_metrics[0].keys()
            }
            
            print(f"Round {round_num + 1} metrics:", avg_metrics)
            
            if avg_metrics['f1'] > best_f1:
                best_f1 = avg_metrics['f1']
                self.best_trial_metrics = avg_metrics
                os.makedirs('models', exist_ok=True)
                torch.save(averaged_state_dict, 
                         f'models/best_mlp_model_trial_{trial.number}_round_{round_num + 1}.pt')
        
        return best_f1

def main():
    print("Loading and preprocessing data...")
    df = pd.read_csv("pca_final.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name="federated_mlp_optimization"
    )
    
    fed_optimizer = FederatedMLPOptimizer(X, y, num_rounds=5)  # Increased rounds
    
    n_trials = 20
    print(f"\nStarting hyperparameter optimization with {n_trials} trials...")
    study.optimize(fed_optimizer.objective, n_trials=n_trials)
    
    print("\n=== Optimization Results ===")
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    print("\nBest model metrics:")
    for metric, value in fed_optimizer.best_trial_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study.trials_dataframe().to_csv(f'results/mlp_optimization_results_{timestamp}.csv')

if __name__ == "__main__":
    main()






# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import optuna
# from torch.utils.data import DataLoader, TensorDataset
# import os
# from datetime import datetime

# from MLP_model import MLPClassifier

# class FederatedMLPOptimizer:
#     def __init__(self, X, y, num_clients=3, num_rounds=3):
#         self.X = X
#         self.y = y
#         self.num_clients = num_clients
#         self.num_rounds = num_rounds
#         self.num_classes = len(np.unique(y))
#         self.input_size = X.shape[1]
#         self.best_metrics = None
#         self.best_trial_metrics = None
        
#     def create_dataloaders(self, batch_size):
#         # Split data for clients
#         client_data = []
#         X_splits = np.array_split(self.X, self.num_clients)
#         y_splits = np.array_split(self.y, self.num_clients)
        
#         for X_client, y_client in zip(X_splits, y_splits):
#             X_train, X_test, y_train, y_test = train_test_split(
#                 X_client, y_client, test_size=0.2, random_state=42
#             )
            
#             # Convert to PyTorch tensors
#             X_train = torch.FloatTensor(X_train)
#             X_test = torch.FloatTensor(X_test)
#             y_train = torch.LongTensor(y_train)
#             y_test = torch.LongTensor(y_test)
            
#             # Create dataloaders
#             train_loader = DataLoader(
#                 TensorDataset(X_train, y_train), 
#                 batch_size=batch_size, 
#                 shuffle=True
#             )
#             test_loader = DataLoader(
#                 TensorDataset(X_test, y_test), 
#                 batch_size=batch_size, 
#                 shuffle=False
#             )
            
#             client_data.append({
#                 'train': train_loader,
#                 'test': test_loader
#             })
        
#         return client_data

#     def train_client(self, model, train_loader, optimizer, criterion):
#         model.train()
#         total_loss = 0
        
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
            
#         return total_loss / len(train_loader)

#     def evaluate_model(self, model, test_loader):
#         model.eval()
#         y_pred = []
#         y_true = []
        
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 y_pred.extend(predicted.numpy())
#                 y_true.extend(labels.numpy())
        
#         return {
#             'accuracy': accuracy_score(y_true, y_pred),
#             'precision': precision_score(y_true, y_pred, average='weighted'),
#             'recall': recall_score(y_true, y_pred, average='weighted'),
#             'f1': f1_score(y_true, y_pred, average='weighted')
#         }

#     def federated_averaging(self, models):
#         """Average the weights of all client models"""
#         state_dict = models[0].state_dict()
#         for key in state_dict.keys():
#             state_dict[key] = torch.stack([model.state_dict()[key] for model in models]).mean(0)
#         return state_dict

#     def objective(self, trial):
#         # Hyperparameters to optimize
#         params = {
#             # 'hidden_sizes': [
#             #     trial.suggest_int('hidden_size_1', 32, 512),
#             #     trial.suggest_int('hidden_size_2', 16, 256)
#             # ],
#             'hidden_size1':trial.suggest_int('hidden_size1', 32, 512),
#             'hidden_size2':trial.suggest_int('hidden_size2', 16, 256),
#             'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
#             'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
#             'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
#         }
        
#         # Create client dataloaders
#         client_data = self.create_dataloaders(params['batch_size'])
        
#         # Initialize models for each client
#         models = []
#         optimizers = []
#         for _ in range(self.num_clients):
#             model = MLPClassifier(
#                 input_size=self.input_size,
#                 hidden_size1=params['hidden_size1'],
#                 hidden_size2=params['hidden_size2'],
#                 # hidden_sizes=params['hidden_sizes'],
#                 dropout_rate=params['dropout_rate'],
#                 num_classes=self.num_classes
#             )
#             optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
#             models.append(model)
#             optimizers.append(optimizer)
        
#         criterion = nn.CrossEntropyLoss()
#         best_f1 = 0.0
        
#         # Federated learning rounds
#         for round_num in range(self.num_rounds):
#             # Train each client
#             for client_idx, (model, optimizer, data) in enumerate(zip(models, optimizers, client_data)):
#                 loss = self.train_client(model, data['train'], optimizer, criterion)
#                 metrics = self.evaluate_model(model, data['test'])
#                 trial.report(metrics['f1'], round_num * self.num_clients + client_idx)
                
#                 if trial.should_prune():
#                     raise optuna.TrialPruned()
            
#             # Federated averaging
#             averaged_state_dict = self.federated_averaging(models)
            
#             # Update all models
#             for model in models:
#                 model.load_state_dict(averaged_state_dict)
            
#             # Evaluate federated model
#             all_metrics = []
#             for model, data in zip(models, client_data):
#                 metrics = self.evaluate_model(model, data['test'])
#                 all_metrics.append(metrics)
            
#             # Average metrics
#             avg_metrics = {
#                 metric: np.mean([m[metric] for m in all_metrics])
#                 for metric in all_metrics[0].keys()
#             }
            
#             if avg_metrics['f1'] > best_f1:
#                 best_f1 = avg_metrics['f1']
#                 self.best_trial_metrics = avg_metrics
                
#                 # Save best model
#                 os.makedirs('models', exist_ok=True)
#                 torch.save(averaged_state_dict, 
#                          f'models/best_mlp_model_trial_{trial.number}_round_{round_num + 1}.pt')
        
#         return best_f1

# def main():
#     # Load and preprocess data
#     print("Loading and preprocessing data...")
#     df = pd.read_csv("pca_final.csv")
#     X = df.iloc[:, :-1].values
#     y = df.iloc[:, -1].values
    
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
    
#     # Create study
#     study = optuna.create_study(
#         direction="maximize",
#         pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
#         study_name="federated_mlp_optimization"
#     )
    
#     # Initialize optimizer
#     fed_optimizer = FederatedMLPOptimizer(X, y)
    
#     # Run optimization
#     n_trials = 20
#     print(f"\nStarting hyperparameter optimization with {n_trials} trials...")
#     study.optimize(fed_optimizer.objective, n_trials=n_trials)
    
#     # Print results
#     print("\n=== Optimization Results ===")
#     print("Best trial:")
#     trial = study.best_trial
    
#     print(f"  Value: {trial.value:.4f}")
#     print("  Params: ")
#     for key, value in trial.params.items():
#         print(f"    {key}: {value}")
    
#     print("\nBest model metrics:")
#     for metric, value in fed_optimizer.best_trial_metrics.items():
#         print(f"{metric}: {value:.4f}")
    
#     # Save study results
#     os.makedirs('results', exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     study.trials_dataframe().to_csv(f'results/mlp_optimization_results_{timestamp}.csv')

# if __name__ == "__main__":
#     main()