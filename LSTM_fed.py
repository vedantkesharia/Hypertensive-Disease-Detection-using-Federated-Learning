import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import os
import optuna
from optuna.trial import Trial

from LSTM_model import LSTMModel
from utils import load_data, create_dataloaders, calculate_metrics


def train_client(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    return calculate_metrics(all_labels, all_preds)

def federated_averaging(models):
    """Average the weights of all client models"""
    state_dict = models[0].state_dict()
    for key in state_dict.keys():
        state_dict[key] = torch.stack([model.state_dict()[key] for model in models]).mean(0)
    return state_dict


class FederatedOptimizer:
    def __init__(self, X, y, num_clients=3, num_rounds=3):
        self.X = X
        self.y = y
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.num_classes = len(np.unique(y))
        self.input_size = X.shape[2]
        self.best_metrics = None
        self.best_trial_metrics = None

    def objective(self, trial: Trial):
        # Single set of hyperparameters for all clients in this trial
        params = {
            'hidden_size': trial.suggest_int('hidden_size', 32, 256),
            'num_layers': trial.suggest_int('num_layers', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        }

        # Create client dataloaders
        client_data = create_dataloaders(self.X, self.y, self.num_clients, params['batch_size'])
        
        # Initialize models with same architecture for all clients
        models = []
        optimizers = []
        for _ in range(self.num_clients):
            model = LSTMModel(
                input_size=self.input_size,
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                num_classes=self.num_classes
            )
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            models.append(model)
            optimizers.append(optimizer)

        criterion = nn.CrossEntropyLoss()
        best_f1 = 0.0

        # Pre-federation individual training
        print(f"\nPre-federation training for trial {trial.number}")
        for client_idx, (model, optimizer, data) in enumerate(zip(models, optimizers, client_data)):
            for epoch in range(5):  # Local optimization epochs
                loss = train_client(model, data['train'], optimizer, criterion)
                metrics = evaluate_model(model, data['test'])
                print(f"Client {client_idx}, Epoch {epoch + 1}, Loss: {loss:.4f}, F1: {metrics['f1']:.4f}")

        # Federated learning rounds
        for round_num in range(self.num_rounds):
            print(f"\nFederated Round {round_num + 1}/{self.num_rounds}")
            
            # Federated averaging
            averaged_state_dict = federated_averaging(models)
            
            # Update all models
            for model in models:
                model.load_state_dict(averaged_state_dict)
            
            # Train clients with federated model
            for client_idx, (model, optimizer, data) in enumerate(zip(models, optimizers, client_data)):
                loss = train_client(model, data['train'], optimizer, criterion)
                metrics = evaluate_model(model, data['test'])
                print(f"Client {client_idx}, Loss: {loss:.4f}, F1: {metrics['f1']:.4f}")

            # Evaluate federated model
            all_metrics = []
            for model, data in zip(models, client_data):
                metrics = evaluate_model(model, data['test'])
                all_metrics.append(metrics)
            
            # Average metrics
            avg_metrics = {
                metric: np.mean([m[metric] for m in all_metrics])
                for metric in all_metrics[0].keys()
            }
            
            print(f"Global metrics for round {round_num + 1}:")
            for metric, value in avg_metrics.items():
                print(f"{metric}: {value:.4f}")
            
            if self.best_trial_metrics is None or avg_metrics['f1'] > self.best_trial_metrics['f1']:
                self.best_trial_metrics = avg_metrics
                os.makedirs('models', exist_ok=True)
                torch.save(averaged_state_dict, 
                         f'models/best_model_trial_{trial.number}_round_{round_num + 1}.pt')
                best_f1 = avg_metrics['f1']

        return best_f1

# class FederatedOptimizer:
#     def __init__(self, X, y, num_clients=3, num_rounds=3):
        
#         self.X = X
#         self.y = y
#         self.num_clients = num_clients
#         self.num_rounds = num_rounds
#         self.num_classes = len(np.unique(y))
#         self.input_size = X.shape[2]
#         self.best_metrics = None
#         self.best_trial_metrics = None

#     def objective(self, trial: Trial):
#         # Hyperparameters to optimize
#         params = {
#             'hidden_size': trial.suggest_int('hidden_size', 32, 256),
#             'num_layers': trial.suggest_int('num_layers', 1, 3),
#             'dropout': trial.suggest_float('dropout', 0.0, 0.5),
#             'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
#             'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
#         }

#         # Create client dataloaders with current batch size
#         client_data = create_dataloaders(self.X, self.y, self.num_clients, params['batch_size'])

#         # Initialize models for each client
#         models = []
#         optimizers = []
#         for _ in range(self.num_clients):
#             model = LSTMModel(
#                 input_size=self.input_size,
#                 hidden_size=params['hidden_size'],
#                 num_layers=params['num_layers'],
#                 dropout=params['dropout'],
#                 num_classes=self.num_classes
#             )
#             optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
#             models.append(model)
#             optimizers.append(optimizer)

#         criterion = nn.CrossEntropyLoss()
#         best_f1 = 0.0

#         # Training loop
#         for round_num in range(self.num_rounds):
#             # Train each client
#             for client_idx, (model, optimizer, data) in enumerate(zip(models, optimizers, client_data)):
#                 loss = train_client(model, data['train'], optimizer, criterion)
#                 trial.report(loss, round_num * self.num_clients + client_idx)

#                 if trial.should_prune():
#                     raise optuna.TrialPruned()

#             # Federated averaging
#             averaged_state_dict = federated_averaging(models)
#             for model in models:
#                 model.load_state_dict(averaged_state_dict)

#             # Evaluate global model
#             all_metrics = []
#             for model, data in zip(models, client_data):
#                 metrics = evaluate_model(model, data['test'])
#                 all_metrics.append(metrics)

#             # Average metrics across clients
#             avg_metrics = {
#                 metric: np.mean([m[metric] for m in all_metrics])
#                 for metric in all_metrics[0].keys()
#             }

#             # Update best metrics
#             if avg_metrics['f1'] > best_f1:
#                 best_f1 = avg_metrics['f1']
#                 if self.best_trial_metrics is None or best_f1 > self.best_trial_metrics['f1']:
#                     self.best_trial_metrics = avg_metrics
#                     os.makedirs('models', exist_ok=True)
#                     torch.save(averaged_state_dict, 
#                              f'models/best_model_trial_{trial.number}_round_{round_num + 1}.pt')

#         return best_f1

def main():
    # Load data
    X, y = load_data()

    # Create study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name="federated_lstm_optimization"
    )

    # Initialize optimizer
    fed_optimizer = FederatedOptimizer(X, y)

    # Run optimization
    n_trials = 20  # Number of trials for hyperparameter optimization
    study.optimize(fed_optimizer.objective, n_trials=n_trials)

    # Print optimization results
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

    # Save study results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('results', exist_ok=True)
    study.trials_dataframe().to_csv(f'results/optimization_results_{timestamp}.csv')

if __name__ == "__main__":
    main()



# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from datetime import datetime
# import os

# from LSTM_model import LSTMModel
# from syft_utils import load_data, create_dataloaders, calculate_metrics

# def train_client(model, train_loader, optimizer, criterion):
#     model.train()
#     total_loss = 0
    
#     for inputs, labels in train_loader:
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
    
#     return total_loss / len(train_loader)

# def evaluate_model(model, test_loader):
#     model.eval()
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.numpy())
#             all_labels.extend(labels.numpy())
    
#     return calculate_metrics(all_labels, all_preds)

# def federated_averaging(models):
#     """Average the weights of all client models"""
#     state_dict = models[0].state_dict()
#     for key in state_dict.keys():
#         state_dict[key] = torch.stack([model.state_dict()[key] for model in models]).mean(0)
#     return state_dict

# def main():
#     # Load data
#     X, y = load_data()
#     num_classes = len(np.unique(y))
#     input_size = X.shape[2]
    
#     # Create client dataloaders
#     num_clients = 3
#     client_data = create_dataloaders(X, y, num_clients)
    
#     # Initialize models for each client
#     models = []
#     optimizers = []
#     for _ in range(num_clients):
#         model = LSTMModel(
#             input_size=input_size,
#             hidden_size=128,
#             num_layers=1,
#             dropout=0.0,
#             num_classes=num_classes
#         )
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
#         models.append(model)
#         optimizers.append(optimizer)
    
#     criterion = nn.CrossEntropyLoss()
#     num_rounds = 3
#     best_metrics = None
    
#     # Training loop
#     for round_num in range(num_rounds):
#         print(f"\n=== Round {round_num + 1}/{num_rounds} ===")
        
#         # Train each client
#         for client_idx, (model, optimizer, data) in enumerate(zip(models, optimizers, client_data)):
#             loss = train_client(model, data['train'], optimizer, criterion)
#             print(f"Client {client_idx + 1} Loss: {loss:.4f}")
        
#         # Federated averaging
#         averaged_state_dict = federated_averaging(models)
        
#         # Update all models with averaged weights
#         for model in models:
#             model.load_state_dict(averaged_state_dict)
        
#         # Evaluate global model
#         all_metrics = []
#         for client_idx, (model, data) in enumerate(zip(models, client_data)):
#             metrics = evaluate_model(model, data['test'])
#             all_metrics.append(metrics)
#             print(f"\nClient {client_idx + 1} Metrics:")
#             for metric, value in metrics.items():
#                 print(f"{metric}: {value:.4f}")
        
#         # Average metrics across clients
#         avg_metrics = {
#             metric: np.mean([m[metric] for m in all_metrics])
#             for metric in all_metrics[0].keys()
#         }
        
#         print("\nGlobal Model Metrics:")
#         for metric, value in avg_metrics.items():
#             print(f"{metric}: {value:.4f}")
        
#         # Save best model
#         if best_metrics is None or avg_metrics['f1'] > best_metrics['f1']:
#             best_metrics = avg_metrics
#             os.makedirs('models', exist_ok=True)
#             torch.save(averaged_state_dict, f'models/best_model_round_{round_num + 1}.pt')
#             print(f"\nSaved best model for round {round_num + 1}")
    
#     print("\nTraining completed!")
#     print("\nBest model metrics:")
#     for metric, value in best_metrics.items():
#         print(f"{metric}: {value:.4f}")

# if __name__ == "__main__":
#     main()