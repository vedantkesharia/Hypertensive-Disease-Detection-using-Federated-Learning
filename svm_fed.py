import numpy as np
import optuna
from sklearn.svm import SVC
from fed_utility import FederatedBaseOptimizer
import os
from datetime import datetime

class FederatedSVMOptimizer(FederatedBaseOptimizer):
    def __init__(self, X, y, num_clients=3, num_rounds=5):
        super().__init__(X, y, num_clients, num_rounds)
    
    def federated_averaging(self, models, client_data):
        """Alternative federation strategy for SVM models"""
        # Combine all training data
        X_combined = []
        y_combined = []
        
        for data in client_data:
            X_train, y_train = data['train']
            X_combined.append(X_train)
            y_combined.append(y_train)
        
        X_combined = np.concatenate(X_combined, axis=0)
        y_combined = np.concatenate(y_combined, axis=0)
        
        # Get hyperparameters from first model
        base_model = models[0]
        params = {
            'C': base_model.C,
            'kernel': base_model.kernel,
            'gamma': base_model.gamma,
            'probability': True
        }
        
        # Create and train new model with combined data
        averaged_model = SVC(**params)
        averaged_model.fit(X_combined, y_combined)
        
        return averaged_model
    
    def objective(self, trial):
        # Hyperparameters to optimize
        params = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.001, 1.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly']),
            'probability': True
        }
        
        # Split data among clients
        client_data = self.split_data()
        
        # Initialize models for each client
        models = []
        for _ in range(self.num_clients):
            model = SVC(**params)
            models.append(model)
        
        # Training and federation
        best_f1 = 0.0
        for round_num in range(self.num_rounds):
            print(f"\nFederated Round {round_num + 1}/{self.num_rounds}")
            
            # Train each client's model
            for client_idx, (model, data) in enumerate(zip(models, client_data)):
                X_train, y_train = data['train']
                model.fit(X_train, y_train)
                
                # Evaluate client model
                X_test, y_test = data['test']
                metrics = self.evaluate_model(model, X_test, y_test)
                print(f"Client {client_idx + 1} metrics:", metrics)
            
            # Federated averaging
            averaged_model = self.federated_averaging(models, client_data)
            
            # Evaluate federated model
            all_metrics = []
            for data in client_data:
                X_test, y_test = data['test']
                metrics = self.evaluate_model(averaged_model, X_test, y_test)
                all_metrics.append(metrics)
            
            # Average metrics across clients
            avg_metrics = {
                metric: np.mean([m[metric] for m in all_metrics])
                for metric in all_metrics[0].keys()
            }
            
            print(f"Global metrics for round {round_num + 1}:", avg_metrics)
            
            if avg_metrics['f1'] > best_f1:
                best_f1 = avg_metrics['f1']
                self.best_metrics = avg_metrics
                
                # Save best model
                if not os.path.exists('models'):
                    os.makedirs('models')
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = f'models/svm_model_trial_{trial.number}_round_{round_num + 1}_{timestamp}.pkl'
                import joblib
                joblib.dump(averaged_model, model_path)
                print(f"Saved best model to {model_path}")
        
        return best_f1

def main():
    # Load and preprocess data
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    
    print("Loading and preprocessing data...")
    df = pd.read_csv("pca_final.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name="federated_svm_optimization"
    )
    
    # Initialize optimizer
    fed_optimizer = FederatedSVMOptimizer(X, y)
    
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
    study.trials_dataframe().to_csv(f'results/svm_optimization_results_{timestamp}.csv')

if __name__ == "__main__":
    main()