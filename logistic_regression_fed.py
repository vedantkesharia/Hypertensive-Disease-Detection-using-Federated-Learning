import numpy as np
import optuna
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import clone
from fed_utility import FederatedBaseOptimizer
import os
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class FederatedLogisticOptimizer(FederatedBaseOptimizer):
    def __init__(self, X, y, num_clients=3, num_rounds=5):
        super().__init__(X, y, num_clients, num_rounds)
        self.best_model = None
        self.best_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.original_labels = np.unique(y)
        self.y = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.original_labels)
        self.best_round = 0
        self.best_score = 0

    def split_data(self):
        """Split data ensuring consistent label encoding"""
        client_data = []
        skf = StratifiedKFold(n_splits=self.num_clients, shuffle=True, random_state=42)
        
        for train_idx, test_idx in skf.split(self.X, self.y):
            X_client = self.X[train_idx]
            y_client = self.y[train_idx]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_client, y_client,
                test_size=0.2,
                stratify=y_client,
                random_state=42
            )
            
            client_data.append({
                'train': (X_train, y_train),
                'test': (X_test, y_test)
            })
        return client_data

    # def federated_averaging(self, models, client_data):
    #     """Average the coefficients and intercepts of logistic regression models"""
    #     try:
    #         model_weights = []
    #         valid_models = []
            
    #         # Calculate weights based on model performance
    #         for model, data in zip(models, client_data):
    #             if not hasattr(model, 'coef_'):
    #                 continue
                    
    #             X_test, y_test = data['test']
    #             try:
    #                 y_pred = model.predict(X_test)
    #                 score = f1_score(y_test, y_pred, average='weighted')
    #                 model_weights.append(score)
    #                 valid_models.append(model)
    #             except Exception as e:
    #                 print(f"Error evaluating model: {str(e)}")
    #                 continue
            
    #         if not valid_models:
    #             raise ValueError("No valid models for federation")
            
    #         # Normalize weights
    #         model_weights = np.array(model_weights)
    #         model_weights = model_weights / (np.sum(model_weights) + 1e-10)
            
    #         # Weighted average of coefficients and intercepts
    #         avg_coef = np.average([model.coef_ for model in valid_models], 
    #                             weights=model_weights, axis=0)
    #         avg_intercept = np.average([model.intercept_ for model in valid_models], 
    #                                  weights=model_weights)
            
    #         # Create federated model with averaged parameters
    #         fed_model = LogisticRegression(
    #             multi_class='multinomial',
    #             max_iter=1000,
    #             solver='lbfgs'
    #         )
            
    #         # Initialize the model with a small subset
    #         X_init, y_init = client_data[0]['train']
    #         fed_model.fit(X_init[:1], y_init[:1])
            
    #         # Set the averaged parameters
    #         fed_model.coef_ = avg_coef
    #         fed_model.intercept_ = avg_intercept
    #         fed_model.classes_ = np.arange(self.num_classes)
            
    #         return fed_model
            
    #     except Exception as e:
    #         print(f"Federation error in federated_averaging: {str(e)}")
    #         raise

    def federated_averaging(self, models, client_data):
        try:
            model_weights = []
            valid_models = []
            
            for model, data in zip(models, client_data):
                if not hasattr(model, 'coef_'):
                    continue
                    
                X_test, y_test = data['test']
                try:
                    y_pred = model.predict(X_test)
                    score = f1_score(y_test, y_pred, average='weighted')
                    model_weights.append(score)
                    valid_models.append(model)
                except Exception as e:
                    print(f"Error evaluating model: {str(e)}")
                    continue
            
            if not valid_models:
                raise ValueError("No valid models for federation")
            
            # Normalize weights
            model_weights = np.array(model_weights)
            model_weights = model_weights / (np.sum(model_weights) + 1e-10)
            
            # Get shapes for validation
            coef_shape = valid_models[0].coef_.shape
            intercept_shape = valid_models[0].intercept_.shape
            
            # Stack coefficients and intercepts
            coef_stack = np.stack([model.coef_ for model in valid_models])
            intercept_stack = np.stack([model.intercept_ for model in valid_models])
            
            # Weighted average with explicit axis
            avg_coef = np.average(coef_stack, weights=model_weights, axis=0)
            avg_intercept = np.average(intercept_stack, weights=model_weights, axis=0)
            
            # Verify shapes
            assert avg_coef.shape == coef_shape, "Coefficient shape mismatch"
            assert avg_intercept.shape == intercept_shape, "Intercept shape mismatch"
            
            # Create federated model with averaged parameters
            fed_model = LogisticRegression(
                multi_class='multinomial',
                max_iter=1000,
                solver='lbfgs'
            )
            
            # Initialize the model
            X_init, y_init = client_data[0]['train']
            fed_model.fit(X_init[:self.num_classes], y_init[:self.num_classes])
            
            # Set the averaged parameters
            fed_model.coef_ = avg_coef
            fed_model.intercept_ = avg_intercept
            fed_model.classes_ = np.arange(self.num_classes)
            
            return fed_model
            
        except Exception as e:
            print(f"Federation error in federated_averaging: {str(e)}")
            raise

    def objective(self, trial):
        params = {
            'C': trial.suggest_float('C', 1e-3, 10, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'saga',  # Compatible with both l1 and l2
            'multi_class': 'multinomial',
            'max_iter': 1000,
            'random_state': 42
        }
        
        client_data = self.split_data()
        models = []
        
        # Initialize models
        for _ in range(self.num_clients):
            model = LogisticRegression(**params)
            models.append(model)
        
        best_f1 = 0.0
        patience = 3
        no_improve_count = 0
        min_improvement = 0.001
        
        for round_num in range(self.num_rounds):
            print(f"\nFederated Round {round_num + 1}/{self.num_rounds}")
            
            # Train client models
            valid_models = []
            for client_idx, (model, data) in enumerate(zip(models, client_data)):
                X_train, y_train = data['train']
                X_test, y_test = data['test']
                
                try:
                    model.fit(X_train, y_train)
                    metrics = self.evaluate_model(model, X_test, y_test)
                    print(f"Client {client_idx + 1} metrics:", metrics)
                    valid_models.append(model)
                except Exception as e:
                    print(f"Error training client {client_idx + 1}: {str(e)}")
                    continue
            
            if not valid_models:
                print("No valid models in this round")
                continue
            
            # Update models list to only include valid models
            models = valid_models
            
            # Federation
            try:
                fed_model = self.federated_averaging(models, client_data)
                
                # Evaluate federated model
                all_metrics = []
                for data in client_data:
                    metrics = self.evaluate_model(fed_model, *data['test'])
                    all_metrics.append(metrics)
                
                avg_metrics = {
                    metric: np.mean([m[metric] for m in all_metrics])
                    for metric in all_metrics[0].keys()
                }
                
                print(f"Global metrics for round {round_num + 1}:", avg_metrics)
                
                current_f1 = avg_metrics['f1']
                if current_f1 > (best_f1 + min_improvement):
                    best_f1 = current_f1
                    self.best_metrics = avg_metrics
                    self.best_model = fed_model
                    self.best_round = round_num + 1
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
            except Exception as e:
                print(f"Federation error: {str(e)}")
                continue
            
            if no_improve_count >= patience:
                print(f"Early stopping at round {round_num + 1}")
                break
            
            # Update client models
            try:
                models = [clone(fed_model) for _ in range(self.num_clients)]
            except Exception as e:
                print(f"Model cloning error: {str(e)}")
                break
        
        print(f"Best F1 score {best_f1:.4f} achieved at round {self.best_round}")
        return best_f1

def main():
    print("Loading and preprocessing data...")
    df = pd.read_csv("pca_final.csv")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    # Initialize optimizer
    fed_optimizer = FederatedLogisticOptimizer(X, y, num_rounds=5)
    n_trials = 10
    
    print(f"\nStarting hyperparameter optimization with {n_trials} trials...")
    study.optimize(fed_optimizer.objective, n_trials=n_trials)
    
    print("\n=== Optimization Results ===")
    trial = study.best_trial
    print(f"Best F1 Score: {trial.value:.4f}")
    
    print("\nBest Parameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    print("\nBest Model Metrics:")
    for metric, value in fed_optimizer.best_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/logistic_model_{timestamp}.pkl'
    results_path = f'results/logistic_results_{timestamp}.csv'
    
    joblib.dump(fed_optimizer.best_model, model_path)
    study.trials_dataframe().to_csv(results_path)
    print(f"\nModel saved to: {model_path}")
    print(f"Results saved to: {results_path}")

if __name__ == "__main__":
    main() 