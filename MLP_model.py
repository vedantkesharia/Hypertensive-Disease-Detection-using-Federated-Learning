import torch
import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout_rate, num_classes):
        super(MLPClassifier, self).__init__()
        # First layer
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        
        # Second layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        
        # Output layer
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
        # Activations and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First block
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Second block
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Output
        x = self.fc3(x)
        return x

# import optuna
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Load dataset
# df = pd.read_csv("pca_final.csv")

# # Split into features and target
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# # Standardize features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Convert to PyTorch tensors
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
# y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# # Define MLP model

# class MLPClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size1, hidden_size2, dropout_rate, num_classes):
#         super(MLPClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.fc3 = nn.Linear(hidden_size2, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_rate)
    
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # Function to train & evaluate the model
# def train_and_evaluate(model, train_loader, test_loader, learning_rate, num_epochs=50):
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     for epoch in range(num_epochs):
#         model.train()
#         for inputs, labels in train_loader:
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#     # Evaluate model
#     model.eval()
#     y_pred = []
#     y_true = []

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             y_pred.extend(predicted.tolist())
#             y_true.extend(labels.tolist())

#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average="weighted")
#     recall = recall_score(y_true, y_pred, average="weighted")
#     f1 = f1_score(y_true, y_pred, average="weighted")

#     return accuracy, precision, recall, f1

# # Train & evaluate before hyperparameter tuning
# print("\n📢 Before Hyperparameter Tuning:")
# default_model = MLPClassifier(X.shape[1], 128, 64, 0.3, len(np.unique(y)))
# train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
# test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

# acc, prec, rec, f1 = train_and_evaluate(default_model, train_loader, test_loader, learning_rate=0.001)
# print(f"Accuracy: {acc:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f} | F1 Score: {f1:.2f}")

# # Optuna hyperparameter tuning
# def objective(trial):
#     hidden_size1 = trial.suggest_int("hidden_size1", 64, 512)
#     hidden_size2 = trial.suggest_int("hidden_size2", 32, 256)
#     dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
#     learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
#     batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

#     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

#     model = MLPClassifier(X.shape[1], hidden_size1, hidden_size2, dropout_rate, len(np.unique(y)))
#     acc, _, _, _ = train_and_evaluate(model, train_loader, test_loader, learning_rate)
#     return acc

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=20)

# # Train & evaluate with best parameters
# best_params = study.best_params
# print("\n🚀 After Hyperparameter Tuning (Best Params):", best_params)

# best_model = MLPClassifier(X.shape[1], best_params["hidden_size1"], best_params["hidden_size2"],
#                            best_params["dropout_rate"], len(np.unique(y)))

# train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=best_params["batch_size"], shuffle=True)
# test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=best_params["batch_size"], shuffle=False)

# acc, prec, rec, f1 = train_and_evaluate(best_model, train_loader, test_loader, best_params["learning_rate"])
# print(f"Accuracy: {acc:.2f} | Precision: {prec:.2f} | Recall: {rec:.2f} | F1 Score: {f1:.2f}")
