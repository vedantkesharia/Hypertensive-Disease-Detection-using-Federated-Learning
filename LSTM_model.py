import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = self.fc(hn[-1])
        return x





# import tensorflow as tf

# def create_lstm_model(input_shape, num_classes):
#     model = tf.keras.Sequential([
#         tf.keras.layers.LSTM(128, input_shape=input_shape),
#         tf.keras.layers.Dense(num_classes, activation='softmax')
#     ])
#     return model

# def create_model_and_optimizer():
#     model = create_lstm_model((1, input_shape[2]), num_classes)
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=['accuracy']
#     )
#     return model





# import torch
# import torch.nn as nn

# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
#                            batch_first=True, dropout=dropout)
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         x = self.fc(hn[-1])
#         return x







# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import flwr as fl
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # -------------------------------
# # Data Loading and Preprocessing
# # -------------------------------
# df = pd.read_csv("pca_final.csv")
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Reshape for LSTM (samples, time steps, features)
# X = X.reshape(X.shape[0], 1, X.shape[1])

# # Simulate 3 clients by splitting the dataset
# num_clients = 3
# X_splits = np.array_split(X, num_clients)
# y_splits = np.array_split(y, num_clients)

# def prepare_data(X_split, y_split):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_split, y_split, test_size=0.2, random_state=42
#     )
#     X_train = torch.tensor(X_train, dtype=torch.float32)
#     X_test = torch.tensor(X_test, dtype=torch.float32)
#     y_train = torch.tensor(y_train, dtype=torch.long)
#     y_test = torch.tensor(y_test, dtype=torch.long)
    
#     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
#     test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
#     return train_loader, test_loader

# train_loaders, test_loaders = zip(*[prepare_data(X_splits[i], y_splits[i]) for i in range(num_clients)])

# # -------------------------------
# # Define the LSTM Model
# # -------------------------------
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
    
#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         x = self.fc(hn[-1])
#         return x

# input_size = X.shape[2]
# hidden_size = 128
# num_classes = len(np.unique(y))
# global_model = LSTMModel(input_size, hidden_size, num_classes)
# criterion = nn.CrossEntropyLoss()

# # -------------------------------
# # Define Flower Client
# # -------------------------------
# class LSTMClient(fl.client.Client):
#     def __init__(self, model, train_loader, test_loader):
#         self.model = model
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
#     def get_parameters(self):
#         return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
#     def set_parameters(self, parameters):
#         params_dict = dict(zip(self.model.state_dict().keys(), parameters))
#         state_dict = {k: torch.tensor(v) for k, v in params_dict.items()}
#         self.model.load_state_dict(state_dict, strict=True)
    
#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.train()
#         for epoch in range(5):  # 5 local training epochs per round
#             for inputs, labels in self.train_loader:
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
#         return self.get_parameters(), len(self.train_loader.dataset), {}
    
#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.eval()
#         y_pred, y_true = [], []
#         with torch.no_grad():
#             for inputs, labels in self.test_loader:
#                 outputs = self.model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 y_pred.extend(predicted.tolist())
#                 y_true.extend(labels.tolist())
        
#         accuracy = accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average="weighted")
#         recall = recall_score(y_true, y_pred, average="weighted")
#         f1 = f1_score(y_true, y_pred, average="weighted")
#         print(f"Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
#         return float(accuracy), len(self.test_loader.dataset), {
#             "accuracy": accuracy,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1
#         }

# if __name__ == "__main__":
#     # Create clients for each simulated dataset
#     clients = [LSTMClient(LSTMModel(input_size, hidden_size, num_classes), train_loaders[i], test_loaders[i])
#                for i in range(num_clients)]
    
#     # IMPORTANT: Flower clients must be started in the main thread.
#     for client in clients:
#         # Save each client instance to a file
#         with open(f"client_{clients.index(client)}.pkl", "wb") as f:
#             import pickle
#             pickle.dump(client, f)


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import flwr as fl
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# print("🚀 Loading dataset...")

# # -------------------------------
# # Data Loading and Preprocessing
# # -------------------------------
# df = pd.read_csv("pca_final.csv")
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Reshape for LSTM (samples, time steps, features)
# X = X.reshape(X.shape[0], 1, X.shape[1])

# # Simulate 3 clients by splitting the dataset
# num_clients = 3
# X_splits = np.array_split(X, num_clients)
# y_splits = np.array_split(y, num_clients)

# def prepare_data(X_split, y_split):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_split, y_split, test_size=0.2, random_state=42
#     )
#     X_train = torch.tensor(X_train, dtype=torch.float32)
#     X_test = torch.tensor(X_test, dtype=torch.float32)
#     y_train = torch.tensor(y_train, dtype=torch.long)
#     y_test = torch.tensor(y_test, dtype=torch.long)

#     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
#     test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)

#     print(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} testing samples")
#     return train_loader, test_loader

# train_loaders, test_loaders = zip(*[prepare_data(X_splits[i], y_splits[i]) for i in range(num_clients)])

# # -------------------------------
# # Define the LSTM Model
# # -------------------------------
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         x = self.fc(hn[-1])
#         return x

# input_size = X.shape[2]
# hidden_size = 128
# num_classes = len(np.unique(y))
# global_model = LSTMModel(input_size, hidden_size, num_classes)
# criterion = nn.CrossEntropyLoss()

# # -------------------------------
# # Define Flower Client using New API
# # -------------------------------
# class LSTMClient(fl.client.NumPyClient):
#     def __init__(self, model, train_loader, test_loader, client_id):
#         self.model = model
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
#         self.client_id = client_id
#         print(f"🚀 Client {client_id} initialized.")

#     def get_parameters(self, config=None):
#         return [val.cpu().numpy() for val in self.model.state_dict().values()]

#     def set_parameters(self, parameters, config=None):
#         print(f"🔄 Client {self.client_id} setting parameters...")
#         params_dict = dict(zip(self.model.state_dict().keys(), parameters))
#         state_dict = {k: torch.tensor(v) for k, v in params_dict.items()}
#         self.model.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.train()
#         print(f"🎯 Client {self.client_id} starting training...")
#         for epoch in range(5):  # 5 local training epochs per round
#             running_loss = 0.0
#             for inputs, labels in self.train_loader:
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
#                 running_loss += loss.item()

#             print(f"✅ Client {self.client_id} Epoch {epoch+1}: Loss = {running_loss / len(self.train_loader):.4f}")

#         print(f"🚀 Client {self.client_id} finished training!")
#         return self.get_parameters(), len(self.train_loader.dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.eval()
#         y_pred, y_true = [], []
#         print(f"📊 Client {self.client_id} evaluating...")

#         with torch.no_grad():
#             for inputs, labels in self.test_loader:
#                 outputs = self.model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 y_pred.extend(predicted.tolist())
#                 y_true.extend(labels.tolist())

#         accuracy = accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average="weighted")
#         recall = recall_score(y_true, y_pred, average="weighted")
#         f1 = f1_score(y_true, y_pred, average="weighted")

#         print(f"📢 Client {self.client_id} Evaluation Results:")
#         print(f"  Accuracy: {accuracy:.4f}")
#         print(f"  Precision: {precision:.4f}")
#         print(f"  Recall: {recall:.4f}")
#         print(f"  F1 Score: {f1:.4f}")

#         return float(accuracy), len(self.test_loader.dataset), {
#             "accuracy": accuracy,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1
#         }

# # -------------------------------
# # Start Flower Clients
# # -------------------------------
# def start_client(client):
#     print(f"🚀 Starting Client {client.client_id}...")
#     fl.client.start_client(
#         server_address="127.0.0.1:9092",
#         client=client.to_client()
#     )
#     print(f"✅ Client {client.client_id} connected to server!")

# if __name__ == "__main__":
#     # Create clients for each simulated dataset
#     clients = [LSTMClient(LSTMModel(input_size, hidden_size, num_classes), train_loaders[i], test_loaders[i], client_id=i+1)
#                for i in range(num_clients)]

#     # Start each client in the main thread
#     for client in clients:
#         start_client(client)




# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import flwr as fl
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # -------------------------------
# # Data Loading and Preprocessing
# # -------------------------------
# df = pd.read_csv("pca_final.csv")
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Reshape for LSTM (samples, time steps, features)
# X = X.reshape(X.shape[0], 1, X.shape[1])

# # Simulate 3 clients by splitting the dataset
# num_clients = 3
# X_splits = np.array_split(X, num_clients)
# y_splits = np.array_split(y, num_clients)

# def prepare_data(X_split, y_split):
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_split, y_split, test_size=0.2, random_state=42
#     )
#     X_train = torch.tensor(X_train, dtype=torch.float32)
#     X_test = torch.tensor(X_test, dtype=torch.float32)
#     y_train = torch.tensor(y_train, dtype=torch.long)
#     y_test = torch.tensor(y_test, dtype=torch.long)
    
#     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
#     test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
#     return train_loader, test_loader

# train_loaders, test_loaders = zip(*[prepare_data(X_splits[i], y_splits[i]) for i in range(num_clients)])

# # -------------------------------
# # Define the LSTM Model
# # -------------------------------
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
    
#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         x = self.fc(hn[-1])
#         return x

# input_size = X.shape[2]
# hidden_size = 128
# num_classes = len(np.unique(y))
# global_model = LSTMModel(input_size, hidden_size, num_classes)
# criterion = nn.CrossEntropyLoss()

# # -------------------------------
# # Define Flower Client using New API
# # -------------------------------
# class LSTMClient(fl.client.NumPyClient):
#     def __init__(self, model, train_loader, test_loader):
#         self.model = model
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
#     def get_parameters(self, config=None):
#         return [val.cpu().numpy() for val in self.model.state_dict().values()]
    
#     def set_parameters(self, parameters, config=None):
#         params_dict = dict(zip(self.model.state_dict().keys(), parameters))
#         state_dict = {k: torch.tensor(v) for k, v in params_dict.items()}
#         self.model.load_state_dict(state_dict, strict=True)
    
#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.train()
#         for epoch in range(5):  # 5 local training epochs per round
#             for inputs, labels in self.train_loader:
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
#         return self.get_parameters(), len(self.train_loader.dataset), {}
    
#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.eval()
#         y_pred, y_true = [], []
#         with torch.no_grad():
#             for inputs, labels in self.test_loader:
#                 outputs = self.model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 y_pred.extend(predicted.tolist())
#                 y_true.extend(labels.tolist())
        
#         accuracy = accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average="weighted")
#         recall = recall_score(y_true, y_pred, average="weighted")
#         f1 = f1_score(y_true, y_pred, average="weighted")
#         print(f"Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
#         return float(accuracy), len(self.test_loader.dataset), {
#             "accuracy": accuracy,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1
#         }

# # -------------------------------
# # Start Flower Clients
# # -------------------------------
# def start_client(client):
#     # Use start_client() with .to_client() as recommended
#     fl.client.start_client(
#         # server_address="127.0.0.1:8080",
#         server_address="127.0.0.1:9092",
#         client=client.to_client()
#     )

# if __name__ == "__main__":
#     # Create clients for each simulated dataset
#     clients = [LSTMClient(LSTMModel(input_size, hidden_size, num_classes), train_loaders[i], test_loaders[i])
#                for i in range(num_clients)]
    
#     # IMPORTANT: Flower clients must be started in the main thread.
#     for client in clients:
#         start_client(client)




# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# import flwr as fl
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Load and preprocess dataset
# df = pd.read_csv("pca_final.csv")
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values

# # Standardize features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Reshape for LSTM (samples, time steps, features)
# X = X.reshape(X.shape[0], 1, X.shape[1])

# # Simulating 3 clients by splitting the dataset
# num_clients = 3
# X_splits = np.array_split(X, num_clients)
# y_splits = np.array_split(y, num_clients)

# # Convert to PyTorch tensors and create DataLoaders
# def prepare_data(X_split, y_split):
#     X_train, X_test, y_train, y_test = train_test_split(X_split, y_split, test_size=0.2, random_state=42)
#     X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
#     y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)
    
#     train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
#     test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
    
#     return train_loader, test_loader

# train_loaders, test_loaders = zip(*[prepare_data(X_splits[i], y_splits[i]) for i in range(num_clients)])

# # Define LSTM Model
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, num_classes)
    
#     def forward(self, x):
#         _, (hn, _) = self.lstm(x)
#         x = self.fc(hn[-1])
#         return x

# # Global model
# input_size = X.shape[2]
# hidden_size = 128
# num_classes = len(np.unique(y))

# global_model = LSTMModel(input_size, hidden_size, num_classes)
# criterion = nn.CrossEntropyLoss()

# # Define Flower Client
# class LSTMClient(fl.client.NumPyClient):
#     def __init__(self, model, train_loader, test_loader):
#         self.model = model
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

#     def get_parameters(self, config=None):
#         return [val.cpu().numpy() for val in self.model.state_dict().values()]

#     def set_parameters(self, parameters, config=None):
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = {k: torch.tensor(v) for k, v in params_dict}
#         self.model.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.train()
#         for epoch in range(5):  # 5 local training epochs
#             for inputs, labels in self.train_loader:
#                 self.optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 self.optimizer.step()
#         return self.get_parameters(), len(self.train_loader.dataset), {}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         self.model.eval()
#         y_pred, y_true = [], []
#         with torch.no_grad():
#             for inputs, labels in self.test_loader:
#                 outputs = self.model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 y_pred.extend(predicted.tolist())
#                 y_true.extend(labels.tolist())
        
#         # Calculate Metrics
#         accuracy = accuracy_score(y_true, y_pred)
#         precision = precision_score(y_true, y_pred, average="weighted")
#         recall = recall_score(y_true, y_pred, average="weighted")
#         f1 = f1_score(y_true, y_pred, average="weighted")
        
#         print(f"Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        
#         return float(accuracy), len(self.test_loader.dataset), {
#             "accuracy": accuracy,
#             "precision": precision,
#             "recall": recall,
#             "f1_score": f1
#         }

# # Start the Flower SuperLink Server (run this in a separate terminal)
# """
# Run this command in a terminal before starting the clients:
# $ flower-superlink --insecure
# """

# # Start Clients
# def start_client(client):
#     fl.client.start_client(
#         server_address="127.0.0.1:8080",
#         client=client.to_client()
#     )

# if __name__ == "__main__":
#     # Create and start each client
#     clients = [LSTMClient(global_model, train_loaders[i], test_loaders[i]) for i in range(num_clients)]
#     for client in clients:
#         start_client(client)
