import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

from preprocess import Preprocess
from encode import TfIdf, GloveEmbedding

# CNN definition
class SimpleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, output_dim=1, kernel_sizes=(2, 3), activation=nn.ReLU()):
        super(SimpleCNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=k) for k in kernel_sizes
        ])
        self.activation = activation
        self.fc = nn.Linear(hidden_dim * len(kernel_sizes), output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # add channel dimension
        conv_outs = [self.activation(conv(x)).squeeze(3).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_outs, dim=1)
        return self.fc(x).squeeze()

def train_cnn(model, train_loader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb.float())
            loss = criterion(preds, yb.float())
            loss.backward()
            optimizer.step()

def evaluate_cnn(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        preds = model(X_test.float()).numpy()
    return mean_squared_error(y_test, preds)

# Data loading
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "./Data/Email Data Run 03.10.2025.csv"))
df = df.dropna(subset=['Subject', 'Body'])
documents = (df["Subject"].astype(str) + " " + df["Body"].astype(str)).tolist()
label_encoder = LabelEncoder()
df['Y_encoded'] = label_encoder.fit_transform(df['Y'])
y = df["Y_encoded"].tolist()

# Preprocess
preprocessor = Preprocess()
x_cleaned = preprocessor.clean_dataset(documents)

# Encode
tfidf_encoder = TfIdf()
tfidf_encoder.fit(x_cleaned)
X_tfidf = tfidf_encoder.transform(x_cleaned)

glove_encoder = GloveEmbedding()
X_glove = glove_encoder.transform(x_cleaned)

# Models
model_configs = {
    "MultinomialNB(alpha=0.1)": {
        "model": MultinomialNB(alpha=0.1),
        "encoding": "tfidf"
    },
    "SVM(C=1)": {
        "model": SVC(C=1, kernel='linear'),
        "encoding": "tfidf"
    },
    "CNN": {
        "model": "cnn",  # to instantiate later
        "encoding": "glove"
    }
}

# Monte Carlo Simulation
num_simulations = 50
np.random.seed(2135)
results = []

for model_name, config in model_configs.items():
    mse_list = []
    for b in range(1, num_simulations + 1):
        print(f"Running Simulation {b}/{num_simulations} for Model: {model_name}")
        idx = np.arange(len(x_cleaned))
        train_idx, test_idx = train_test_split(idx, stratify=np.array(y), test_size=0.3, random_state=2135 + b)
        y_train = np.array([y[i] for i in train_idx])
        y_test = np.array([y[i] for i in test_idx])

        if config["encoding"] == "tfidf":
            X_train = X_tfidf[train_idx]
            X_test = X_tfidf[test_idx]
        elif config["encoding"] == "glove":
            X_train = np.array([X_glove[i] for i in train_idx])
            X_test = np.array([X_glove[i] for i in test_idx])
        else:
            raise ValueError(f"Unsupported encoding type: {config['encoding']}")

        if model_name == "CNN":
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

            model = SimpleCNN(input_dim=X_train.shape[1])
            train_cnn(model, train_loader, epochs=10)
            mse = evaluate_cnn(model, X_test_tensor, y_test)
        else:
            model = config["model"]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)

        mse_list.append(mse)

    results.append({
        "Model": model_name,
        "MSE Mean": np.mean(mse_list),
        "MSE Variance": np.var(mse_list)
    })

results_df = pd.DataFrame(results)
print(results_df)
