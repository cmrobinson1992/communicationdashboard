import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from skorch import NeuralNetClassifier
from torch.nn import ReLU, Tanh, ELU
from skorch.callbacks import ProgressBar
from skorch import NeuralNetClassifier
import torch.nn.utils as utils
from skorch.callbacks import Callback
import sys, os

# Import local modules
from preprocess import Preprocess
from glove import Glove

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv("Data/Email_Data.csv")
# df = pd.read_csv(os.path.join(script_dir, "./Data/Email Data Run 03.10.2025.csv"))
df = df.dropna(subset = ['Subject']).dropna(subset = ['Body'])
documents = (df["Subject"].astype(str) + " " + df["Body"].astype(str)).tolist()
label_encoder = LabelEncoder()
df['Y_encoded'] = label_encoder.fit_transform(df['Y'])
y = df["Y_encoded"].tolist()

# Split data
x_train, x_test, y_train, y_test = train_test_split(documents, y, stratify=y, test_size=0.3, random_state=19)

# Preprocess
preprocessor = Preprocess()
x_train_cleaned = preprocessor.clean_dataset(x_train)
x_test_cleaned = preprocessor.clean_dataset(x_test)

# Tokenization
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def numericalize(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

class PrintParamsCallback(Callback):
    def on_train_begin(self, net, X, y):
        print("\n🔧 Training with parameters:")
        for k, v in net.get_params().items():
            if any(sub in k for sub in ["module__", "lr", "batch_size", "max_epochs"]):
                print(f"  {k}: {v}")

tokenized_docs = [tokenize(doc) for doc in documents]
counter = Counter(word for doc in tokenized_docs for word in doc)
vocab = {word: i + 2 for i, (word, _) in enumerate(counter.most_common(20000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1
tokenized_train = [tokenize(doc) for doc in x_train_cleaned]
indexed_train = [torch.tensor(numericalize(doc, vocab)) for doc in tokenized_train]
padded_train = pad_sequence(indexed_train, batch_first=True, padding_value=vocab["<PAD>"])
padded_train_tensor = padded_train
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
# print("y_train_tensor dtype:", y_train_tensor.dtype)
# print("y_train_tensor min/max:", y_train_tensor.min(), y_train_tensor.max())

indexed_docs = [torch.tensor(numericalize(doc, vocab)) for doc in tokenized_docs]
padded_docs = pad_sequence(indexed_docs, batch_first=True, padding_value=vocab["<PAD>"])

# Load GloVe
glove_loader = Glove()
# glove_model = glove_loader.load_glove_model(os.path.join(script_dir, "./Data/glove.6B.50d.txt"))
glove_model = glove_loader.load_glove_model("Data/glove.6B.50d.txt")

embedding_dim = 50
embedding_matrix = np.random.normal(scale=0.6, size=(len(vocab), embedding_dim))
for word, idx in vocab.items():
    if word in glove_model:
        embedding_matrix[idx] = glove_model[word]

from types import SimpleNamespace
w2vmodel = SimpleNamespace()
w2vmodel.wv = SimpleNamespace()
w2vmodel.wv.vectors = embedding_matrix

# CNN definition
class CNNWrapper(nn.Module):
    def __init__(self, w2vmodel, num_classes, window_sizes=(3, 4, 5), num_conv_layers=1,
                 hidden_dim=100, activation=nn.ReLU):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(w2vmodel.wv.vectors), freeze=False)

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 10, (ws, embedding_dim), padding=(ws - 1, 0)),
                activation(),
                nn.MaxPool2d(kernel_size=(2, 1))
            )
            for ws in window_sizes
        ])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(10 * len(window_sizes), hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conv_results = [torch.max(conv(x).squeeze(3), dim=2)[0] for conv in self.convs]
        x = torch.cat(conv_results, dim=1)
        x = self.dropout(self.activation()(self.fc(x)))
        return self.out(x)

class ClippedNeuralNetClassifier(NeuralNetClassifier):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("callbacks", [])
        super().__init__(*args, **kwargs)

    def train_step(self, batch, **fit_params):
        self.module_.train()
        Xi, yi = batch
        self.optimizer_.zero_grad()
        y_pred = self.infer(Xi)
        loss = self.get_loss(y_pred, yi, X=Xi)

 #       with torch.autograd.detect_anomaly():
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.module_.parameters(), max_norm=1.0)

            # Optional: show gradient magnitudes
    #        for name, param in self.module_.named_parameters():
    #            if param.grad is not None:
    #                print(f"{name}: grad max {param.grad.abs().max().item():.4f}")

        self.optimizer_.step()
        return {'loss': loss.detach(), 'y_pred': y_pred}

# Wrap with skorch
net = ClippedNeuralNetClassifier(
    CNNWrapper,
    module__w2vmodel=w2vmodel,
    module__num_classes=len(label_encoder.classes_),
    max_epochs=5,
    batch_size=64,
    lr=0.001,
    optimizer=torch.optim.Adam,
    iterator_train__shuffle=True,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    criterion=torch.nn.CrossEntropyLoss,
    callbacks=[PrintParamsCallback(), ProgressBar()]
)
x_train_indexed = [torch.tensor(numericalize(tokenize(doc), vocab)) for doc in x_train_cleaned]
x_train_padded = pad_sequence(x_train_indexed, batch_first=True, padding_value=vocab["<PAD>"])

# net.fit(x_train_padded, np.array(y_train, dtype=np.int64))
# net.initialize()
# with torch.no_grad():
#    sample_input = padded_train[:4]  # small batch
#    output = net.forward(sample_input)
#    print("Model output:", output)
#    print("Contains NaNs:", torch.isnan(output).any().item())

# Grid search
param_grid = {
    'lr': [0.001, 0.005],
    'max_epochs': [5, 10, 20],
    'batch_size': [32, 64],
    'module__activation': [ReLU, Tanh, ELU],
    'module__num_conv_layers': [1],
    'module__window_sizes': [(2, 3), (3, 4, 5)],
    'module__hidden_dim': [50, 100]
}
# max_index = padded_train.max().item()
# embedding_size = w2vmodel.wv.vectors.shape[0]
# print("Max index in padded_train:", max_index)
# print("Embedding matrix size:", embedding_size)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(net, param_grid, cv=cv, scoring='accuracy', verbose=1, error_score='raise')
print("📦 Input tensor device:", padded_train_tensor.device)
print("🎯 Target tensor device:", y_train_tensor.device)

grid.fit(padded_train_tensor, y_train_tensor)

# Save results
results_df = pd.DataFrame(grid.cv_results_)
results_df.to_csv("cnn_glove_cv_results.csv", index=False)
best_params = grid.best_params_
best_score = grid.best_score_
best_params, best_score
