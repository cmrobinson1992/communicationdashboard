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

embedding_dim = 50

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