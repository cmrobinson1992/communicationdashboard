import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import os
import sys
import matplotlib.pyplot as plt
from plots import plot_confusion_matrix, plot_roc

from svm import SVM
from metrics import Metrics
from plots import plot_confusion_matrix, plot_roc
from preprocess import Preprocess
from encode import TfIdf

script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "./Data/Email Data Run 03.10.2025.csv"))
df = df.dropna(subset = ['Subject']).dropna(subset = ['Body'])
documents = (df["Subject"].astype(str) + " " + df["Body"].astype(str)).tolist()

label_encoder = LabelEncoder()
df['Y_encoded'] = label_encoder.fit_transform(df['Y'])
y = df["Y_encoded"].tolist()

docs_train, docs_val, y_train, y_val = train_test_split(documents, y, test_size=0.1, random_state=52892, stratify=y)

preprocessor = Preprocess()
docs_train_clean = preprocessor.clean_dataset(docs_train)
docs_val_clean = preprocessor.clean_dataset(docs_val)

encoder = TfIdf()
encoder.fit(docs_train_clean)
X_train = encoder.transform(docs_train_clean)
X_val = encoder.transform(docs_val_clean)

# Train SVM model
svm_model = SVC(C=1, kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)

# Predict on validation set
y_pred = svm_model.predict(X_val)
y_probs = svm_model.predict_proba(X_val)[:, 1] 

# Confusion matrix
cm = confusion_matrix(y_val, y_pred)
plot_confusion_matrix(cm, classes=label_encoder.classes_, title="Confusion Matrix on Validation Set")

# ROC/AUC
if len(label_encoder.classes_) == 2:
    auc = roc_auc_score(y_val, y_probs)
    fpr, tpr, _ = roc_curve(y_val, y_probs)
    plot_roc(fpr, tpr, auc, title="ROC Curve on Validation Set")
else:
    print("ROC curve requires binary classification.")