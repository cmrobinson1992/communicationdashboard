import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

from split import split_data
from preprocess import Preprocess
from encode import TfIdf
from classify import MNB 
from metrics import Metrics
from plots import plot_confusion_matrix, plot_roc

# STEP 1: Load CSV and extract documents
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "./Data/Email Data Run 03.10.2025.csv"))
df = df.dropna(subset = ['Subject']).dropna(subset = ['Body'])
documents = (df["Subject"].astype(str) + " " + df["Body"].astype(str)).tolist()
label_encoder = LabelEncoder()
df['Y_encoded'] = label_encoder.fit_transform(df['Y'])
y = df["Y_encoded"].tolist()

# STEP 2: Split data into train/test
x_train, x_test, y_train, y_test = train_test_split(documents, y, stratify=y, test_size=0.3, random_state = 19)

# STEP 3: Preprocess
preprocessor = Preprocess()
x_train_cleaned = preprocessor.clean_dataset(x_train)
x_test_cleaned = preprocessor.clean_dataset(x_test)

# STEP 4: Encode (choose BagOfWords or TfIdf)
encoder = TfIdf()
encoder.fit(x_train_cleaned)
X_train_vec = encoder.transform(x_train_cleaned)
X_test_vec = encoder.transform(x_test_cleaned)

# STEP 5: Train and Evaluate
param_grid = {'alpha': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train_vec, y_train)
best_alpha = grid_search.best_params_['alpha']
print(f"Best alpha from 10-fold CV: {best_alpha}")

alphas = grid_search.cv_results_['param_alpha'].data
mean_scores = grid_search.cv_results_['mean_test_score']
std_scores = grid_search.cv_results_['std_test_score']
print(alphas, std_scores)

plt.figure(figsize = (14,8))
plt.plot(alphas, mean_scores, marker='o', color = '#418FDE', linewidth = 5)
plt.xlabel("Alpha", fontsize = 24, fontweight = 'bold')
plt.ylabel("Mean CV Accuracy", fontsize = 24, fontweight = 'bold')
plt.title("Accuracy by Alpha (10-Fold CV) for Multinomial Naive Bayes", fontsize = 32, fontweight = 'bold', pad = 20)
plt.xticks(fontsize = 20, fontweight = 'bold')
plt.yticks(fontsize = 20, fontweight = 'bold')
plt.grid(False)

best_alpha = grid_search.best_params_['alpha']
best_score = grid_search.best_score_
plt.axvline(x=best_alpha, linestyle='--', color='#C8102E', label=f"Best alpha = {best_alpha}", linewidth = 3)
plt.scatter(best_alpha, best_score, color = '#C8102E', zorder = 5, s=100)
label_text = f"(Best Alpha: {best_alpha}, Accuracy: {best_score*100:.1f}%)"
plt.annotate(label_text, xy= (best_alpha, best_score), xytext = (best_alpha + 0.2, best_score), ha = 'center', fontsize = 16, fontweight = 'bold')
plt.tight_layout()
plt.show()


clf = MNB()
clf.fit(X_train_vec, y_train)
y_hat = clf.predict(X_test_vec)

metrics = Metrics()
print("Accuracy (Subject):", metrics.accuracy(y_test, y_hat))
print("Precision (Subject):", metrics.precision(y_test, y_hat, average = 'macro'))
print("Recall (Subject):", metrics.recall(y_test, y_hat, average = 'macro'))
print("F1 Score (Subject):", metrics.f1_score(y_test, y_hat, average = 'macro'))
print("Confusion Matrix (Subject):", confusion_matrix(y_test, y_hat))
plot_confusion_matrix(y_test, y_hat, label_encoder.classes_, normalize = True)
# y_score = clf.clf.predict_proba(x_test)
# plot_roc(y_test, y_score, label_encoder.classes_)

y_pred_subject = label_encoder.inverse_transform(y_hat)

# Repeat for Body:

# STEP 1: Load CSV and extract documents
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "./Data/Email Data Run 03.10.2025.csv"))
documents = df["Body"]
label_encoder = LabelEncoder()
df['Y_encoded'] = label_encoder.fit_transform(df['Y'])
y = df["Y_encoded"].tolist()

# STEP 2: Split data into train/test
x_train, x_test, y_train, y_test = train_test_split(documents, y, stratify=y, test_size=0.3, random_state = 19)
print(pd.Series(y_train).value_counts())
print(pd.Series(y_test).value_counts())

# STEP 3: Preprocess
preprocessor = Preprocess()
x_train_cleaned = preprocessor.clean_dataset(x_train)
x_test_cleaned = preprocessor.clean_dataset(x_test)

# STEP 4: Encode (choose BagOfWords or TfIdf)
encoder = TfIdf()  # or TfIdf()
encoder.fit(x_train_cleaned)
X_train_vec = encoder.transform(x_train_cleaned)
X_test_vec = encoder.transform(x_test_cleaned)

# STEP 5: Train and Evaluate
clf = MNB()  # or GNB()
clf.fit(X_train_vec, y_train)
y_hat = clf.predict(X_test_vec)

metrics = Metrics()
print("Accuracy (Body):", metrics.accuracy(y_test, y_hat))
print("Precision (Body):", metrics.precision(y_test, y_hat, average = 'macro'))
print("Recall (Body):", metrics.recall(y_test, y_hat, average = 'macro'))
print("F1 Score (Body):", metrics.f1_score(y_test, y_hat, average = 'macro'))
print("Confusion Matrix (Body):", confusion_matrix(y_test, y_hat))
plot_confusion_matrix(y_test, y_hat, label_encoder.classes_, normalize = True)
# y_score = clf.clf.predict_proba(x_test)
# plot_roc(y_test, y_score, label_encoder.classes_)

y_pred_body = label_encoder.inverse_transform(y_hat)
