import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import os
import sys
import matplotlib.pyplot as plt

from svm import SVM
from metrics import Metrics
from plots import plot_confusion_matrix, plot_roc
from preprocess import Preprocess
from encode import TfIdf

script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "./Data/Email Data Run 03.10.2025.csv"))
df = df.dropna(subset = ['Subject']).dropna(subset = ['Body'])
documents = (df["Subject"].astype(str) + " " + df["Body"].astype(str)).tolist()

# label_encoder = LabelEncoder()
# df['Y_encoded'] = label_encoder.fit_transform(df['Y'])
# y = df["Y_encoded"].tolist()

# X_train, X_test, y_train, y_test = train_test_split(documents, y, test_size = 0.3, random_state = 19)
# preprocessor = Preprocess()
# x_train_cleaned = preprocessor.clean_dataset(X_train)
# x_test_cleaned = preprocessor.clean_dataset(X_test)

# encoder = TfIdf()
# encoder.fit(x_train_cleaned)
# X_train_vec = encoder.transform(x_train_cleaned)
# X_test_vec = encoder.transform(x_test_cleaned)

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear']
 #   'gamma': ['scale', 'auto'],  
#    'degree': [2, 3, 4] 
}

# grid_search = GridSearchCV(SVC(), param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose = 1)
# grid_search.fit(X_train_vec, y_train)

# Get the best parameters
#best_params = grid_search.best_params_
# print("Best Parameters:", best_params)

# csv_results = pd.DataFrame(grid_search.cv_results_)
# csv_results.to_csv('svm_cv_results.csv', index = False)

cv_results = pd.read_csv(os.path.join(script_dir, "svm_cv_results.csv"))
linear_results = cv_results[cv_results['param_kernel'] == 'linear']

# Convert C to numeric
linear_results['param_C'] = linear_results['param_C'].astype(float)

# Sort by C for plotting
linear_results = linear_results.sort_values('param_C')

# Plot accuracy by C
plt.figure(figsize=(14, 8))
plt.plot(linear_results['param_C'], linear_results['mean_test_score'],
         marker='o', linewidth=5, color='#418FDE')
plt.xlabel("C", fontsize = 24, fontweight = 'bold')
plt.ylabel("Mean CV Accuracy", fontsize = 24, fontweight = 'bold')
plt.title("Accuracy by Cost (10-Fold CV) for SVM Linear", fontsize = 32, fontweight = 'bold', pad = 20)
plt.xticks(fontsize = 20, fontweight = 'bold')
plt.yticks(fontsize = 20, fontweight = 'bold')
plt.grid(False)

best_c = 10.0
best_score = 0.95202781
plt.axvline(x=best_c, linestyle='--', color='#C8102E', label=f"Best alpha = {best_c}", linewidth = 3)
plt.scatter(best_c, best_score, color = '#C8102E', zorder = 5, s=100)
label_text = f"(Best Alpha: {best_c}, Accuracy: {best_score*100:.1f}%)"
plt.annotate(label_text, xy= (best_c, best_score), xytext = (best_c - 1.5, best_score - 0.003), ha = 'center', fontsize = 16, fontweight = 'bold')
plt.tight_layout()
plt.show()