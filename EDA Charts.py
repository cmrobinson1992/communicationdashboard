import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.cm import get_cmap
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import nltk
import re
import sys, os
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from preprocess import Preprocess
from encode import TfIdf

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "./Data/Email Data Run 03.10.2025.csv"))
df = df.dropna(subset = ['Subject']).dropna(subset = ['Body'])
documents = (df["Subject"].astype(str) + " " + df["Body"].astype(str)).tolist()
label_encoder = LabelEncoder()
df['Y_encoded'] = label_encoder.fit_transform(df['Y'])
y = df["Y_encoded"].tolist()

x_train, x_test, y_train, y_test = train_test_split(df['Subject'], y, stratify=y, test_size=0.3, random_state = 19)
x_train_body, x_test_body, y_train_body, y_test_body = train_test_split(df['Body'], y, stratify=y, test_size=0.3, random_state = 19)
df_train = pd.DataFrame({'Subject':x_train, 'Body': x_train_body, 'Y': label_encoder.inverse_transform(y_train)})
print(df_train.head(10))
# Custom color palette for known categories
custom_palette = {
    'Filing Assistance': '#0F7DBC',
    'Technical Support': '#009EED',
    'Reports': '#D55B0D',
    'API Mapping': '#EC8000',
    'Audits': '#446688',
    'Other': '#DEF1FC'
}

# Helper function to clean and tokenize text
def clean_text(text):
    text = str(text).lower()
    words = text.split()
    words = [word for word in words if word.isalpha()] 
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return words

# Add text lengths
df_train['Subject_Length'] = df_train['Subject'].apply(lambda x: len(str(x).split()))
df_train['Body_Length'] = df_train['Body'].apply(lambda x: len(str(x).split()))
alphabetical_order = sorted(df_train['Y'].unique())

# 1. Email count per category (horizontal, no border)
plt.figure(figsize=(8, 5))
sns.countplot(data=df_train, y='Y', order=df_train['Y'].value_counts().index, palette=custom_palette)
plt.title('Email Count per Category', fontsize = 32, fontfamily = 'century gothic')
plt.xlabel('Count', fontsize = 24, fontfamily = 'century gothic')
plt.ylabel('')
plt.xticks(rotation=0, fontsize = 20, fontfamily = 'century gothic')
plt.yticks(fontsize = 20, fontfamily = 'century gothic')
sns.despine()
plt.tight_layout()
plt.show()

# 2. Subject Length per Category
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_train, x='Y', y='Subject_Length', order = alphabetical_order, palette=custom_palette)
plt.title('Subject Word Count by Category', fontsize = 32, fontfamily = 'century gothic')
plt.ylabel('Word Count', fontsize = 24, fontfamily = 'century gothic')
plt.xlabel('')
plt.xticks(rotation=45, fontsize = 20, fontfamily = 'century gothic')
plt.yticks(fontsize = 20, fontfamily = 'century gothic')
sns.despine(left = True, bottom = True)
plt.tight_layout()
plt.show()

#PCA Show Categories
labels = df['Y'].tolist()

# Preprocess text
preprocessor = Preprocess()
x_cleaned = preprocessor.clean_dataset(documents)

# TF-IDF encoding
tfidf_encoder = TfIdf()
tfidf_encoder.fit(x_cleaned)
X_tfidf = tfidf_encoder.transform(x_cleaned)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf)  # No need to call .toarray() if it's already a dense array
variance_ratio = pca.explained_variance_ratio_
total_variance = variance_ratio.sum()
print(f"Total variance explained by the first 2 principal components: {total_variance * 100:.2f}%")

# Plot
plt.figure(figsize=(10, 8))
for category, color in custom_palette.items():
    idx = df[df['Y'] == category].index
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], 
                label=category, 
                color=color, 
                alpha=0.7, 
                edgecolor='k', 
                s=60)

plt.title("Email Categories Projected onto First 2 Principal Components", fontsize=32, fontfamily='century gothic')
plt.xlabel("Principal Component 1", fontsize=24, fontfamily = 'century gothic')
plt.ylabel("Principal Component 2", fontsize=24, fontfamily = 'century gothic')
plt.xticks(fontsize = 20, fontfamily = 'century gothic')
plt.yticks(fontsize = 20, fontfamily = 'century gothic')
plt.legend(title="Category")
plt.grid(False)
plt.tight_layout()
plt.show()

# 3. Body Length per Category
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_train, x='Y', y='Body_Length', order = alphabetical_order, palette=custom_palette)
plt.title('Body Word Count by Category', fontsize = 32, fontfamily = 'century gothic')
plt.ylabel('Word Count', fontsize = 24, fontfamily = 'century gothic')
plt.xlabel('')
plt.xticks(rotation=45, fontsize = 20, fontfamily = 'century gothic')
plt.yticks(fontsize = 20, fontfamily = 'century gothic')
sns.despine(left = True, bottom = True)
plt.tight_layout()
plt.show()

# 4. Histogram of Body Length
plt.figure(figsize=(8, 5))
sns.histplot(df_train['Body_Length'], bins=30, kde=True, color='#446688')
plt.title('Distribution of Body Word Count', fontsize = 32, fontfamily = 'century gothic')
plt.xlabel('Word Count', fontsize = 24, fontfamily = 'century gothic')
plt.ylabel('Frequency', fontsize = 24, fontfamily = 'century gothic')
plt.xticks(fontsize = 20, fontfamily = 'century gothic')
plt.yticks(fontsize = 20, fontfamily = 'century gothic')
sns.despine()
plt.tight_layout()
plt.show()

# 5. Histogram of Subject Length
plt.figure(figsize=(8, 5))
sns.histplot(df_train['Subject_Length'], bins=30, kde=True, color='#0F7DBC')
plt.title('Distribution of Subject Word Count', fontsize = 32, fontfamily = 'century gothic')
plt.xlabel('Word Count', fontsize = 24, fontfamily = 'century gothic')
plt.ylabel('Frequency', fontsize = 24, fontfamily = 'century gothic')
plt.xticks(fontsize = 20, fontfamily = 'century gothic')
plt.yticks(fontsize = 20, fontfamily = 'century gothic')
sns.despine()
plt.tight_layout()
plt.show()

# 6. TF-IDF for Subject
subject_corpus = df_train['Subject'].fillna('').tolist()
tfidf = TfidfVectorizer(stop_words='english', max_features=20)
X_tfidf = tfidf.fit_transform(subject_corpus)
tfidf_scores = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())
top_scores = tfidf_scores.sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_scores.values, y=top_scores.index, color='#0F7DBC')
plt.title('Top 20 TF-IDF Words in Subject', fontsize = 32, fontfamily = 'century gothic')
plt.xlabel('TF-IDF Score', fontsize = 24, fontfamily = 'century gothic')
plt.ylabel('')
plt.xticks(fontsize = 20, fontfamily = 'century gothic')
plt.yticks(fontsize = 20, fontfamily = 'century gothic')
sns.despine()
plt.tight_layout()
plt.show()

# 7. TF-IDF for Body
subject_corpus = df_train['Body'].fillna('').tolist()
tfidf = TfidfVectorizer(stop_words='english', max_features=20)
X_tfidf = tfidf.fit_transform(subject_corpus)
tfidf_scores = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())
top_scores = tfidf_scores.sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_scores.values, y=top_scores.index, color='#446688')
plt.title('Top 20 TF-IDF Words in Body', fontsize = 32, fontfamily = 'century gothic')
plt.xlabel('TF-IDF Score', fontsize = 24, fontfamily = 'century gothic')
plt.ylabel('')
plt.xticks(fontsize = 20, fontfamily = 'century gothic')
plt.yticks(fontsize = 20, fontfamily = 'century gothic')
sns.despine()
plt.tight_layout()
plt.show()

# 8. Top 5 most common words in subject by category (cleaned + lemmatized)
df_train['Cleaned_Subject'] = df_train['Subject'].fillna('').apply(clean_text)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Get the top 6 categories (or define manually)
top_categories = df_train['Y'].value_counts().index[:6]

for i, category in enumerate(top_categories):
    ax = axes[i]
    # Get all words for this category
    all_words = df_train[df_train['Y'] == category]['Cleaned_Subject'].sum()
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(5)

    if top_words:
        words, counts = zip(*top_words)
        sns.barplot(x=list(counts), y=list(words), ax=ax, color=custom_palette.get(category, '#333333'))
        ax.set_title(f'{category}', fontsize=20, fontfamily='century gothic')
        ax.set_xlabel('Frequency', fontsize=14, fontfamily='century gothic')
        ax.set_ylabel('')
        ax.tick_params(labelsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontfamily='century gothic')
    else:
        ax.set_visible(False)

# Hide any unused subplots if fewer than 6 categories
for j in range(i + 1, 6):
    fig.delaxes(axes[j])

plt.suptitle('Top 5 Words in Subject per Category', fontsize=28, fontfamily='century gothic')
sns.despine()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 9. Repeat, but for Body
df_train['Cleaned_Body'] = df_train['Body'].fillna('').apply(clean_text)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Get the top 6 categories (or define manually)
top_categories = df_train['Y'].value_counts().index[:6]

for i, category in enumerate(top_categories):
    ax = axes[i]
    # Get all words for this category
    all_words = df_train[df_train['Y'] == category]['Cleaned_Body'].sum()
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(5)

    if top_words:
        words, counts = zip(*top_words)
        sns.barplot(x=list(counts), y=list(words), ax=ax, color=custom_palette.get(category, '#333333'))
        ax.set_title(f'{category}', fontsize=20, fontfamily='century gothic')
        ax.set_xlabel('Frequency', fontsize=14, fontfamily='century gothic')
        ax.set_ylabel('')
        ax.tick_params(labelsize=12)
        ax.set_yticklabels(ax.get_yticklabels(), fontfamily='century gothic')
    else:
        ax.set_visible(False)

# Hide any unused subplots if fewer than 6 categories
for j in range(i + 1, 6):
    fig.delaxes(axes[j])

plt.suptitle('Top 5 Words in Body per Category', fontsize=28, fontfamily='century gothic')
sns.despine()
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
