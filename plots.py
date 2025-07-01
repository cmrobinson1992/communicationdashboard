import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, title="Confusion Matrix", cmap="Blues"):
    """
    Plots a pretty confusion matrix.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        class_names: list of class names, in the correct order
        normalize: if True, show proportions instead of counts
        title: title of the plot
        cmap: color map for heatmap
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap,
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=0.5, square=True, cbar=False)

    plt.title(title, fontsize=14)
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_roc(y_true, y_score, class_names, title='ROC-AUC Curve'):
    """
    Plots ROC curves for multiclass classification.
    
    Args:
        y_true: array of true labels
        y_score: array of predicted probabilities (N, C) from classifier.predict_proba
        class_names: list of class names
        title: title for the plot
    """
    # Binarize the output labels for multiclass ROC
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = y_true_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Compute ROC curve and AUC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr["macro"], tpr["macro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot
    plt.figure(figsize=(10, 7))
    colors = cycle(plt.cm.tab10.colors)

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.2f})')

    plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle='--', lw=2,
             label=f'Macro-average (AUC = {roc_auc["macro"]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()