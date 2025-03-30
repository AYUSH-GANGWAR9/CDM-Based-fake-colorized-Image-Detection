import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

def plot_training_history(history, model_name, save_path=None):
    """
    Plot training history for a model.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.history["loss"], label="Training Loss")
    ax1.plot(history.history["val_loss"], label="Validation Loss")
    ax1.set_title(f"{model_name} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)
    
    if "accuracy" in history.history:
        ax2.plot(history.history["accuracy"], label="Training Accuracy")
        ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
        ax2.set_title(f"{model_name} - Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Training history saved to {save_path}")
    plt.show()

def plot_confusion_matrix(model, test_gen, class_names=["Real", "Fake"], save_path=None):
    """
    Plot a confusion matrix for the model.
    """
    x_test, y_test = next(test_gen())
    y_pred = model.predict(x_test)
    y_test_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_roc_curve(model, test_gen, save_path=None):
    """
    Plot ROC curve for the model.
    """
    x_test, y_test = next(test_gen())
    y_test_binary = np.argmax(y_test, axis=1)
    y_pred_proba = model.predict(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    plt.show()
