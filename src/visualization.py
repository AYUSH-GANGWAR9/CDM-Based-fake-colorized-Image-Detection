# src/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
import glob
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tensorflow.keras import models

def plot_training_history(history, model_name, save_path=None):
    """
    Plot the training history.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    if 'accuracy' in history.history:
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title(f'{model_name} - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    plt.show()

def plot_confusion_matrix(model, test_gen, class_names=['Real', 'Fake'], save_path=None):
    """
    Plot confusion matrix.
    """
    x_test, y_test = next(test_gen())
    y_pred = model.predict(x_test)
    y_test_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_roc_curve(model, test_gen, save_path=None):
    """
    Plot ROC curve.
    """
    x_test, y_test = next(test_gen())
    y_test_binary = np.argmax(y_test, axis=1)
    y_pred_proba = model.predict(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    plt.show()

def visualize_activations(model, image_path, layer_name, img_size=(256,256)):
    """
    Visualize activations of a specified layer.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 127.5 - 1.0
    img_expanded = np.expand_dims(img, axis=0)
    
    activation_model = models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    activations = activation_model.predict(img_expanded)
    
    plt.figure(figsize=(15, 8))
    plt.subplot(1,2,1)
    plt.imshow((img + 1.0)/2.0)
    plt.title('Input Image')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    if len(activations.shape) == 4:
        act_display = np.mean(activations[0], axis=-1)
        plt.imshow(act_display, cmap='viridis')
        plt.title(f'{layer_name} (Average over channels)')
    else:
        plt.bar(range(activations.shape[1]), activations[0])
        plt.title(f'{layer_name} activations')
    plt.tight_layout()
    plt.show()
    return activations

def batch_detection(model, folder_path, output_csv=None, img_size=(256,256)):
    """
    Detect fake images in a folder.
    """
    import pandas as pd
    from src.models import predict_image
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_paths.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
    if not image_paths:
        print(f"No images found in {folder_path}")
        return
    print(f"Processing {len(image_paths)} images...")
    results = []
    from tqdm import tqdm
    for path in tqdm(image_paths):
        result, confidence = predict_image(model, path, img_size)
        results.append({'image': os.path.basename(path), 'prediction': result, 'confidence': confidence})
    if output_csv:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    fake_count = sum(1 for r in results if r['prediction'] == 'Fake Colorized')
    real_count = len(results) - fake_count
    print("Detection Summary:")
    print(f"  Total images: {len(results)}")
    print(f"  Real images: {real_count} ({real_count/len(results)*100:.1f}%)")
    print(f"  Fake colorized images: {fake_count} ({fake_count/len(results)*100:.1f}%)")
    return results
