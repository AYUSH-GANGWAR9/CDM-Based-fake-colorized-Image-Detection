import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def calculate_hter(y_true, y_pred):
    """
    Calculate Half Total Error Rate (HTER).
    """
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    tn, fp, fn, tp = confusion_matrix(true_classes, pred_classes).ravel()
    fpr = fp / (tn + fp + 1e-8)
    fnr = fn / (tp + fn + 1e-8)
    hter = (fpr + fnr) / 2
    return hter, fpr, fnr

def evaluate_model(model, test_gen, test_steps):
    """
    Evaluate the detection model with various metrics.
    Gather predictions from all batches for consistency.
    """
    all_y_true = []
    all_y_pred = []
    all_y_pred_proba = []
    
    for _ in range(test_steps):
        x_batch, y_batch = next(test_gen())
        pred_batch = model.predict(x_batch)
        all_y_true.extend(np.argmax(y_batch, axis=1))
        all_y_pred.extend(np.argmax(pred_batch, axis=1))
        all_y_pred_proba.extend(pred_batch[:, 1])
    
    accuracy = accuracy_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred)
    recall = recall_score(all_y_true, all_y_pred)
    f1 = f1_score(all_y_true, all_y_pred)
    auc = roc_auc_score(np.array(all_y_true), np.array(all_y_pred_proba))
    tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    hter = (fpr + fnr) / 2
    
    print("Model Evaluation Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc:.4f}")
    print(f"  HTER:      {hter:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'hter': hter,
        'fpr': fpr,
        'fnr': fnr
    }

def predict_image(model, image_path, img_size=(256,256)):
    """
    Predict whether an image is real or fake colorized.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]
    result = "Real" if class_idx == 0 else "Fake Colorized"
    print(f"Image: {image_path}")
    print(f"Prediction: {result} with {confidence:.2f} confidence")
    return result, confidence
