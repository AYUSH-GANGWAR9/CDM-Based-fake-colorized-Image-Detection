# src/evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_model(model, test_gen, test_steps):
    """
    Evaluate the detection model with various metrics.
    """
    y_true, y_pred = [], []
    for _ in range(test_steps):
        x_batch, y_batch = next(test_gen())
        pred_batch = model.predict(x_batch)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(pred_batch, axis=1))
        
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    x_batch, _ = next(test_gen())
    y_pred_proba = model.predict(x_batch)[:, 1]
    auc_value = roc_auc_score(np.array(y_true[:len(y_pred_proba)]), y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    hter = (fpr + fnr) / 2
    
    print("Model Evaluation Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  ROC AUC:   {auc_value:.4f}")
    print(f"  HTER:      {hter:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_value,
        'hter': hter,
        'fpr': fpr,
        'fnr': fnr
    }
