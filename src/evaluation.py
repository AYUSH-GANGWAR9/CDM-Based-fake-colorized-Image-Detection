#src/evaluation.py
import numpy as np
import tensorflow as tf
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define function to calculate HTER (Half Total Error Rate) as per the paper
def calculate_hter(y_true, y_pred):
    """
    Calculate Half Total Error Rate (HTER).
    """
    pred_classes = np.argmax(y_pred, axis=1)
    true_classes = np.argmax(y_true, axis=1)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_classes, pred_classes).ravel()
    
    # Handle zero division cases
    fpr = fp / (tn + fp + 1e-8)  # Add epsilon to avoid division by zero
    fnr = fn / (tp + fn + 1e-8)
    
    hter = (fpr + fnr) / 2  # Average of FPR and FNR
    
    return hter, fpr, fnr

# Function to evaluate model performance with different metrics
def evaluate_model(model, test_gen, test_steps):
    """
    Evaluate the detection model with various metrics.
    """
    # Get predictions and true labels
    y_true = []
    y_pred = []
    
    for _ in range(test_steps):
        x_batch, y_batch = next(test_gen())
        pred_batch = model.predict(x_batch)
        
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(pred_batch, axis=1))
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # For ROC AUC, we need probability scores
    y_true_binary = np.array(y_true)
    y_pred_proba = model.predict(next(test_gen())[0])[:, 1]  # Probability of fake class
    auc = roc_auc_score(y_true_binary, y_pred_proba)
    
    # Calculate HTER
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    hter = (fpr + fnr) / 2
    
    print(f"Model Evaluation Metrics:")
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

# Function to detect fake images in a folder
def batch_detection(model, folder_path, output_csv=None, img_size=(256, 256)):
    """
    Process all images in a folder and detect fake colorized images.
    """
    # Get all image paths
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_paths.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
    
    if not image_paths:
        print(f"No images found in {folder_path}")
        return
    
    print(f"Processing {len(image_paths)} images...")
    
    # Process each image
    results = []
    for path in tqdm(image_paths):
        result, confidence = predict_image(model, path, img_size)
        results.append({
            'image': os.path.basename(path),
            'prediction': result,
            'confidence': confidence
        })
    
    # Save results to CSV if requested
    if output_csv:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    
    # Print summary
    fake_count = sum(1 for r in results if r['prediction'] == 'Fake Colorized')
    real_count = len(results) - fake_count
    
    print(f"Detection Summary:")
    print(f"  Total images: {len(results)}")
    print(f"  Real images: {real_count} ({real_count/len(results)*100:.1f}%)")
    print(f"  Fake colorized images: {fake_count} ({fake_count/len(results)*100:.1f}%)")
    
    return results

# Function to test the model on a single image
def predict_image(model, image_path, img_size=(256, 256)):

    """
    Predict whether an image is real or fake colorized.
    """
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img)[0]
    
    # Interpret prediction
    class_idx = np.argmax(prediction)
    confidence = prediction[class_idx]
    
    result = "Real" if class_idx == 0 else "Fake Colorized"
    
    print(f"Image: {image_path}")
    print(f"Prediction: {result} with {confidence:.2f}")
    print(f"Image: {image_path}")

    print(f"Prediction: {result} with {confidence:.2f} confidence")
    
    return result, confidence

# If this script is run directly
if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Fake Colorized Image Detection")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset containing 'real' and 'fake' folders")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=256, help="Image size (square) for processing")
    parser.add_argument("--epochs_regen", type=int, default=100, help="Number of epochs for regeneration network")
    parser.add_argument("--epochs_detect", type=int, default=50, help="Number of epochs for detection network")
    parser.add_argument("--mode", type=str, choices=["train", "test", "visualize"], default="train", 
                        help="Mode: 'train' to train models, 'test' to test on single image, 'visualize' to visualize CDM")
    parser.add_argument("--test_image", type=str, help="Path to test image (used with --mode test)")
    parser.add_argument("--model_path", type=str, help="Path to saved detection model (used with --mode test)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Run the training pipeline
        regen_model, detect_model, hter = main(
            args.data_path, 
            batch_size=args.batch_size, 
            img_size=(args.img_size, args.img_size),
            epochs_regen=args.epochs_regen,
            epochs_detect=args.epochs_detect
        )
        
        print(f"Training complete with HTER: {hter:.4f}")
        
    elif args.mode == "test":
        if not args.test_image or not args.model_path:
            print("Error: --test_image and --model_path are required for test mode")
            exit(1)
            
        # Load the detection model
        detect_model = models.load_model(args.model_path)
        
        # Run prediction
        result, confidence = predict_image(
            detect_model, 
            args.test_image, 
            img_size=(args.img_size, args.img_size)
        )
        
        print(f"Image classification: {result} with {confidence:.2f} confidence")
        
    elif args.mode == "visualize":
        if not args.test_image:
            print("Error: --test_image is required for visualize mode")
            exit(1)
            
        # Generate and display CDM visualization
        cdm = visualize_cdm(args.test_image, output_path="cdm_visualization.png")
        print("CDM visualization complete")
        
    else:
        print(f"Error: Unknown mode {args.mode}")

