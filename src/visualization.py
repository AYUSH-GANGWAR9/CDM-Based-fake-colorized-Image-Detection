#src/models.py
import numpy as np
import matplotlib.pyplot as plt

# Function to plot ROC curve
def plot_roc_curve(model, test_gen, save_path=None):
    """
    Plot ROC curve for the model.
    """
    from sklearn.metrics import roc_curve, auc
    
    # Get test data
    x_test, y_test = next(test_gen())
    y_test_binary = np.argmax(y_test, axis=1)
    
    # Get probability predictions
    y_pred_proba = model.predict(x_test)[:, 1]  # Probability of fake class
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    
    plt.show()

# Function to visualize model activations
def visualize_activations(model, image_path, layer_name, img_size=(256, 256)):
    """
    Visualize activations of a specific layer for a given image.
    """
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Create a model that will output the activations of the specified layer
    activation_model = models.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    # Get activations
    activations = activation_model.predict(img)
    
    # Plot activations
    plt.figure(figsize=(15, 8))
    
    # Show the original image
    plt.subplot(1, 2, 1)
    plt.imshow((img[0] + 1.0) / 2.0)  # Convert back to [0,1] range
    plt.title('Input Image')
    plt.axis('off')
    
    # Show the activations (first channel or an average if there are many)
    plt.subplot(1, 2, 2)
    
    if len(activations.shape) == 4:  # Conv layer with multiple filters
        # Average across all filters
        act_display = np.mean(activations[0], axis=-1)
        plt.imshow(act_display, cmap='viridis')
        plt.title(f'Layer: {layer_name} (Average of {activations.shape[-1]} channels)')
    else:  # Dense layer
        plt.bar(range(activations.shape[1]), activations[0])
        plt.title(f'Layer: {layer_name} (Neuron activations)')
    
    plt.tight_layout()
    plt.show()
    
    return activations

# Function to plot training history
def plot_training_history(history, model_name, save_path=None):
    """
    Plot the training history for a model.
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{model_name} - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy if available
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

# Function to generate a confusion matrix visualization
def plot_confusion_matrix(model, test_gen, class_names=['Real', 'Fake'], save_path=None):
    """
    Generate and plot a confusion matrix for the model.
    """
    # Get predictions
    x_test, y_test = next(test_gen())
    y_pred = model.predict(x_test)
    
    y_test_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test_classes, y_pred_classes)
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
