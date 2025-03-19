#src/cdm_utils.py
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# Create the Channel Difference Map (CDM)
def create_cdm(img):
    """
    Create Channel Difference Maps from RGB image.
    Returns concatenated [R-G, G-B, R-B] maps.
    """
    # Extract RGB channels
    if len(img.shape) == 4:  # Handle batch dimension
        r = img[:, :, :, 0]
        g = img[:, :, :, 1]
        b = img[:, :, :, 2]
        
        # Calculate differences
        r_g = tf.expand_dims(r - g, axis=-1)
        g_b = tf.expand_dims(g - b, axis=-1)
        r_b = tf.expand_dims(r - b, axis=-1)
        
        # Concatenate the differences
        return tf.concat([r_g, g_b, r_b], axis=-1)
    else:  # Single image
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        
        # Calculate differences
        r_g = np.expand_dims(r - g, axis=-1)
        g_b = np.expand_dims(g - b, axis=-1)
        r_b = np.expand_dims(r - b, axis=-1)
        
        # Concatenate the differences
        return np.concatenate([r_g, g_b, r_b], axis=-1)
    
# Function to generate CDM visualization
def visualize_cdm(image_path, output_path=None):
    """
    Visualize the Channel Difference Map for a given image.
    """
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create CDM
    cdm = create_cdm(img)
    
    # Normalize CDM for visualization
    cdm_norm = (cdm - cdm.min()) / (cdm.max() - cdm.min())
    
    # Create subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # R-G channel
    axes[1].imshow(cdm_norm[:, :, 0], cmap='coolwarm')
    axes[1].set_title("R-G Channel")
    axes[1].axis('off')
    
    # G-B channel
    axes[2].imshow(cdm_norm[:, :, 1], cmap='coolwarm')
    axes[2].set_title("G-B Channel")
    axes[2].axis('off')
    
    # R-B channel
    axes[3].imshow(cdm_norm[:, :, 2], cmap='coolwarm')
    axes[3].set_title("R-B Channel")
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        print(f"CDM visualization saved to {output_path}")
    
    plt.show()
    
    return cdm
