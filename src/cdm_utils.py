import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

@tf.keras.utils.register_keras_serializable(package="Custom")
def create_cdm(img):
    """
    Create Channel Difference Maps from an RGB image.
    Returns concatenated [R-G, G-B, R-B] maps.
    """
    if len(img.shape) == 4:  # Batched input
        r = img[:, :, :, 0]
        g = img[:, :, :, 1]
        b = img[:, :, :, 2]
        r_g = tf.expand_dims(r - g, axis=-1)
        g_b = tf.expand_dims(g - b, axis=-1)
        r_b = tf.expand_dims(r - b, axis=-1)
        return tf.concat([r_g, g_b, r_b], axis=-1)
    else:
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        r_g = np.expand_dims(r - g, axis=-1)
        g_b = np.expand_dims(g - b, axis=-1)
        r_b = np.expand_dims(r - b, axis=-1)
        return np.concatenate([r_g, g_b, r_b], axis=-1)

def visualize_cdm(image_path, output_path=None):
    """
    Visualize the Channel Difference Map for a given image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cdm = create_cdm(img)
    cdm_norm = (cdm - cdm.min()) / (cdm.max() - cdm.min())
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(cdm_norm[:, :, 0], cmap="coolwarm")
    axes[1].set_title("R-G Channel")
    axes[1].axis("off")
    
    axes[2].imshow(cdm_norm[:, :, 1], cmap="coolwarm")
    axes[2].set_title("G-B Channel")
    axes[2].axis("off")
    
    axes[3].imshow(cdm_norm[:, :, 2], cmap="coolwarm")
    axes[3].set_title("R-B Channel")
    axes[3].axis("off")
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
        print(f"CDM visualization saved to {output_path}")
    plt.show()
    return cdm
