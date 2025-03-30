# src/data_utils.py
import numpy as np
import cv2
import concurrent.futures
import tensorflow as tf

def process_image(path, target_size):
    """
    Load and preprocess a single image.
    """
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not load image {path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 127.5 - 1.0  # Normalize to [-1,1]
    return img

def load_images_parallel(image_paths, target_size=(256,256)):
    images = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda p: process_image(p, target_size), image_paths)
        for img in results:
            if img is not None:
                images.append(img)
    if not images:
        raise ValueError("No valid images were loaded. Check the image paths.")
    return np.array(images)

def generate_batches(real_paths, fake_paths, batch_size, is_training=True):
    """
    Generator yielding batches.
    For autoencoder training: targets are images.
    For classifier training: targets are one-hot labels.
    """
    while True:
        all_paths = real_paths + fake_paths
        all_labels = [0] * len(real_paths) + [1] * len(fake_paths)
        indices = np.arange(len(all_paths))
        np.random.shuffle(indices)
        shuffled_paths = [all_paths[i] for i in indices]
        shuffled_labels = [all_labels[i] for i in indices]
        num_batches = len(shuffled_paths) // batch_size
        for i in range(num_batches):
            batch_paths = shuffled_paths[i*batch_size:(i+1)*batch_size]
            batch_labels = shuffled_labels[i*batch_size:(i+1)*batch_size]
            batch_images = load_images_parallel(batch_paths)
            batch_labels_onehot = tf.keras.utils.to_categorical(batch_labels, num_classes=2)
            if is_training:
                yield batch_images, batch_images  # For autoencoder
            else:
                yield batch_images, batch_labels_onehot

def create_data_generators(real_image_paths, fake_image_paths, batch_size=16, train_ratio=0.8):
    np.random.shuffle(real_image_paths)
    np.random.shuffle(fake_image_paths)
    
    n_real_train = int(len(real_image_paths) * train_ratio)
    n_fake_train = int(len(fake_image_paths) * train_ratio)
    
    real_train = real_image_paths[:n_real_train]
    real_val   = real_image_paths[n_real_train:]
    fake_train = fake_image_paths[:n_fake_train]
    fake_val   = fake_image_paths[n_fake_train:]
    
    train_gen = lambda: generate_batches(real_train, fake_train, batch_size, True)
    val_gen   = lambda: generate_batches(real_val, fake_val, batch_size, True)
    train_det_gen = lambda: generate_batches(real_train, fake_train, batch_size, False)
    val_det_gen   = lambda: generate_batches(real_val, fake_val, batch_size, False)
    
    train_steps = int(np.ceil((len(real_train)+len(fake_train)) / batch_size))
    val_steps   = int(np.ceil((len(real_val)+len(fake_val)) / batch_size))
    
    return train_gen, val_gen, train_det_gen, val_det_gen, train_steps, val_steps
