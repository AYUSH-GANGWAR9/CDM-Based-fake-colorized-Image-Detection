import numpy as np
import tensorflow as tf
import cv2
import concurrent.futures

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
    img = img.astype(np.float32) / 127.5 - 1.0
    return img

def load_images_parallel(image_paths, target_size=(256, 256)):
    """
    Load and preprocess images in parallel.
    """
    images = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(lambda path: process_image(path, target_size), image_paths)
        for img in results:
            if img is not None:
                images.append(img)
    if not images:
        raise ValueError("No valid images loaded. Check the image paths.")
    return np.array(images)

def create_data_generators(real_image_paths, fake_image_paths, batch_size=16, train_ratio=0.8):
    """
    Create data generators for training and validation splits.
    """
    np.random.shuffle(real_image_paths)
    np.random.shuffle(fake_image_paths)
    n_real_train = int(len(real_image_paths) * train_ratio)
    n_fake_train = int(len(fake_image_paths) * train_ratio)
    real_train_paths = real_image_paths[:n_real_train]
    real_val_paths = real_image_paths[n_real_train:]
    fake_train_paths = fake_image_paths[:n_fake_train]
    fake_val_paths = fake_image_paths[n_fake_train:]
    
    def generate_batches(real_paths, fake_paths, batch_size, is_training=True):
        all_paths = real_paths + fake_paths
        all_labels = [0]*len(real_paths) + [1]*len(fake_paths)
        indices = np.arange(len(all_paths))
        np.random.shuffle(indices)
        shuffled_paths = [all_paths[i] for i in indices]
        shuffled_labels = [all_labels[i] for i in indices]
        num_batches = len(shuffled_paths) // batch_size
        for i in range(num_batches):
            batch_paths = shuffled_paths[i * batch_size:(i+1)*batch_size]
            batch_labels = shuffled_labels[i * batch_size:(i+1)*batch_size]
            batch_images = load_images_parallel(batch_paths)
            batch_labels_onehot = tf.keras.utils.to_categorical(batch_labels, num_classes=2)
            if is_training:
                yield batch_images, batch_images  # For autoencoder training
            else:
                yield batch_images, batch_labels_onehot  # For classifier training
                
    train_gen = lambda: generate_batches(real_train_paths, fake_train_paths, batch_size, True)
    val_gen = lambda: generate_batches(real_val_paths, fake_val_paths, batch_size, True)
    train_det_gen = lambda: generate_batches(real_train_paths, fake_train_paths, batch_size, False)
    val_det_gen = lambda: generate_batches(real_val_paths, fake_val_paths, batch_size, False)
    train_steps = int(np.ceil((len(real_train_paths) + len(fake_train_paths)) / batch_size))
    val_steps = int(np.ceil((len(real_val_paths) + len(fake_val_paths)) / batch_size))
    
    return train_gen, val_gen, train_det_gen, val_det_gen, train_steps, val_steps
