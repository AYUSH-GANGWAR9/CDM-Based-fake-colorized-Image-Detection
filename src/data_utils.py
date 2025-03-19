#src/data_utils.py
import numpy as np
import tensorflow as tf
import cv2
import concurrent.futures

# Function to load and preprocess individual images
def process_image(path, target_size):
    """
    Loads and preprocesses a single image for deep learning.
    
    Args:
        path: File path to the image
        target_size: Tuple of (width, height) for resizing
        
    Returns:
        Preprocessed image as numpy array or None if loading fails
    """
    # Load image using OpenCV
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Could not load image {path}")
        return None
    
    # Convert from BGR (OpenCV default) to RGB (used by most ML frameworks)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Ensure 3-channel image by converting grayscale to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Resize to target dimensions - INTER_AREA provides better quality for downsampling
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to [-1, 1] range
    # First divide by 127.5 to get [0, 2] range, then subtract 1
    # This normalization works well for models like GANs and certain CNN architectures
    img = img.astype(np.float32) / 127.5 - 1.0
    
    return img

def load_images_parallel(image_paths, target_size=(256, 256)):
    """
    Load and preprocess a batch of images using parallel processing for efficiency.
    
    Args:
        image_paths: List of file paths to images
        target_size: Tuple of (width, height) for resizing, default (256, 256)
        
    Returns:
        Numpy array of preprocessed images
    """
    images = []
    
    # Use ThreadPoolExecutor for parallel I/O operations (more efficient than ProcessPoolExecutor for I/O)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map executes the function for each path in parallel
        results = executor.map(lambda path: process_image(path, target_size), image_paths)
        
        # Filter out any None results (failed loads)
        for img in results:
            if img is not None:
                images.append(img)
    
    # Check if we have any valid images
    if not images:
        raise ValueError("No valid images were loaded. Please check the image paths.")
        
    return np.array(images)

# Function to create data generators for training and validation
def create_data_generators(real_image_paths, fake_image_paths, batch_size=16, train_ratio=0.8):
    """
    Create data generators for training and validation splits.
    
    Args:
        real_image_paths: List of file paths to real images
        fake_image_paths: List of file paths to fake/generated images
        batch_size: Number of images per batch
        train_ratio: Fraction of data to use for training (vs validation)
        
    Returns:
        Tuple of (train_generator, val_generator, train_detector_generator, 
                val_detector_generator, train_steps_per_epoch, val_steps_per_epoch)
    """
    # Shuffle paths for randomization
    np.random.shuffle(real_image_paths)
    np.random.shuffle(fake_image_paths)
    
    # Split data into training and validation sets
    n_real_train = int(len(real_image_paths) * train_ratio)
    n_fake_train = int(len(fake_image_paths) * train_ratio)
    
    real_train_paths = real_image_paths[:n_real_train]
    real_val_paths = real_image_paths[n_real_train:]
    
    fake_train_paths = fake_image_paths[:n_fake_train]
    fake_val_paths = fake_image_paths[n_fake_train:]
    
    def generate_batches(real_paths, fake_paths, batch_size, is_training=True):
        """
        Generator function that yields batches of images indefinitely.
        
        Args:
            real_paths: List of file paths to real images
            fake_paths: List of file paths to fake images
            batch_size: Number of images per batch
            is_training: If True, yields (images, images) for autoencoder training
                         If False, yields (images, labels) for classifier training
        
        Yields:
            Tuples of (batch_images, batch_targets) where targets are either
            the same images (autoencoder) or one-hot encoded labels (classifier)
        """
        print(f"Generator initialized with {len(real_paths)} real and {len(fake_paths)} fake images")
        
        # Infinite loop to keep generating batches (required for tf.keras.Model.fit)
        while True:
            # Combine real and fake image paths
            all_paths = real_paths + fake_paths
            
            # Create labels: 0 for real images, 1 for fake images
            all_labels = [0] * len(real_paths) + [1] * len(fake_paths)
            
            # Shuffle data while keeping paths and labels in sync
            indices = np.arange(len(all_paths))
            np.random.shuffle(indices)
            shuffled_paths = [all_paths[i] for i in indices]
            shuffled_labels = [all_labels[i] for i in indices]
            
            # Calculate number of complete batches
            num_batches = len(shuffled_paths) // batch_size
            
            # Generate batches
            for i in range(num_batches):
                # Get paths and labels for current batch
                batch_paths = shuffled_paths[i * batch_size:(i + 1) * batch_size]
                batch_labels = shuffled_labels[i * batch_size:(i + 1) * batch_size]
                
                # Load images using parallel processing
                batch_images = load_images_parallel(batch_paths)
                
                # Convert integer labels to one-hot encoded vectors
                batch_labels_onehot = tf.keras.utils.to_categorical(batch_labels, num_classes=2)
                
                if is_training:
                    # For autoencoder training: input = output
                    yield batch_images, batch_images
                else:
                    # For classifier training: input = images, output = labels
                    yield batch_images, batch_labels_onehot
    
    # Create generator functions for different purposes
    # Lambda functions allow for lazy evaluation
    train_gen = lambda: generate_batches(real_train_paths, fake_train_paths, batch_size, True)
    val_gen = lambda: generate_batches(real_val_paths, fake_val_paths, batch_size, True)
    
    # Detector (classifier) generators
    train_det_gen = lambda: generate_batches(real_train_paths, fake_train_paths, batch_size, False)
    val_det_gen = lambda: generate_batches(real_val_paths, fake_val_paths, batch_size, False)
    
    # Calculate steps per epoch (number of batches)
    train_steps = int(np.ceil((len(real_train_paths) + len(fake_train_paths)) / batch_size))
    val_steps = int(np.ceil((len(real_val_paths) + len(fake_val_paths)) / batch_size))
    
    return train_gen, val_gen, train_det_gen, val_det_gen, train_steps, val_steps

# Example usage:
# real_paths = ['/path/to/real/img1.jpg', '/path/to/real/img2.jpg', ...]
# fake_paths = ['/path/to/fake/img1.jpg', '/path/to/fake/img2.jpg', ...]
# train_gen, val_gen, train_det_gen, val_det_gen, train_steps, val_steps = create_data_generators(real_paths, fake_paths)
# 
# # Create a tf.data.Dataset for the autoencoder
# train_dataset = tf.data.Dataset.from_generator(
#     train_gen,
#     output_types=(tf.float32, tf.float32),
#     output_shapes=((None, 256, 256, 3), (None, 256, 256, 3))
# )
# 
# # Create a tf.data.Dataset for the classifier
# train_det_dataset = tf.data.Dataset.from_generator(
#     train_det_gen,
#     output_types=(tf.float32, tf.float32),
#     output_shapes=((None, 256, 256, 3), (None, 2))
# )