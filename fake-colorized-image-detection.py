# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers
# from tensorflow.keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt
# import os
# import cv2
# from sklearn.metrics import confusion_matrix
# from tensorflow.keras.applications import VGG16
# import glob
# from tqdm import tqdm

# # Ensure TensorFlow uses GPU if available
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
#     print(f"Using GPU: {physical_devices[0].name}")
# else:
#     print("No GPU found, using CPU")

# # Set random seeds for reproducibility
# np.random.seed(42)
# tf.random.set_seed(42)

# # Define the scale block as described in the paper
# def scale_block(inputs, filters):
#     """
#     Scale Block for feature correlation as described in the paper.
#     """
#     x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     return x

# # Define the residual block for the detection network
# def residual_block(inputs, filters):
#     """
#     Residual block for the detection network.
#     """
#     x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.2)(x)
    
#     x = layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = layers.BatchNormalization()(x)
    
#     # Add the residual connection
#     x = layers.Add()([x, inputs])
#     x = layers.LeakyReLU(alpha=0.2)(x)
    
#     return x

# # Create the Channel Difference Map (CDM)
# def create_cdm(img):
#     """
#     Create Channel Difference Maps from RGB image.
#     Returns concatenated [R-G, G-B, R-B] maps.
#     """
#     # Extract RGB channels
#     if len(img.shape) == 4:  # Handle batch dimension
#         r = img[:, :, :, 0]
#         g = img[:, :, :, 1]
#         b = img[:, :, :, 2]
        
#         # Calculate differences
#         r_g = tf.expand_dims(r - g, axis=-1)
#         g_b = tf.expand_dims(g - b, axis=-1)
#         r_b = tf.expand_dims(r - b, axis=-1)
        
#         # Concatenate the differences
#         return tf.concat([r_g, g_b, r_b], axis=-1)
#     else:  # Single image
#         r = img[:, :, 0]
#         g = img[:, :, 1]
#         b = img[:, :, 2]
        
#         # Calculate differences
#         r_g = np.expand_dims(r - g, axis=-1)
#         g_b = np.expand_dims(g - b, axis=-1)
#         r_b = np.expand_dims(r - b, axis=-1)
        
#         # Concatenate the differences
#         return np.concatenate([r_g, g_b, r_b], axis=-1)

# # Function to create a parallel encoder path as described in the paper
# def create_encoder_path(inputs, name_prefix):
#     """
#     Create a parallel encoder path for CDM processing.
#     Each encoder consists of Conv->BatchNorm->MaxPool.
#     """
#     # Initial number of filters as mentioned in paper (f=16)
#     filters = 16
    
#     # First encoder layer
#     e1 = layers.Conv2D(filters, (3, 3), padding='same', name=f"{name_prefix}_conv1")(inputs)
#     e1 = layers.BatchNormalization(name=f"{name_prefix}_bn1")(e1)
#     e1 = layers.LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu1")(e1)
#     e1 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool1")(e1)
    
#     # Second encoder layer (2*filters)
#     e2 = layers.Conv2D(filters*2, (3, 3), padding='same', name=f"{name_prefix}_conv2")(e1)
#     e2 = layers.BatchNormalization(name=f"{name_prefix}_bn2")(e2)
#     e2 = layers.LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu2")(e2)
#     e2 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool2")(e2)
    
#     # Third encoder layer (4*filters)
#     e3 = layers.Conv2D(filters*4, (3, 3), padding='same', name=f"{name_prefix}_conv3")(e2)
#     e3 = layers.BatchNormalization(name=f"{name_prefix}_bn3")(e3)
#     e3 = layers.LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu3")(e3)
#     e3 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool3")(e3)
    
#     # Fourth encoder layer (8*filters)
#     e4 = layers.Conv2D(filters*8, (3, 3), padding='same', name=f"{name_prefix}_conv4")(e3)
#     e4 = layers.BatchNormalization(name=f"{name_prefix}_bn4")(e4)
#     e4 = layers.LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu4")(e4)
#     e4 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool4")(e4)
    
#     return e1, e2, e3, e4

# # Function to create the regeneration network (autoencoder)
# def build_regeneration_network(input_shape=(256, 256, 3)):
#     """
#     Build the CDM-based regeneration network (autoencoder).
#     """
#     # Input layer for the original RGB image
#     inputs = layers.Input(shape=input_shape)
    
#     # Create the CDM from the input image (this is a Lambda layer that applies our function)
#     cdm = layers.Lambda(lambda x: create_cdm(x))(inputs)
    
#     # Split the CDM into three channels
#     cdm_r_g = layers.Lambda(lambda x: x[:, :, :, 0:1])(cdm)
#     cdm_g_b = layers.Lambda(lambda x: x[:, :, :, 1:2])(cdm)
#     cdm_r_b = layers.Lambda(lambda x: x[:, :, :, 2:3])(cdm)
    
#     # Create parallel encoder paths for each CDM channel
#     e1_rg, e2_rg, e3_rg, e4_rg = create_encoder_path(cdm_r_g, "rg_path")
#     e1_gb, e2_gb, e3_gb, e4_gb = create_encoder_path(cdm_g_b, "gb_path")
#     e1_rb, e2_rb, e3_rb, e4_rb = create_encoder_path(cdm_r_b, "rb_path")
    
#     # Concatenate the features from all three paths
#     concat = layers.Concatenate()([e4_rg, e4_gb, e4_rb])
    
#     # Dense module with 5 scale blocks as per paper
#     # The dense module helps correlate color and edge information
#     dense = concat
#     base_filters = 128  # Starting with 8*16 as per the paper's encoder structure
#     for i in range(5):  # 5 scale blocks as described in the paper
#         dense = scale_block(dense, base_filters)
    
#     # Decoder path (mirror of encoder with upsampling)
#     # First decoder layer
#     d1 = layers.Conv2D(base_filters, (3, 3), padding='same')(dense)
#     d1 = layers.BatchNormalization()(d1)
#     d1 = layers.LeakyReLU(alpha=0.2)(d1)
#     d1 = layers.UpSampling2D((2, 2))(d1)
    
#     # Concatenate with encoder features (skip connections)
#     d1 = layers.Concatenate()([d1, layers.Concatenate()([e3_rg, e3_gb, e3_rb])])
    
#     # Second decoder layer
#     d2 = layers.Conv2D(base_filters//2, (3, 3), padding='same')(d1)
#     d2 = layers.BatchNormalization()(d2)
#     d2 = layers.LeakyReLU(alpha=0.2)(d2)
#     d2 = layers.UpSampling2D((2, 2))(d2)
    
#     # Concatenate with encoder features
#     d2 = layers.Concatenate()([d2, layers.Concatenate()([e2_rg, e2_gb, e2_rb])])
    
#     # Third decoder layer
#     d3 = layers.Conv2D(base_filters//4, (3, 3), padding='same')(d2)
#     d3 = layers.BatchNormalization()(d3)
#     d3 = layers.LeakyReLU(alpha=0.2)(d3)
#     d3 = layers.UpSampling2D((2, 2))(d3)
    
#     # Concatenate with encoder features
#     d3 = layers.Concatenate()([d3, layers.Concatenate()([e1_rg, e1_gb, e1_rb])])
    
#     # Fourth decoder layer
#     d4 = layers.Conv2D(base_filters//8, (3, 3), padding='same')(d3)
#     d4 = layers.BatchNormalization()(d4)
#     d4 = layers.LeakyReLU(alpha=0.2)(d4)
#     d4 = layers.UpSampling2D((2, 2))(d4)
    
#     # Final output layer to reconstruct the image
#     outputs = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(d4)
    
#     # Create the model
#     model = models.Model(inputs=inputs, outputs=outputs, name="CDM_Regeneration_Network")
    
#     # Return the model and the encoder outputs for later use in detection network
#     return model, [e4_rg, e4_gb, e4_rb]

# # Function to build the detection network using trained encoder weights
# def build_detection_network(input_shape=(256, 256, 3), encoder_weights=None):
#     """
#     Build the fake colorized image detection network with transfer learning.
#     """
#     # Input layer for the original RGB image
#     inputs = layers.Input(shape=input_shape)
    
#     # Create the CDM from the input image
#     cdm = layers.Lambda(lambda x: create_cdm(x))(inputs)
    
#     # Split the CDM into three channels
#     cdm_r_g = layers.Lambda(lambda x: x[:, :, :, 0:1])(cdm)
#     cdm_g_b = layers.Lambda(lambda x: x[:, :, :, 1:2])(cdm)
#     cdm_r_b = layers.Lambda(lambda x: x[:, :, :, 2:3])(cdm)
    
#     # Create parallel encoder paths for each CDM channel (same as regeneration network)
#     e1_rg, e2_rg, e3_rg, e4_rg = create_encoder_path(cdm_r_g, "det_rg_path")
#     e1_gb, e2_gb, e3_gb, e4_gb = create_encoder_path(cdm_g_b, "det_gb_path")
#     e1_rb, e2_rb, e3_rb, e4_rb = create_encoder_path(cdm_r_b, "det_rb_path")
    
#     # Concatenate the features from all three paths
#     concat = layers.Concatenate()([e4_rg, e4_gb, e4_rb])
    
#     # Apply residual block for correlated features
#     res_features = residual_block(concat, 128)
    
#     # Additional encoding layers for classification
#     x = layers.Conv2D(64, (3, 3), padding='same')(res_features)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     x = layers.MaxPooling2D((2, 2))(x)
    
#     x = layers.Conv2D(32, (3, 3), padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     x = layers.MaxPooling2D((2, 2))(x)
    
#     # Flatten the features
#     x = layers.Flatten()(x)
    
#     # Dense layers for classification
#     x = layers.Dense(256)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.2)(x)
#     x = layers.Dropout(0.5)(x)
    
#     x = layers.Dense(64)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.LeakyReLU(alpha=0.2)(x)
    
#     # Output layer with 2 neurons for binary classification (real/fake)
#     outputs = layers.Dense(2, activation='softmax')(x)
    
#     # Create the model
#     model = models.Model(inputs=inputs, outputs=outputs, name="Fake_Colorized_Image_Detection_Network")
    
#     # If encoder weights are provided, apply transfer learning
#     if encoder_weights is not None:
#         # Here you would need to write logic to map the encoder weights from the 
#         # regeneration network to the detection network's encoder layers
#         print("Applying transfer learning with encoder weights...")
#         # This is a placeholder for the actual weight transfer code
    
#     return model

# # Define function to calculate HTER (Half Total Error Rate) as per the paper
# def calculate_hter(y_true, y_pred):
#     """
#     Calculate Half Total Error Rate (HTER).
#     """
#     pred_classes = np.argmax(y_pred, axis=1)
#     true_classes = np.argmax(y_true, axis=1)
    
#     # Compute confusion matrix
#     tn, fp, fn, tp = confusion_matrix(true_classes, pred_classes).ravel()
    
#     # Handle zero division cases
#     fpr = fp / (tn + fp + 1e-8)  # Add epsilon to avoid division by zero
#     fnr = fn / (tp + fn + 1e-8)
    
#     hter = (fpr + fnr) / 2  # Average of FPR and FNR
    
#     return hter, fpr, fnr

# # Function to load and preprocess images
# def process_image(path, target_size):
#     img = cv2.imread(path)
#     if img is None:
#         print(f"Warning: Could not load image {path}")
#         return None
    
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Ensure 3-channel image (handle grayscale images)
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

#     img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
#     img = img.astype(np.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
    
#     return img

# def load_images(image_paths, target_size=(256, 256)):
#     """
#     Load and preprocess a batch of images efficiently.
#     """
#     images = []
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         results = executor.map(lambda path: process_image(path, target_size), image_paths)
    
#     for img in results:
#         if img is not None:
#             images.append(img)
    
#     return np.array(images)

# # Function to create data generators
# def create_data_generators(real_image_paths, fake_image_paths, batch_size=16, train_ratio=0.8):
#     """
#     Create data generators for training and validation.
#     """
#     np.random.shuffle(real_image_paths)
#     np.random.shuffle(fake_image_paths)
    
#     n_real_train = int(len(real_image_paths) * train_ratio)
#     n_fake_train = int(len(fake_image_paths) * train_ratio)
    
#     real_train_paths, real_val_paths = real_image_paths[:n_real_train], real_image_paths[n_real_train:]
#     fake_train_paths, fake_val_paths = fake_image_paths[:n_fake_train], fake_image_paths[n_fake_train:]

#     def generate_batches(real_paths, fake_paths, batch_size, is_training=True):
#         all_paths = real_paths + fake_paths
#         all_labels = [0] * len(real_paths) + [1] * len(fake_paths)  # 0: real, 1: fake
        
#         indices = np.arange(len(all_paths))
#         np.random.shuffle(indices)
        
#         shuffled_paths = [all_paths[i] for i in indices]
#         shuffled_labels = [all_labels[i] for i in indices]

#         num_batches = int(np.ceil(len(shuffled_paths) / batch_size))  # Fix for small datasets
        
#         for i in range(num_batches):
#             batch_paths = shuffled_paths[i * batch_size:(i + 1) * batch_size]
#             batch_labels = shuffled_labels[i * batch_size:(i + 1) * batch_size]
            
#             batch_images = load_images_parallel(batch_paths)

#             batch_labels_onehot = tf.keras.utils.to_categorical(batch_labels, num_classes=2)

#             if is_training:
#                 yield batch_images, batch_images
#             else:
#                 yield batch_images, batch_labels_onehot

#     train_gen = lambda: generate_batches(real_train_paths, fake_train_paths, batch_size, True)
#     val_gen = lambda: generate_batches(real_val_paths, fake_val_paths, batch_size, True)
    
#     train_det_gen = lambda: generate_batches(real_train_paths, fake_train_paths, batch_size, False)
#     val_det_gen = lambda: generate_batches(real_val_paths, fake_val_paths, batch_size, False)
    
#     train_steps = int(np.ceil((len(real_train_paths) + len(fake_train_paths)) / batch_size))
#     val_steps = int(np.ceil((len(real_val_paths) + len(fake_val_paths)) / batch_size))

#     return train_gen, val_gen, train_det_gen, val_det_gen, train_steps, val_steps
# # Function to train the regeneration network
# def train_regeneration_network(model, train_gen, val_gen, train_steps, val_steps, epochs=100, batch_size=16):
#     """
#     Train the regeneration network.
#     """
#     # Compile the model with MSE loss as mentioned in the paper
#     model.compile(optimizer=optimizers.SGD(learning_rate=0.01), 
#                  loss='mse')
    
#     # Create early stopping callback
#     early_stopping = EarlyStopping(monitor='val_loss', 
#                                    patience=10, 
#                                    restore_best_weights=True)
    
#     # Train the model
#     history = model.fit(
#         train_gen(),
#         steps_per_epoch=train_steps,
#         validation_data=val_gen(),
#         validation_steps=val_steps,
#         epochs=epochs,
#         callbacks=[early_stopping]
#     )
    
#     return history, model

# # Function to train the detection network
# def train_detection_network(model, train_gen, val_gen, train_steps, val_steps, epochs=50, batch_size=16):
#     """
#     Train the detection network.
#     """
#     # Compile the model with MSE loss as mentioned in the paper
#     model.compile(optimizer=optimizers.SGD(learning_rate=0.01), 
#                  loss='mse',
#                  metrics=['accuracy'])
    
#     # Create early stopping callback monitoring validation accuracy
#     early_stopping = EarlyStopping(monitor='val_accuracy', 
#                                    patience=10, 
#                                    restore_best_weights=True)
    
#     # Train the model
#     history = model.fit(
#         train_gen(),
#         steps_per_epoch=train_steps,
#         validation_data=val_gen(),
#         validation_steps=val_steps,
#         epochs=epochs,
#         callbacks=[early_stopping]
#     )
    
#     return history, model

# # Function to transfer encoder weights from regeneration network to detection network
# def transfer_encoder_weights(regen_model, detect_model):
#     """
#     Transfer the encoder weights from the regeneration network to the detection network.
#     """
#     # Get the layer names in both models
#     regen_layers = {layer.name: layer for layer in regen_model.layers}
#     detect_layers = {layer.name: layer for layer in detect_model.layers}
    
#     # Map the encoder layer names (This assumes consistent naming between the models)
#     encoder_mapping = {
#         "rg_path_conv1": "det_rg_path_conv1",
#         "rg_path_bn1": "det_rg_path_bn1",
#         "rg_path_conv2": "det_rg_path_conv2",
#         "rg_path_bn2": "det_rg_path_bn2",
#         "rg_path_conv3": "det_rg_path_conv3",
#         "rg_path_bn3": "det_rg_path_bn3",
#         "rg_path_conv4": "det_rg_path_conv4",
#         "rg_path_bn4": "det_rg_path_bn4",
        
#         "gb_path_conv1": "det_gb_path_conv1",
#         "gb_path_bn1": "det_gb_path_bn1",
#         "gb_path_conv2": "det_gb_path_conv2",
#         "gb_path_bn2": "det_gb_path_bn2",
#         "gb_path_conv3": "det_gb_path_conv3",
#         "gb_path_bn3": "det_gb_path_bn3",
#         "gb_path_conv4": "det_gb_path_conv4",
#         "gb_path_bn4": "det_gb_path_bn4",
        
#         "rb_path_conv1": "det_rb_path_conv1",
#         "rb_path_bn1": "det_rb_path_bn1",
#         "rb_path_conv2": "det_rb_path_conv2",
#         "rb_path_bn2": "det_rb_path_bn2",
#         "rb_path_conv3": "det_rb_path_conv3",
#         "rb_path_bn3": "det_rb_path_bn3",
#         "rb_path_conv4": "det_rb_path_conv4",
#         "rb_path_bn4": "det_rb_path_bn4",
#     }
    
#     # Transfer weights
#     for regen_name, detect_name in encoder_mapping.items():
#         if regen_name in regen_layers and detect_name in detect_layers:
#             detect_layers[detect_name].set_weights(regen_layers[regen_name].get_weights())
#             print(f"Transferred weights from {regen_name} to {detect_name}")
#         else:
#             print(f"Warning: Could not transfer weights for {regen_name} to {detect_name}")
    
#     return detect_model

# # Main function to run the entire pipeline
# def main(data_path, batch_size=16, img_size=(256, 256), epochs_regen=100, epochs_detect=50):
#     """
#     Main function to run the entire pipeline.
#     """
#     # 1. Load data paths
#     print("Loading data paths...")
#     real_image_paths = glob.glob(os.path.join(data_path, "real", "*.jpg"))
#     fake_image_paths = glob.glob(os.path.join(data_path, "fake", "*.jpg"))
    
#     if len(real_image_paths) == 0 or len(fake_image_paths) == 0:
#         print(f"Error: No images found in {data_path}")
#         return
    
#     print(f"Found {len(real_image_paths)} real images and {len(fake_image_paths)} fake images")
    
#     # 2. Create data generators
#     print("Creating data generators...")
#     train_gen, val_gen, train_det_gen, val_det_gen, train_steps, val_steps = create_data_generators(
#         real_image_paths, fake_image_paths, batch_size=batch_size
#     )
    
#     # 3. Build and train regeneration network
#     print("Building regeneration network...")
#     regen_model, encoder_outputs = build_regeneration_network(input_shape=(*img_size, 3))
#     regen_model.summary()
    
#     print("Training regeneration network...")
#     regen_history, trained_regen_model = train_regeneration_network(
#         regen_model, train_gen, val_gen, train_steps, val_steps, epochs=epochs_regen, batch_size=batch_size
#     )
    
#     # 4. Save the regeneration model
#     trained_regen_model.save("regeneration_model.h5")
#     print("Regeneration model saved as 'regeneration_model.h5'")
    
#     # 5. Build detection network
#     print("Building detection network...")
#     detect_model = build_detection_network(input_shape=(*img_size, 3))
    
#     # 6. Transfer encoder weights from regeneration to detection network
#     print("Transferring encoder weights...")
#     detect_model = transfer_encoder_weights(trained_regen_model, detect_model)
#     detect_model.summary()
    
#     # 7. Train detection network
#     print("Training detection network...")
#     detect_history, trained_detect_model = train_detection_network(
#         detect_model, train_det_gen, val_det_gen, train_steps, val_steps, epochs=epochs_detect, batch_size=batch_size
#     )
    
#     # 8. Save the detection model
#     trained_detect_model.save("detection_model.h5")
#     print("Detection model saved as 'detection_model.h5'")
    
#     # 9. Evaluate the detection model
#     print("Evaluating detection model...")
#     # Create a test generator with a single batch containing all test samples
#     test_gen = lambda: generate_batches(real_image_paths, fake_image_paths, len(real_image_paths) + len(fake_image_paths), False)
    
#     # Get all test data in a single batch
#     test_images, test_labels = next(test_gen())
    
#     # Make predictions
#     predictions = trained_detect_model.predict(test_images)
    
#     # Calculate HTER
#     hter, fpr, fnr = calculate_hter(test_labels, predictions)
#     print(f"HTER: {hter:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}")
    
#     return trained_regen_model, trained_detect_model, hter

# # Function to generate CDM visualization
# def visualize_cdm(image_path, output_path=None):
#     """
#     Visualize the Channel Difference Map for a given image.
#     """
#     # Load image
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     # Create CDM
#     cdm = create_cdm(img)
    
#     # Normalize CDM for visualization
#     cdm_norm = (cdm - cdm.min()) / (cdm.max() - cdm.min())
    
#     # Create subplots
#     fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
#     # Original image
#     axes[0].imshow(img)
#     axes[0].set_title("Original Image")
#     axes[0].axis('off')
    
#     # R-G channel
#     axes[1].imshow(cdm_norm[:, :, 0], cmap='coolwarm')
#     axes[1].set_title("R-G Channel")
#     axes[1].axis('off')
    
#     # G-B channel
#     axes[2].imshow(cdm_norm[:, :, 1], cmap='coolwarm')
#     axes[2].set_title("G-B Channel")
#     axes[2].axis('off')
    
#     # R-B channel
#     axes[3].imshow(cdm_norm[:, :, 2], cmap='coolwarm')
#     axes[3].set_title("R-B Channel")
#     axes[3].axis('off')
    
#     plt.tight_layout()
    
#     if output_path:
#         plt.savefig(output_path)
#         print(f"CDM visualization saved to {output_path}")
    
#     plt.show()
    
#     return cdm

# # Function to test the model on a single image
# def predict_image(model, image_path, img_size=(256, 256)):
#     """
#     Predict whether an image is real or fake colorized.
#     """
#     # Load and preprocess the image
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, img_size)
#     img = img.astype(np.float32) / 127.5 - 1.0
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
    
#     # Make prediction
#     prediction = model.predict(img)[0]
    
#     # Interpret prediction
#     class_idx = np.argmax(prediction)
#     confidence = prediction[class_idx]
    
#     result = "Real" if class_idx == 0 else "Fake Colorized"
    
#     print(f"Image: {image_path}")
#     print(f"Image: {image_path}")
#     print(f"Prediction: {result} with {confidence:.2f} confidence")

    
#     return result, confidence

# # If this script is run directly
# if __name__ == "__main__":
#     import argparse
    
#     # Create argument parser
#     parser = argparse.ArgumentParser(description="Fake Colorized Image Detection")
#     parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset containing 'real' and 'fake' folders")
#     parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
#     parser.add_argument("--img_size", type=int, default=256, help="Image size (square) for processing")
#     parser.add_argument("--epochs_regen", type=int, default=100, help="Number of epochs for regeneration network")
#     parser.add_argument("--epochs_detect", type=int, default=50, help="Number of epochs for detection network")
#     parser.add_argument("--mode", type=str, choices=["train", "test", "visualize"], default="train", 
#                         help="Mode: 'train' to train models, 'test' to test on single image, 'visualize' to visualize CDM")
#     parser.add_argument("--test_image", type=str, help="Path to test image (used with --mode test)")
#     parser.add_argument("--model_path", type=str, help="Path to saved detection model (used with --mode test)")
    
#     args = parser.parse_args()
    
#     if args.mode == "train":
#         # Run the training pipeline
#         regen_model, detect_model, hter = main(
#             args.data_path, 
#             batch_size=args.batch_size, 
#             img_size=(args.img_size, args.img_size),
#             epochs_regen=args.epochs_regen,
#             epochs_detect=args.epochs_detect
#         )
        
#         print(f"Training complete with HTER: {hter:.4f}")
        
#     elif args.mode == "test":
#         if not args.test_image or not args.model_path:
#             print("Error: --test_image and --model_path are required for test mode")
#             exit(1)
            
#         # Load the detection model
#         detect_model = models.load_model(args.model_path)
        
#         # Run prediction
#         result, confidence = predict_image(
#             detect_model, 
#             args.test_image, 
#             img_size=(args.img_size, args.img_size)
#         )
        
#         print(f"Image classification: {result} with {confidence:.2f} confidence")
        
#     elif args.mode == "visualize":
#         if not args.test_image:
#             print("Error: --test_image is required for visualize mode")
#             exit(1)
            
#         # Generate and display CDM visualization
#         cdm = visualize_cdm(args.test_image, output_path="cdm_visualization.png")
#         print("CDM visualization complete")
        
#     else:
#         print(f"Error: Unknown mode {args.mode}")

# # Function to evaluate model performance with different metrics
# def evaluate_model(model, test_gen, test_steps):
#     """
#     Evaluate the detection model with various metrics.
#     """
#     # Get predictions and true labels
#     y_true = []
#     y_pred = []
    
#     for _ in range(test_steps):
#         x_batch, y_batch = next(test_gen())
#         pred_batch = model.predict(x_batch)
        
#         y_true.extend(np.argmax(y_batch, axis=1))
#         y_pred.extend(np.argmax(pred_batch, axis=1))
    
#     # Calculate metrics
#     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred)
#     recall = recall_score(y_true, y_pred)
#     f1 = f1_score(y_true, y_pred)
    
#     # For ROC AUC, we need probability scores
#     y_true_binary = np.array(y_true)
#     y_pred_proba = model.predict(next(test_gen())[0])[:, 1]  # Probability of fake class
#     auc = roc_auc_score(y_true_binary, y_pred_proba)
    
#     # Calculate HTER
#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
#     fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
#     hter = (fpr + fnr) / 2
    
#     print(f"  Model Evaluation Metrics:")
#     print(f"  Accuracy:  {accuracy:.4f}")
#     print(f"  Precision: {precision:.4f}")
#     print(f"  Recall:    {recall:.4f}")
#     print(f"  F1 Score:  {f1:.4f}")
#     print(f"  ROC AUC:   {auc:.4f}")
#     print(f"  HTER:      {hter:.4f}")
    
#     return {
#         'accuracy': accuracy,
#         'precision': precision,
#         'recall': recall,
#         'f1': f1,
#         'auc': auc,
#         'hter': hter,
#         'fpr': fpr,
#         'fnr': fnr
#     }

# # Function to plot training history
# def plot_training_history(history, model_name, save_path=None):
#     """
#     Plot the training history for a model.
#     """
#     # Create figure with subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
#     # Plot loss
#     ax1.plot(history.history['loss'], label='Training Loss')
#     ax1.plot(history.history['val_loss'], label='Validation Loss')
#     ax1.set_title(f'{model_name} - Loss')
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Loss')
#     ax1.legend()
#     ax1.grid(True)
    
#     # Plot accuracy if available
#     if 'accuracy' in history.history:
#         ax2.plot(history.history['accuracy'], label='Training Accuracy')
#         ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
#         ax2.set_title(f'{model_name} - Accuracy')
#         ax2.set_xlabel('Epoch')
#         ax2.set_ylabel('Accuracy')
#         ax2.legend()
#         ax2.grid(True)
    
#     plt.tight_layout()
    
#     if save_path:
#         plt.savefig(save_path)
#         print(f"Training history plot saved to {save_path}")
    
#     plt.show()

# # Function to generate a confusion matrix visualization
# def plot_confusion_matrix(model, test_gen, class_names=['Real', 'Fake'], save_path=None):
#     """
#     Generate and plot a confusion matrix for the model.
#     """
#     # Get predictions
#     x_test, y_test = next(test_gen())
#     y_pred = model.predict(x_test)
    
#     y_test_classes = np.argmax(y_test, axis=1)
#     y_pred_classes = np.argmax(y_pred, axis=1)
    
#     # Calculate confusion matrix
#     cm = confusion_matrix(y_test_classes, y_pred_classes)
    
#     # Plot
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.title('Confusion Matrix')
    
#     if save_path:
#         plt.savefig(save_path)
#         print(f"Confusion matrix saved to {save_path}")
    
#     plt.show()

# # Function to visualize model activations
# def visualize_activations(model, image_path, layer_name, img_size=(256, 256)):
#     """
#     Visualize activations of a specific layer for a given image.
#     """
#     # Load and preprocess the image
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, img_size)
#     img = img.astype(np.float32) / 127.5 - 1.0
#     img = np.expand_dims(img, axis=0)  # Add batch dimension
    
#     # Create a model that will output the activations of the specified layer
#     activation_model = models.Model(
#         inputs=model.input,
#         outputs=model.get_layer(layer_name).output
#     )
    
#     # Get activations
#     activations = activation_model.predict(img)
    
#     # Plot activations
#     plt.figure(figsize=(15, 8))
    
#     # Show the original image
#     plt.subplot(1, 2, 1)
#     plt.imshow((img[0] + 1.0) / 2.0)  # Convert back to [0,1] range
#     plt.title('Input Image')
#     plt.axis('off')
    
#     # Show the activations (first channel or an average if there are many)
#     plt.subplot(1, 2, 2)
    
#     if len(activations.shape) == 4:  # Conv layer with multiple filters
#         # Average across all filters
#         act_display = np.mean(activations[0], axis=-1)
#         plt.imshow(act_display, cmap='viridis')
#         plt.title(f'Layer: {layer_name} (Average of {activations.shape[-1]} channels)')
#     else:  # Dense layer
#         plt.bar(range(activations.shape[1]), activations[0])
#         plt.title(f'Layer: {layer_name} (Neuron activations)')
    
#     plt.tight_layout()
#     plt.show()
    
#     return activations

# # Function to detect fake images in a folder
# def batch_detection(model, folder_path, output_csv=None, img_size=(256, 256)):
#     """
#     Process all images in a folder and detect fake colorized images.
#     """
#     # Get all image paths
#     image_paths = []
#     for ext in ['jpg', 'jpeg', 'png']:
#         image_paths.extend(glob.glob(os.path.join(folder_path, f'*.{ext}')))
    
#     if not image_paths:
#         print(f"No images found in {folder_path}")
#         return
    
#     print(f"Processing {len(image_paths)} images...")
    
#     # Process each image
#     results = []
#     for path in tqdm(image_paths):
#         result, confidence = predict_image(model, path, img_size)
#         results.append({
#             'image': os.path.basename(path),
#             'prediction': result,
#             'confidence': confidence
#         })
    
#     # Save results to CSV if requested
#     if output_csv:
#         import pandas as pd
#         df = pd.DataFrame(results)
#         df.to_csv(output_csv, index=False)
#         print(f"Results saved to {output_csv}")
    
#     # Print summary
#     fake_count = sum(1 for r in results if r['prediction'] == 'Fake Colorized')
#     real_count = len(results) - fake_count
    
#     print(f"Detection Summary:")
#     print(f"  Total images: {len(results)}")
#     print(f"  Real images: {real_count} ({real_count/len(results)*100:.1f}%)")
#     print(f"  Fake colorized images: {fake_count} ({fake_count/len(results)*100:.1f}%)")
    
#     return results

# # Update the main function to include the new evaluation metrics
# def main(data_path, batch_size=16, img_size=(256, 256), epochs_regen=100, epochs_detect=50):
#     """
#     Main function to run the entire pipeline with enhanced evaluation.
#     """
#     # Original implementation (shortened for brevity)
#     print("Loading data paths...")
#     real_image_paths = glob.glob(os.path.join(data_path, "real", "*.jpg"))
#     fake_image_paths = glob.glob(os.path.join(data_path, "fake", "*.jpg"))
    
#     print("Creating data generators...")
#     train_gen, val_gen, train_det_gen, val_det_gen, train_steps, val_steps = create_data_generators(
#         real_image_paths, fake_image_paths, batch_size=batch_size
#     )
    
#     print("Building and training regeneration network...")
#     regen_model, encoder_outputs = build_regeneration_network(input_shape=(*img_size, 3))
#     regen_history, trained_regen_model = train_regeneration_network(
#         regen_model, train_gen, val_gen, train_steps, val_steps, epochs=epochs_regen, batch_size=batch_size
#     )
    
#     # Plot regeneration network training history
#     plot_training_history(regen_history, "Regeneration Network", save_path="regen_history.png")
    
#     # Save the regeneration model
#     trained_regen_model.save("regeneration_model.h5")
    
#     print("Building and training detection network...")
#     detect_model = build_detection_network(input_shape=(*img_size, 3))
#     detect_model = transfer_encoder_weights(trained_regen_model, detect_model)
#     detect_history, trained_detect_model = train_detection_network(
#         detect_model, train_det_gen, val_det_gen, train_steps, val_steps, epochs=epochs_detect, batch_size=batch_size
#     )
    
#     # Plot detection network training history
#     plot_training_history(detect_history, "Detection Network", save_path="detect_history.png")
    
#     # Save the detection model
#     trained_detect_model.save("detection_model.h5")
    
#     # Comprehensive evaluation
#     print("Performing comprehensive evaluation...")
#     test_gen = lambda: generate_batches(real_image_paths, fake_image_paths, batch_size, False)
#     metrics = evaluate_model(trained_detect_model, test_gen, (len(real_image_paths) + len(fake_image_paths)) // batch_size)
    
#     # Plot confusion matrix
#     plot_confusion_matrix(trained_detect_model, test_gen, save_path="confusion_matrix.png")
    
#     # Plot ROC curve
#     plot_roc_curve(trained_detect_model, test_gen, save_path="roc_curve.png")
    
#     return trained_regen_model, trained_detect_model, metrics


# # Function to plot ROC curve
# def plot_roc_curve(model, test_gen, save_path=None):
#     """
#     Plot ROC curve for the model.
#     """
#     from sklearn.metrics import roc_curve, auc
    
#     # Get test data
#     x_test, y_test = next(test_gen())
#     y_test_binary = np.argmax(y_test, axis=1)
    
#     # Get probability predictions
#     y_pred_proba = model.predict(x_test)[:, 1]  # Probability of fake class
    
#     # Calculate ROC curve
#     fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba)
#     roc_auc = auc(fpr, tpr)
    
#     # Plot ROC curve
#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.grid(True)
    
#     if save_path:
#         plt.savefig(save_path)
#         print(f"ROC curve saved to {save_path}")
    
#     plt.show()