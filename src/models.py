#src/models.py
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import models
# In models.py, add this import at the top
from .cdm_utils import create_cdm
import glob
from tqdm import tqdm
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Define the scale block as described in the paper
def scale_block(inputs, filters):
    """
    Scale Block for feature correlation as described in the paper.
    """
    x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def residual_block(inputs, filters):
    """
    Residual block for the detection network.
    """
    x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Adjust input dimensions if needed
    if inputs.shape[-1] != filters:
        inputs = layers.Conv2D(filters, (1, 1), padding='same')(inputs)

    # Add the residual connection
    x = layers.Add()([x, inputs])
    x = layers.LeakyReLU(alpha=0.2)(x)

    return x

# Function to create a parallel encoder path as described in the paper
def create_encoder_path(inputs, name_prefix):
    """
    Create a parallel encoder path for CDM processing.
    Each encoder consists of Conv->BatchNorm->MaxPool.
    """
    # Initial number of filters as mentioned in paper (f=16)
    filters = 16
    
    # First encoder layer
    e1 = layers.Conv2D(filters, (3, 3), padding='same', name=f"{name_prefix}_conv1")(inputs)
    e1 = layers.BatchNormalization(name=f"{name_prefix}_bn1")(e1)
    e1 = layers.LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu1")(e1)
    e1 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool1")(e1)
    
    # Second encoder layer (2*filters)
    e2 = layers.Conv2D(filters*2, (3, 3), padding='same', name=f"{name_prefix}_conv2")(e1)
    e2 = layers.BatchNormalization(name=f"{name_prefix}_bn2")(e2)
    e2 = layers.LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu2")(e2)
    e2 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool2")(e2)
    
    # Third encoder layer (4*filters)
    e3 = layers.Conv2D(filters*4, (3, 3), padding='same', name=f"{name_prefix}_conv3")(e2)
    e3 = layers.BatchNormalization(name=f"{name_prefix}_bn3")(e3)
    e3 = layers.LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu3")(e3)
    e3 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool3")(e3)
    
    # Fourth encoder layer (8*filters)
    e4 = layers.Conv2D(filters*8, (3, 3), padding='same', name=f"{name_prefix}_conv4")(e3)
    e4 = layers.BatchNormalization(name=f"{name_prefix}_bn4")(e4)
    e4 = layers.LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu4")(e4)
    e4 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool4")(e4)
    
    return e1, e2, e3, e4

# Function to create the regeneration network (autoencoder)
def build_regeneration_network(input_shape=(256, 256, 3)):
    """
    Build the CDM-based regeneration network (autoencoder).
    """
    # Input layer for the original RGB image
    inputs = layers.Input(shape=input_shape)
    
    # Create the CDM from the input image (this is a Lambda layer that applies our function)
    cdm = layers.Lambda(lambda x: create_cdm(x), output_shape=lambda input_shape: input_shape)(inputs)

    
    # Split the CDM into three channels
    cdm_r_g = layers.Lambda(lambda x: x[:, :, :, 0:1])(cdm)
    cdm_g_b = layers.Lambda(lambda x: x[:, :, :, 1:2])(cdm)
    cdm_r_b = layers.Lambda(lambda x: x[:, :, :, 2:3])(cdm)
    
    # Create parallel encoder paths for each CDM channel
    e1_rg, e2_rg, e3_rg, e4_rg = create_encoder_path(cdm_r_g, "rg_path")
    e1_gb, e2_gb, e3_gb, e4_gb = create_encoder_path(cdm_g_b, "gb_path")
    e1_rb, e2_rb, e3_rb, e4_rb = create_encoder_path(cdm_r_b, "rb_path")
    
    # Concatenate the features from all three paths
    concat = layers.Concatenate()([e4_rg, e4_gb, e4_rb])
    
    # Dense module with 5 scale blocks as per paper
    # The dense module helps correlate color and edge information
    dense = concat
    base_filters = 128  # Starting with 8*16 as per the paper's encoder structure
    for i in range(5):  # 5 scale blocks as described in the paper
        dense = scale_block(dense, base_filters)
    
    # Decoder path (mirror of encoder with upsampling)
    # First decoder layer
    d1 = layers.Conv2D(base_filters, (3, 3), padding='same')(dense)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.LeakyReLU(alpha=0.2)(d1)
    d1 = layers.UpSampling2D((2, 2))(d1)
    
    # Concatenate with encoder features (skip connections)
    d1 = layers.Concatenate()([d1, layers.Concatenate()([e3_rg, e3_gb, e3_rb])])
    
    # Second decoder layer
    d2 = layers.Conv2D(base_filters//2, (3, 3), padding='same')(d1)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.LeakyReLU(alpha=0.2)(d2)
    d2 = layers.UpSampling2D((2, 2))(d2)
    
    # Concatenate with encoder features
    d2 = layers.Concatenate()([d2, layers.Concatenate()([e2_rg, e2_gb, e2_rb])])
    
    # Third decoder layer
    d3 = layers.Conv2D(base_filters//4, (3, 3), padding='same')(d2)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.LeakyReLU(alpha=0.2)(d3)
    d3 = layers.UpSampling2D((2, 2))(d3)
    
    # Concatenate with encoder features
    d3 = layers.Concatenate()([d3, layers.Concatenate()([e1_rg, e1_gb, e1_rb])])
    
    # Fourth decoder layer
    d4 = layers.Conv2D(base_filters//8, (3, 3), padding='same')(d3)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.LeakyReLU(alpha=0.2)(d4)
    d4 = layers.UpSampling2D((2, 2))(d4)
    
    # Final output layer to reconstruct the image
    outputs = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(d4)
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs, name="CDM_Regeneration_Network")
    
    # Return the model and the encoder outputs for later use in detection network
    return model, [e4_rg, e4_gb, e4_rb]

# Function to build the detection network using trained encoder weights
def build_detection_network(input_shape=(256, 256, 3), encoder_weights=None):
    """
    Build the fake colorized image detection network with transfer learning.
    """
    # Input layer for the original RGB image
    inputs = layers.Input(shape=input_shape)
    
    # Create the CDM from the input image
    cdm = layers.Lambda(lambda x: create_cdm(x))(inputs)
    
    # Split the CDM into three channels
    cdm_r_g = layers.Lambda(lambda x: x[:, :, :, 0:1])(cdm)
    cdm_g_b = layers.Lambda(lambda x: x[:, :, :, 1:2])(cdm)
    cdm_r_b = layers.Lambda(lambda x: x[:, :, :, 2:3])(cdm)
    
    # Create parallel encoder paths for each CDM channel (same as regeneration network)
    e1_rg, e2_rg, e3_rg, e4_rg = create_encoder_path(cdm_r_g, "det_rg_path")
    e1_gb, e2_gb, e3_gb, e4_gb = create_encoder_path(cdm_g_b, "det_gb_path")
    e1_rb, e2_rb, e3_rb, e4_rb = create_encoder_path(cdm_r_b, "det_rb_path")
    
    # Concatenate the features from all three paths
    concat = layers.Concatenate()([e4_rg, e4_gb, e4_rb])
    
    # Apply residual block for correlated features
    res_features = residual_block(concat, 128)
    
    # Additional encoding layers for classification
    x = layers.Conv2D(64, (3, 3), padding='same')(res_features)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Flatten the features
    x = layers.Flatten()(x)
    
    # Dense layers for classification
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    # Output layer with 2 neurons for binary classification (real/fake)
    outputs = layers.Dense(2, activation='softmax')(x)
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs, name="Fake_Colorized_Image_Detection_Network")
    
    # If encoder weights are provided, apply transfer learning
    if encoder_weights is not None:
        # Here you would need to write logic to map the encoder weights from the 
        # regeneration network to the detection network's encoder layers
        print("Applying transfer learning with encoder weights...")
        # This is a placeholder for the actual weight transfer code
    
    return model
