# src/models.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from .cdm_utils import create_cdm

def scale_block(inputs, filters):
    """
    Scale Block for feature correlation.
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
    if inputs.shape[-1] != filters:
        inputs = layers.Conv2D(filters, (1, 1), padding='same')(inputs)
    x = layers.Add()([x, inputs])
    x = layers.LeakyReLU(alpha=0.2)(x)
    return x

def create_encoder_path(inputs, name_prefix):
    """
    Create a parallel encoder path.
    """
    filters = 16
    e1 = layers.Conv2D(filters, (3, 3), padding='same', name=f"{name_prefix}_conv1")(inputs)
    e1 = layers.BatchNormalization(name=f"{name_prefix}_bn1")(e1)
    e1 = layers.LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu1")(e1)
    e1 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool1")(e1)
    
    e2 = layers.Conv2D(filters*2, (3, 3), padding='same', name=f"{name_prefix}_conv2")(e1)
    e2 = layers.BatchNormalization(name=f"{name_prefix}_bn2")(e2)
    e2 = layers.LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu2")(e2)
    e2 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool2")(e2)
    
    e3 = layers.Conv2D(filters*4, (3, 3), padding='same', name=f"{name_prefix}_conv3")(e2)
    e3 = layers.BatchNormalization(name=f"{name_prefix}_bn3")(e3)
    e3 = layers.LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu3")(e3)
    e3 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool3")(e3)
    
    e4 = layers.Conv2D(filters*8, (3, 3), padding='same', name=f"{name_prefix}_conv4")(e3)
    e4 = layers.BatchNormalization(name=f"{name_prefix}_bn4")(e4)
    e4 = layers.LeakyReLU(alpha=0.2, name=f"{name_prefix}_lrelu4")(e4)
    e4 = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool4")(e4)
    
    return e1, e2, e3, e4

def build_regeneration_network(input_shape=(256,256,3)):
    """
    Build the CDM-based regeneration network (autoencoder).
    """
    inputs = layers.Input(shape=input_shape)
    cdm = layers.Lambda(lambda x: create_cdm(x), output_shape=lambda input_shape: input_shape)(inputs)
    cdm_r_g = layers.Lambda(lambda x: x[:, :, :, 0:1])(cdm)
    cdm_g_b = layers.Lambda(lambda x: x[:, :, :, 1:2])(cdm)
    cdm_r_b = layers.Lambda(lambda x: x[:, :, :, 2:3])(cdm)
    
    e1_rg, e2_rg, e3_rg, e4_rg = create_encoder_path(cdm_r_g, "rg_path")
    e1_gb, e2_gb, e3_gb, e4_gb = create_encoder_path(cdm_g_b, "gb_path")
    e1_rb, e2_rb, e3_rb, e4_rb = create_encoder_path(cdm_r_b, "rb_path")
    
    concat = layers.Concatenate()([e4_rg, e4_gb, e4_rb])
    dense = concat
    base_filters = 128
    for _ in range(5):
        dense = scale_block(dense, base_filters)
    
    d1 = layers.Conv2D(base_filters, (3, 3), padding='same')(dense)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.LeakyReLU(alpha=0.2)(d1)
    d1 = layers.UpSampling2D((2, 2))(d1)
    d1 = layers.Concatenate()([d1, layers.Concatenate()([e3_rg, e3_gb, e3_rb])])
    
    d2 = layers.Conv2D(base_filters//2, (3, 3), padding='same')(d1)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.LeakyReLU(alpha=0.2)(d2)
    d2 = layers.UpSampling2D((2, 2))(d2)
    d2 = layers.Concatenate()([d2, layers.Concatenate()([e2_rg, e2_gb, e2_rb])])
    
    d3 = layers.Conv2D(base_filters//4, (3, 3), padding='same')(d2)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.LeakyReLU(alpha=0.2)(d3)
    d3 = layers.UpSampling2D((2, 2))(d3)
    d3 = layers.Concatenate()([d3, layers.Concatenate()([e1_rg, e1_gb, e1_rb])])
    
    d4 = layers.Conv2D(base_filters//8, (3, 3), padding='same')(d3)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.LeakyReLU(alpha=0.2)(d4)
    d4 = layers.UpSampling2D((2, 2))(d4)
    
    outputs = layers.Conv2D(3, (3, 3), activation='tanh', padding='same')(d4)
    model = models.Model(inputs=inputs, outputs=outputs, name="CDM_Regeneration_Network")
    return model, [e4_rg, e4_gb, e4_rb]

def build_detection_network(input_shape=(256,256,3), encoder_weights=None):
    """
    Build the fake colorized image detection network.
    """
    inputs = layers.Input(shape=input_shape)
    cdm = layers.Lambda(lambda x: create_cdm(x))(inputs)
    cdm_r_g = layers.Lambda(lambda x: x[:, :, :, 0:1])(cdm)
    cdm_g_b = layers.Lambda(lambda x: x[:, :, :, 1:2])(cdm)
    cdm_r_b = layers.Lambda(lambda x: x[:, :, :, 2:3])(cdm)
    
    # Use "det_" prefix to differentiate from regeneration network
    e1_rg, e2_rg, e3_rg, e4_rg = create_encoder_path(cdm_r_g, "det_rg_path")
    e1_gb, e2_gb, e3_gb, e4_gb = create_encoder_path(cdm_g_b, "det_gb_path")
    e1_rb, e2_rb, e3_rb, e4_rb = create_encoder_path(cdm_r_b, "det_rb_path")
    
    concat = layers.Concatenate()([e4_rg, e4_gb, e4_rb])
    res_features = residual_block(concat, 128)
    
    x = layers.Conv2D(64, (3,3), padding='same')(res_features)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Conv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.MaxPooling2D((2,2))(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    
    outputs = layers.Dense(2, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="Fake_Colorized_Image_Detection_Network")
    return model

def transfer_encoder_weights(regen_model, detect_model):
    """
    Transfer weights from the regeneration network's encoder layers to the detection network.
    Mapping assumes regeneration layer names (e.g., "rg_path_conv1") correspond to detection ones (e.g., "det_rg_path_conv1").
    """
    mapping = {
        "rg_path_conv1": "det_rg_path_conv1",
        "rg_path_bn1":   "det_rg_path_bn1",
        "rg_path_conv2": "det_rg_path_conv2",
        "rg_path_bn2":   "det_rg_path_bn2",
        "rg_path_conv3": "det_rg_path_conv3",
        "rg_path_bn3":   "det_rg_path_bn3",
        "rg_path_conv4": "det_rg_path_conv4",
        "rg_path_bn4":   "det_rg_path_bn4",
        "gb_path_conv1": "det_gb_path_conv1",
        "gb_path_bn1":   "det_gb_path_bn1",
        "gb_path_conv2": "det_gb_path_conv2",
        "gb_path_bn2":   "det_gb_path_bn2",
        "gb_path_conv3": "det_gb_path_conv3",
        "gb_path_bn3":   "det_gb_path_bn3",
        "gb_path_conv4": "det_gb_path_conv4",
        "gb_path_bn4":   "det_gb_path_bn4",
        "rb_path_conv1": "det_rb_path_conv1",
        "rb_path_bn1":   "det_rb_path_bn1",
        "rb_path_conv2": "det_rb_path_conv2",
        "rb_path_bn2":   "det_rb_path_bn2",
        "rb_path_conv3": "det_rb_path_conv3",
        "rb_path_bn3":   "det_rb_path_bn3",
        "rb_path_conv4": "det_rb_path_conv4",
        "rb_path_bn4":   "det_rb_path_bn4",
    }
    
    regen_layers = {layer.name: layer for layer in regen_model.layers}
    detect_layers = {layer.name: layer for layer in detect_model.layers}
    
    for regen_name, detect_name in mapping.items():
        if regen_name in regen_layers and detect_name in detect_layers:
            try:
                detect_layers[detect_name].set_weights(regen_layers[regen_name].get_weights())
                print(f"Transferred weights from {regen_name} to {detect_name}")
            except Exception as e:
                print(f"Error transferring {regen_name} to {detect_name}: {e}")
        else:
            print(f"Mapping missing for: {regen_name} or {detect_name}")
    return detect_model

def predict_image(model, image_path, img_size=(256,256)):
    """
    Predict whether an image is real or fake colorized.
    Returns (label, confidence).
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return None, 0
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0]
    class_idx = np.argmax(pred)
    label = "Real" if class_idx == 0 else "Fake Colorized"
    confidence = pred[class_idx]
    return label, confidence
