import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

def train_regeneration_network(model, train_gen, val_gen, train_steps, val_steps, epochs=50, batch_size=16):
    """
    Train the regeneration network (autoencoder) using MSE loss.
    """
    model.compile(optimizer=optimizers.SGD(learning_rate=0.01), loss="mse")
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(
        train_gen(),
        steps_per_epoch=train_steps,
        validation_data=val_gen(),
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=[early_stopping]
    )
    return history, model

def train_detection_network(model, train_gen, val_gen, train_steps, val_steps, epochs=50, batch_size=16):
    """
    Train the detection network for classification using MSE loss.
    This version is identical to your original code. 
    To help reduce overfitting, consider increasing dropout in your model architecture or
    adding data augmentation in your data generator.
    """
    model.compile(optimizer=optimizers.SGD(learning_rate=0.01), loss="mse", metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
    history = model.fit(
        train_gen(),
        steps_per_epoch=train_steps,
        validation_data=val_gen(),
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=[early_stopping]
    )
    return history, model

def transfer_encoder_weights(regen_model, detect_model):
    """
    Transfer encoder weights from the regeneration network to the detection network.
    """
    regen_layers = {layer.name: layer for layer in regen_model.layers}
    detect_layers = {layer.name: layer for layer in detect_model.layers}
    
    encoder_mapping = {
        "rg_path_conv1": "det_rg_path_conv1",
        "rg_path_bn1": "det_rg_path_bn1",
        "rg_path_conv2": "det_rg_path_conv2",
        "rg_path_bn2": "det_rg_path_bn2",
        "rg_path_conv3": "det_rg_path_conv3",
        "rg_path_bn3": "det_rg_path_bn3",
        "rg_path_conv4": "det_rg_path_conv4",
        "rg_path_bn4": "det_rg_path_bn4",
        "gb_path_conv1": "det_gb_path_conv1",
        "gb_path_bn1": "det_gb_path_bn1",
        "gb_path_conv2": "det_gb_path_conv2",
        "gb_path_bn2": "det_gb_path_bn2",
        "gb_path_conv3": "det_gb_path_conv3",
        "gb_path_bn3": "det_gb_path_bn3",
        "gb_path_conv4": "det_gb_path_conv4",
        "gb_path_bn4": "det_gb_path_bn4",
        "rb_path_conv1": "det_rb_path_conv1",
        "rb_path_bn1": "det_rb_path_bn1",
        "rb_path_conv2": "det_rb_path_conv2",
        "rb_path_bn2": "det_rb_path_bn2",
        "rb_path_conv3": "det_rb_path_conv3",
        "rb_path_bn3": "det_rb_path_bn3",
        "rb_path_conv4": "det_rb_path_conv4",
        "rb_path_bn4": "det_rb_path_bn4",
    }
    
    for regen_name, detect_name in encoder_mapping.items():
        if regen_name in regen_layers and detect_name in detect_layers:
            detect_layers[detect_name].set_weights(regen_layers[regen_name].get_weights())
            print(f"Transferred weights from {regen_name} to {detect_name}")
        else:
            print(f"Warning: Could not transfer weights for {regen_name} -> {detect_name}")
    return detect_model
