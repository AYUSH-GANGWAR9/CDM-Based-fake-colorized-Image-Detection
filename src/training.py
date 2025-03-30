# src/training.py
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping

def train_regeneration_network(model, train_gen, val_gen, train_steps, val_steps, epochs=50, batch_size=16):
    """
    Train the regeneration (autoencoder) network.
    """
    model.compile(optimizer=optimizers.SGD(learning_rate=0.01), loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        train_gen(),
        steps_per_epoch=train_steps,
        validation_data=val_gen(),
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=[early_stopping]
    )
    return history, model

def train_detection_network(model, train_gen, val_gen, train_steps, val_steps, epochs=20, batch_size=16):
    """
    Train the detection (classifier) network.
    """
    model.compile(optimizer=optimizers.SGD(learning_rate=0.01), loss='mse', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    history = model.fit(
        train_gen(),
        steps_per_epoch=train_steps,
        validation_data=val_gen(),
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=[early_stopping]
    )
    return history, model
