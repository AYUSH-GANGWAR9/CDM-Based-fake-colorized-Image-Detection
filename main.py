# main.py
import numpy as np
import tensorflow as tf
import os
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import argparse

# Import functions from our modules
from src.data_utils import create_data_generators, generate_batches
from src.models import (build_regeneration_network, build_detection_network, 
                        transfer_encoder_weights, predict_image, create_cdm)
from src.training import train_regeneration_network, train_detection_network
from src.evaluation import evaluate_model
from src.visualization import plot_training_history, plot_confusion_matrix, plot_roc_curve

# Optional: Print available devices
print("TensorFlow devices:", tf.config.list_physical_devices())
print("MPS devices (if any):", tf.config.experimental.list_physical_devices('MPS'))

np.random.seed(42)
tf.random.set_seed(42)

def main(data_path, batch_size=16, img_size=(256,256), epochs_regen=50, epochs_detect=50):
    # Load image paths
    print("Loading data paths...")
    real_image_paths = glob.glob(os.path.join(data_path, "real", "*.jpg"))
    fake_image_paths = glob.glob(os.path.join(data_path, "fake", "*.jpg"))
    if len(real_image_paths) == 0 or len(fake_image_paths) == 0:
        print(f"Error: No images found in {data_path}")
        return
    print(f"Found {len(real_image_paths)} real and {len(fake_image_paths)} fake images")
    
    # Create data generators
    print("Creating data generators...")
    train_gen, val_gen, train_det_gen, val_det_gen, train_steps, val_steps = create_data_generators(
        real_image_paths, fake_image_paths, batch_size=batch_size
    )
    
    # Build and train regeneration network
    print("Building regeneration network...")
    regen_model, encoder_outputs = build_regeneration_network(input_shape=(*img_size, 3))
    regen_model.summary()
    
    print("Training regeneration network...")
    regen_history, trained_regen_model = train_regeneration_network(
        regen_model, train_gen, val_gen, train_steps, val_steps, epochs=epochs_regen, batch_size=batch_size
    )
    
    os.makedirs("outputs/models", exist_ok=True)
    trained_regen_model.save("outputs/models/regeneration_model.h5")
    print("Regeneration model saved as 'outputs/models/regeneration_model.h5'")
    
    # Reload regeneration model with custom objects if necessary
    trained_regen_model = load_model("outputs/models/regeneration_model.h5",
                                     custom_objects={'create_cdm': create_cdm, 'mse': MeanSquaredError()})
    
    # Build detection network
    print("Building detection network...")
    detect_model = build_detection_network(input_shape=(*img_size, 3))
    
    # Transfer encoder weights from regeneration to detection network
    print("Transferring encoder weights...")
    detect_model = transfer_encoder_weights(trained_regen_model, detect_model)
    detect_model.summary()
    
    # Train detection network
    print("Training detection network...")
    detect_history, trained_detect_model = train_detection_network(
        detect_model, train_det_gen, val_det_gen, train_steps, val_steps, epochs=epochs_detect, batch_size=batch_size
    )
    
    os.makedirs("outputs/results", exist_ok=True)
    plot_training_history(regen_history, "Regeneration Network", save_path="outputs/results/regen_history.png")
    plot_training_history(detect_history, "Detection Network", save_path="outputs/results/detect_history.png")
    
    trained_detect_model.save("outputs/models/detection_model.h5")
    print("Detection model saved as 'outputs/models/detection_model.h5'")
    
    # Evaluate detection network
    print("Performing evaluation...")
    test_gen = lambda: generate_batches(real_image_paths, fake_image_paths, batch_size, is_training=False)
    metrics = evaluate_model(trained_detect_model, test_gen, test_steps=int(np.ceil((len(real_image_paths) + len(fake_image_paths))/batch_size)))
    plot_confusion_matrix(trained_detect_model, test_gen, save_path="outputs/results/confusion_matrix.png")
    plot_roc_curve(trained_detect_model, test_gen, save_path="outputs/results/roc_curve.png")
    
    return trained_regen_model, trained_detect_model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake Colorized Image Detection")
    parser.add_argument("--data_path", type=str, default="data", help="Path to dataset folder (contains 'real' and 'fake' subfolders)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--img_size", type=int, default=256, help="Image size (square)")
    parser.add_argument("--epochs_regen", type=int, default=1, help="Epochs for regeneration network")
    parser.add_argument("--epochs_detect", type=int, default=1, help="Epochs for detection network")
    args = parser.parse_args()
    
    main(
        args.data_path,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        epochs_regen=args.epochs_regen,
        epochs_detect=args.epochs_detect
    )