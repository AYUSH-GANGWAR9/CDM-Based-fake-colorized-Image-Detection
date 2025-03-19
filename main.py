#main.py
import numpy as np
import tensorflow as tf
import os
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tqdm import tqdm
import matplotlib.pyplot as plt


# Import functions from our modules
from src.data_utils import create_data_generators
from src.models import build_regeneration_network, build_detection_network
from src.training import train_regeneration_network, train_detection_network, transfer_encoder_weights
from src.evaluation import evaluate_model
from src.visualization import plot_training_history, plot_confusion_matrix, plot_roc_curve

print(tf.config.experimental.list_physical_devices('MPS'))

    
# Verify the device being used
print("TensorFlow is using:", tf.config.list_physical_devices())
print("Available devices:", tf.config.list_physical_devices())
print("MPS detected:", tf.config.experimental.list_physical_devices('MPS'))
# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def main(data_path, batch_size=16, img_size=(256, 256), epochs_regen=50, epochs_detect=50):
    # """
    # Main function to run the entire pipeline with enhanced evaluation.
    # """
    # # Load data paths
    # print("Loading data paths...")
    # real_image_paths = glob.glob(os.path.join(data_path, "real", "*.[jJpP][pPnN][gG]"))
    # fake_image_paths = glob.glob(os.path.join(data_path, "fake", "*.[jJpP][pPnN][gG]"))

    
    # if len(real_image_paths) == 0 or len(fake_image_paths) == 0:
    #     print(f"Error: No images found in {data_path}")
    #     return
    
    # print(f"Found {len(real_image_paths)} real images and {len(fake_image_paths)} fake images")
    
    # # Create data generators
    # print("Creating data generators...")
    # train_gen, val_gen, train_det_gen, val_det_gen, train_steps, val_steps = create_data_generators(
    #     real_image_paths, fake_image_paths, batch_size=batch_size
    # )
    
    # # Build and train regeneration network
    # print("Building and training regeneration network...")
    # regen_model, encoder_outputs = build_regeneration_network(input_shape=(*img_size, 3))
    # regen_history, trained_regen_model = train_regeneration_network(
    #     regen_model, train_gen, val_gen, train_steps, val_steps, epochs=epochs_regen, batch_size=batch_size
    # )
    # os.makedirs("outputs/results", exist_ok=True)
    # # Plot regeneration network training history
    # print("Plotting regeneration network training history...")
    # plot_training_history(regen_history, "Regeneration Network", save_path="outputs/results/regen_history.png")
    
    # # Save the regeneration model
    # print("Saving regeneration model...")
    # os.makedirs("outputs/models", exist_ok=True)
    # trained_regen_model.save("outputs/models/regeneration_model.h5")

    trained_regen_model = load_model(
    'outputs/models/regeneration_model.h5', 
    custom_objects={'mse': MeanSquaredError()}
    )
    # Build detection network
    print("Building and training detection network...")
    detect_model = build_detection_network(input_shape=(*img_size, 3))
    detect_model = transfer_encoder_weights(trained_regen_model, detect_model)
    if detect_history is None:
        print("⚠️ Error: detect_history is None. Check train_detection_network function.")
        
    plot_training_history(detect_history, "Detection Network", save_path="outputs/results/detect_history.png")

    detect_history, trained_detect_model = train_detection_network(
        detect_model, train_det_gen, val_det_gen, train_steps, val_steps, epochs=epochs_detect, batch_size=batch_size
    )
    
    # Plot detection network training history
    print("Plotting detection network training history...")
    plot_training_history(detect_history, "Detection Network", save_path="outputs/results/detect_history.png")
    
    # Save the detection model
    print("Saving detection model...")
    trained_detect_model.save("outputs/models/detection_model.h5")
    
    # Comprehensive evaluation
    print("Performing comprehensive evaluation...")
    # This function would need to be imported from your data_utils module
    from src.data_utils import generate_batches
    test_gen = lambda: generate_batches(real_image_paths, fake_image_paths, batch_size, False)
    metrics = evaluate_model(trained_detect_model, test_gen, (len(real_image_paths) + len(fake_image_paths)) // batch_size)
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(trained_detect_model, test_gen, save_path="outputs/results/confusion_matrix.png")
    
    # Plot ROC curve
    print("Plotting ROC curve...")
    plot_roc_curve(trained_detect_model, test_gen, save_path="outputs/results/roc_curve.png")
    
    return trained_regen_model, trained_detect_model, metrics

if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Fake Colorized Image Detection")
    parser.add_argument("--data_path", type=str, default='data', help="Path to the dataset containing 'real' and 'fake' folders")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=256, help="Image size (square) for processing")
    parser.add_argument("--epochs_regen", type=int, default=1, help="Number of epochs for regeneration network")
    parser.add_argument("--epochs_detect", type=int, default=1, help="Number of epochs for detection network")
    
    args = parser.parse_args()
    
    # Run the main function
    main(
        args.data_path,
        batch_size=args.batch_size,
        img_size=(args.img_size, args.img_size),
        epochs_regen=args.epochs_regen,
        epochs_detect=args.epochs_detect
    )