import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import argparse

# Import functions from our modules
from src.data_utils import create_data_generators
from src.models import build_regeneration_network, build_detection_network, custom_create_cdm
from src.training import train_regeneration_network, train_detection_network, transfer_encoder_weights
from src.evaluation import evaluate_model, predict_image
from src.visualization import plot_training_history, plot_confusion_matrix, plot_roc_curve
from src.cdm_utils import visualize_cdm

# Print available devices (optional)
print("TensorFlow devices:", tf.config.list_physical_devices())

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main(data_path, batch_size=16, img_size=(256,256), epochs_regen=50, epochs_detect=50):
    # 1. Load data paths
    print("Loading data paths...")
    real_image_paths = glob.glob(os.path.join(data_path, "real", "*.[jJ][pP][gG]"))
    fake_image_paths = glob.glob(os.path.join(data_path, "fake", "*.[jJ][pP][gG]"))
    if len(real_image_paths) == 0 or len(fake_image_paths) == 0:
        print(f"Error: No images found in {data_path}")
        return

    print(f"Found {len(real_image_paths)} real images and {len(fake_image_paths)} fake images")
    
    # 2. Create data generators
    print("Creating data generators...")
    train_gen, val_gen, train_det_gen, val_det_gen, train_steps, val_steps = create_data_generators(
        real_image_paths, fake_image_paths, batch_size=batch_size
    )
    
    # 3. Build and train regeneration network (autoencoder)
    print("Building regeneration network...")
    regen_model, encoder_outputs = build_regeneration_network(input_shape=(*img_size, 3))
    regen_model.summary()
    
    print("Training regeneration network...")
    regen_history, trained_regen_model = train_regeneration_network(
        regen_model, train_gen, val_gen, train_steps, val_steps, epochs=epochs_regen, batch_size=batch_size
    )
    
    # Plot regeneration network training history
    os.makedirs("outputs/results", exist_ok=True)
    plot_training_history(regen_history, "Regeneration Network", save_path="outputs/results/regen_history.png")
    
    # Save the regeneration model
    os.makedirs("outputs/models", exist_ok=True)
    trained_regen_model.save("outputs/models/regeneration_model.h5")
    print("Regeneration model saved.")
    
    # 4. Build detection network
    print("Building detection network...")
    detect_model = build_detection_network(input_shape=(*img_size, 3))
    
    # 5. Transfer encoder weights from regeneration to detection network
    print("Transferring encoder weights...")
    detect_model = transfer_encoder_weights(trained_regen_model, detect_model)
    detect_model.summary()
    
    # 6. Train detection network
    print("Training detection network...")
    detect_history, trained_detect_model = train_detection_network(
        detect_model, train_det_gen, val_det_gen, train_steps, val_steps, epochs=epochs_detect, batch_size=batch_size
    )
    
    # Plot detection network training history
    plot_training_history(detect_history, "Detection Network", save_path="outputs/results/detect_history.png")
    
    # Save the detection model
    trained_detect_model.save("outputs/models/detection_model.h5")
    print("Detection model saved.")
    
    # 7. Evaluation (optional)
    print("Performing evaluation...")
    test_gen = lambda: create_data_generators(real_image_paths, fake_image_paths, batch_size=batch_size)[2]()
    metrics = evaluate_model(trained_detect_model, test_gen, (len(real_image_paths) + len(fake_image_paths)) // batch_size)
    
    plot_confusion_matrix(trained_detect_model, test_gen, save_path="outputs/results/confusion_matrix.png")
    plot_roc_curve(trained_detect_model, test_gen, save_path="outputs/results/roc_curve.png")
    
    return trained_regen_model, trained_detect_model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake Colorized Image Detection")
    parser.add_argument("--data_path", type=str, default="data", help="Path to dataset with 'real' and 'fake' folders")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=256, help="Image size (square)")
    parser.add_argument("--epochs_regen", type=int, default=50, help="Epochs for regeneration network")
    parser.add_argument("--epochs_detect", type=int, default=50, help="Epochs for detection network")
    parser.add_argument("--mode", type=str, choices=["train", "test", "visualize"], default="train", 
                        help="Mode: 'train' to train, 'test' to classify an image, 'visualize' to see CDM")
    parser.add_argument("--test_image", type=str, help="Path to test image (for test or visualize modes)")
    parser.add_argument("--model_path", type=str, help="Path to saved detection model (for test mode)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        regen_model, detect_model, metrics = main(
            args.data_path, 
            batch_size=args.batch_size, 
            img_size=(args.img_size, args.img_size),
            epochs_regen=args.epochs_regen,
            epochs_detect=args.epochs_detect
        )
        print("Training complete. Evaluation metrics:")
        print(metrics)
    elif args.mode == "test":
        if not args.test_image or not args.model_path:
            print("Error: --test_image and --model_path are required for test mode.")
            exit(1)
        # Load the detection model with custom_objects so that custom_create_cdm and mse are resolved.
        detect_model = load_model(args.model_path, custom_objects={
            'custom_create_cdm': custom_create_cdm,
            'mse': MeanSquaredError()
        })
        result, confidence = predict_image(detect_model, args.test_image, img_size=(args.img_size, args.img_size))
        print(f"Classification result: {result} with {confidence:.2f} confidence")
    elif args.mode == "visualize":
        if not args.test_image:
            print("Error: --test_image is required for visualize mode.")
            exit(1)
        cdm = visualize_cdm(args.test_image, output_path="outputs/results/cdm_visualization.png")
        print("CDM visualization complete.")
    else:
        print("Unknown mode. Please use train, test, or visualize.")
