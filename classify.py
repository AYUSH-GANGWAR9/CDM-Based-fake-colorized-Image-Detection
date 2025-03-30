import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from src.cdm_utils import create_cdm
from src.models import predict_image

def main(image_path, model_path, img_size):
    # Load the detection model with any custom objects
    model = load_model(model_path, custom_objects={'create_cdm': create_cdm, 'mse': MeanSquaredError()})
    # Predict the class for the given image
    label, confidence = predict_image(model, image_path, img_size=(img_size, img_size))
    print(f"Image: {image_path}")
    print(f"Prediction: {label} with {confidence:.2f} confidence")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an image as Real or Fake Colorized")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the test image")
    parser.add_argument("--model_path", type=str, default="outputs/models/detection_model.h5", help="Path to the saved detection model")
    parser.add_argument("--img_size", type=int, default=256, help="Image size (square)")
    args = parser.parse_args()
    main(args.image_path, args.model_path, args.img_size)