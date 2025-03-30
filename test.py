import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from src.evaluation import predict_image
from src.cdm_utils import create_cdm
from src.models import custom_create_cdm

def main(test_image, img_size):
    # Set the default detection model path
    model_path = "outputs/models/detection_model.h5"
    
    # Load the detection model with custom_objects so that the custom Lambda functions are resolved.
    detect_model = load_model(model_path, custom_objects={
        'custom_create_cdm': custom_create_cdm,
        'create_cdm': create_cdm,
        'MSE': MeanSquaredError()
    })
    
    result, confidence = predict_image(detect_model, test_image, img_size=(img_size, img_size))
    print(f"Classification's result: {result} with {confidence:.2f} Accuracy")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Detection Model with Custom Image Input")
    parser.add_argument("--test_image", type=str, help="Full path to the image to be classified")
    parser.add_argument("--img_size", type=int, default=256, help="Image size (square, default: 256)")
    args = parser.parse_args()
    
    if not args.test_image:
        args.test_image = input("Enter the full path to the image to be classified: ")
    
    main(args.test_image, args.img_size)
