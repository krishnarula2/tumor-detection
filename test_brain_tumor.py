#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from imutils import paths
import random

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for prediction"""
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict_brain_tumor(model, image_path):
    """Predict if an image contains a brain tumor"""
    # Load and preprocess the image
    processed_image = load_and_preprocess_image(image_path)
    
    # Make prediction
    prediction = model.predict(processed_image, verbose=0)
    
    # Get the class with highest probability
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    # Map to class names
    class_names = ['no', 'yes']  # 0: no tumor, 1: yes tumor
    result = class_names[predicted_class]
    
    return result, confidence, prediction[0]

def display_prediction(image_path, result, confidence):
    """Display the image with prediction result"""
    # Load original image for display
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(f"Prediction: {result.upper()} tumor\nConfidence: {confidence:.2%}")
    plt.axis('off')
    plt.show()

def test_random_images_from_dataset(model, dataset_path, num_images=5):
    """Test the model on random images from the dataset"""
    print("Testing on random images from dataset...")
    
    # Get all image paths
    image_paths = list(paths.list_images(dataset_path))
    
    # Select random images
    random_images = random.sample(image_paths, min(num_images, len(image_paths)))
    
    for image_path in random_images:
        # Get actual label from folder name
        actual_label = image_path.split(os.path.sep)[-2]
        
        # Make prediction
        predicted_label, confidence, raw_prediction = predict_brain_tumor(model, image_path)
        
        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"Actual: {actual_label}")
        print(f"Predicted: {predicted_label} (confidence: {confidence:.2%})")
        print(f"Raw prediction - No: {raw_prediction[0]:.3f}, Yes: {raw_prediction[1]:.3f}")
        
        # Display the image with prediction
        display_prediction(image_path, predicted_label, confidence)
        
        # Check if prediction is correct
        is_correct = (actual_label == predicted_label)
        print(f"Correct: {'✓' if is_correct else '✗'}")

def test_single_image(model, image_path):
    """Test the model on a single image"""
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    print(f"Testing image: {image_path}")
    
    # Make prediction
    predicted_label, confidence, raw_prediction = predict_brain_tumor(model, image_path)
    
    print(f"Prediction: {predicted_label} tumor")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw prediction - No: {raw_prediction[0]:.3f}, Yes: {raw_prediction[1]:.3f}")
    
    # Display the image with prediction
    display_prediction(image_path, predicted_label, confidence)

def interactive_test():
    """Interactive testing interface"""
    # Load the trained model
    try:
        print("Loading trained model...")
        model = load_model("brain_tumor_model.h5")
        print("Model loaded successfully!")
    except:
        print("Error: Could not load model. Make sure 'brain_tumor_model.h5' exists.")
        print("Run the training script first to create the model.")
        return
    
    while True:
        print("\n" + "="*50)
        print("BRAIN TUMOR DETECTION - TEST INTERFACE")
        print("="*50)
        print("1. Test random images from dataset")
        print("2. Test specific image")
        print("3. Test all images in a folder")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            dataset_path = "./dataset/brain_tumor_dataset"
            if os.path.exists(dataset_path):
                num_images = input("How many random images to test? (default 5): ").strip()
                num_images = int(num_images) if num_images.isdigit() else 5
                test_random_images_from_dataset(model, dataset_path, num_images)
            else:
                print("Dataset not found. Make sure the dataset is in the correct location.")
        
        elif choice == '2':
            image_path = input("Enter the path to the image: ").strip()
            test_single_image(model, image_path)
        
        elif choice == '3':
            folder_path = input("Enter the folder path: ").strip()
            if os.path.exists(folder_path):
                image_paths = list(paths.list_images(folder_path))
                print(f"Found {len(image_paths)} images. Testing all...")
                for image_path in image_paths:
                    test_single_image(model, image_path)
                    input("Press Enter to continue to next image...")
            else:
                print("Folder not found.")
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    interactive_test() 