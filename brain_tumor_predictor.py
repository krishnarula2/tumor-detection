#!/usr/bin/env python3
"""
Advanced Brain Tumor Detection Testing Interface
Author: Krish Narula
Created: 2025

Professional testing and prediction interface for the brain tumor detection system.
Features real-time prediction, confidence analysis, and professional visualization.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from imutils import paths
import random

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img


class BrainTumorPredictor:
    """
    Professional Brain Tumor Prediction System
    Developed by Krish Narula for real-time medical image analysis
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.class_names = ['no', 'yes']  # No tumor, Yes tumor
        self.image_size = (224, 224)
        self.confidence_threshold = 0.7
        
        print("="*70)
        print("ADVANCED BRAIN TUMOR DETECTION SYSTEM")
        print("Professional Medical Image Analysis Tool")
        print("Developed by Krish Narula")
        print("="*70)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self._find_and_load_model()
    
    def _find_and_load_model(self):
        """Automatically find and load the most recent model"""
        model_paths = [
            'krish_narula_brain_tumor_model.h5',
            'models/krish_narula_brain_tumor_model.h5',
            'brain_tumor_model.h5',
            'models/best_model_checkpoint.h5'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"Found model: {path}")
                self.load_model(path)
                return
        
        print("❌ No trained model found. Please train the model first.")
        print("Run: python brain_tumor_classifier.py --train")
        sys.exit(1)
    
    def load_model(self, model_path: str):
        """Load the trained brain tumor detection model"""
        try:
            print(f"🔄 Loading model from: {model_path}")
            self.model = load_model(model_path)
            print("✅ Model loaded successfully!")
            print(f"📊 Model architecture: {self.model.name}")
            print(f"🎯 Total parameters: {self.model.count_params():,}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Advanced image preprocessing for brain MRI analysis
        
        Args:
            image_path: Path to the brain MRI image
            
        Returns:
            Preprocessed image array ready for prediction
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize to model input size
            image = cv2.resize(image, self.image_size)
            
            # Advanced preprocessing
            image = self._enhance_medical_image(image)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {e}")
            return None
    
    def _enhance_medical_image(self, image: np.ndarray) -> np.ndarray:
        """Apply medical image enhancement techniques"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Apply subtle Gaussian blur for noise reduction
        enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
        
        return enhanced
    
    def predict_single_image(self, image_path: str) -> Dict:
        """
        Predict brain tumor presence in a single MRI image
        
        Args:
            image_path: Path to the brain MRI image
            
        Returns:
            Dictionary containing prediction results and analysis
        """
        if not os.path.exists(image_path):
            return {'error': f'Image not found: {image_path}'}
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        if processed_image is None:
            return {'error': 'Failed to preprocess image'}
        
        # Make prediction
        try:
            prediction = self.model.predict(processed_image, verbose=0)
            
            # Extract results
            confidence_scores = prediction[0]
            predicted_class_idx = np.argmax(confidence_scores)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(confidence_scores[predicted_class_idx])
            
            # Determine risk level
            risk_level = self._assess_risk_level(confidence, predicted_class)
            
            # Create comprehensive result
            result = {
                'image_path': image_path,
                'prediction': predicted_class,
                'confidence': confidence,
                'confidence_percentage': confidence * 100,
                'risk_level': risk_level,
                'class_probabilities': {
                    'no_tumor': float(confidence_scores[0]),
                    'tumor_present': float(confidence_scores[1])
                },
                'timestamp': datetime.now().isoformat(),
                'model_verdict': self._generate_medical_verdict(predicted_class, confidence),
                'recommendation': self._generate_recommendation(predicted_class, confidence)
            }
            
            return result
            
        except Exception as e:
            return {'error': f'Prediction failed: {e}'}
    
    def _assess_risk_level(self, confidence: float, prediction: str) -> str:
        """Assess the risk level based on prediction and confidence"""
        if prediction == 'yes':  # Tumor detected
            if confidence >= 0.9:
                return "HIGH RISK"
            elif confidence >= 0.7:
                return "MODERATE RISK"
            else:
                return "LOW-MODERATE RISK"
        else:  # No tumor
            if confidence >= 0.9:
                return "LOW RISK"
            elif confidence >= 0.7:
                return "LOW-MODERATE RISK"
            else:
                return "UNCERTAIN"
    
    def _generate_medical_verdict(self, prediction: str, confidence: float) -> str:
        """Generate a professional medical verdict"""
        if prediction == 'yes':
            if confidence >= 0.9:
                return "Strong indication of brain tumor presence detected"
            elif confidence >= 0.7:
                return "Probable brain tumor detected - requires further analysis"
            else:
                return "Possible abnormality detected - inconclusive"
        else:
            if confidence >= 0.9:
                return "No significant abnormalities detected"
            elif confidence >= 0.7:
                return "Likely normal brain tissue"
            else:
                return "Analysis inconclusive - recommend additional imaging"
    
    def _generate_recommendation(self, prediction: str, confidence: float) -> str:
        """Generate professional medical recommendations"""
        if prediction == 'yes' and confidence >= 0.7:
            return "URGENT: Consult neurology specialist immediately for comprehensive evaluation"
        elif prediction == 'yes' and confidence < 0.7:
            return "Recommend additional MRI sequences and specialist consultation"
        elif prediction == 'no' and confidence >= 0.8:
            return "Continue routine monitoring as per standard protocols"
        else:
            return "Recommend repeat imaging with enhanced protocols"
    
    def display_prediction_results(self, result: Dict, show_image: bool = True):
        """Display comprehensive prediction results with visualization"""
        if 'error' in result:
            print(f"❌ {result['error']}")
            return
        
        # Display results header
        print("\n" + "="*80)
        print("🧠 BRAIN TUMOR ANALYSIS REPORT")
        print("="*80)
        
        # Basic information
        print(f"📁 Image: {os.path.basename(result['image_path'])}")
        print(f"⏰ Analysis Time: {result['timestamp']}")
        print(f"🔬 AI Model: Advanced Brain Tumor Detection System")
        
        print("\n" + "-"*50)
        print("📊 ANALYSIS RESULTS")
        print("-"*50)
        
        # Main prediction
        tumor_status = "TUMOR DETECTED" if result['prediction'] == 'yes' else "NO TUMOR DETECTED"
        status_color = "🔴" if result['prediction'] == 'yes' else "🟢"
        print(f"{status_color} Status: {tumor_status}")
        print(f"📈 Confidence: {result['confidence_percentage']:.1f}%")
        print(f"⚠️  Risk Level: {result['risk_level']}")
        
        # Detailed probabilities
        print(f"\n📋 Detailed Analysis:")
        print(f"   • No Tumor Probability: {result['class_probabilities']['no_tumor']*100:.1f}%")
        print(f"   • Tumor Present Probability: {result['class_probabilities']['tumor_present']*100:.1f}%")
        
        # Medical verdict
        print(f"\n🏥 Medical Verdict:")
        print(f"   {result['model_verdict']}")
        
        # Recommendation
        print(f"\n💡 Recommendation:")
        print(f"   {result['recommendation']}")
        
        # Confidence bar visualization
        self._print_confidence_bar(result['confidence'], result['prediction'])
        
        # Display image if requested
        if show_image:
            self._display_image_with_prediction(result)
        
        print("="*80)
        print("⚠️  DISCLAIMER: This is an AI analysis tool for research purposes.")
        print("   Always consult qualified medical professionals for diagnosis.")
        print("="*80)
    
    def _print_confidence_bar(self, confidence: float, prediction: str):
        """Print a visual confidence bar"""
        print(f"\n📊 Confidence Visualization:")
        bar_length = 50
        filled_length = int(bar_length * confidence)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        color = "🔴" if prediction == 'yes' else "🟢"
        print(f"   {color} [{bar}] {confidence*100:.1f}%")
    
    def _display_image_with_prediction(self, result: Dict):
        """Display the image with prediction overlay"""
        try:
            # Load and display image
            image = cv2.imread(result['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(12, 6))
            
            # Original image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title("Original MRI Scan", fontsize=12, fontweight='bold')
            plt.axis('off')
            
            # Enhanced image
            plt.subplot(1, 2, 2)
            enhanced = self._enhance_medical_image(image)
            plt.imshow(enhanced)
            
            # Prediction overlay
            status = "TUMOR DETECTED" if result['prediction'] == 'yes' else "NO TUMOR DETECTED"
            color = 'red' if result['prediction'] == 'yes' else 'green'
            
            plt.title(f"Enhanced + AI Analysis\n{status}\nConfidence: {result['confidence_percentage']:.1f}%\nRisk: {result['risk_level']}", 
                     fontsize=12, fontweight='bold', color=color)
            plt.axis('off')
            
            plt.suptitle(f"Brain Tumor Detection Analysis - {os.path.basename(result['image_path'])}", 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not display image: {e}")
    
    def interactive_testing_interface(self):
        """Interactive command-line interface for testing"""
        while True:
            print("\n" + "="*60)
            print("🧠 BRAIN TUMOR DETECTION INTERFACE")
            print("="*60)
            print("1. 🔍 Analyze single MRI image")
            print("2. 📁 Batch analyze folder of images")
            print("3. 🎲 Test random images from dataset")
            print("4. 🖼️ Visual batch analysis (see each image)")
            print("5. 📊 Model performance summary")
            print("6. ❌ Exit")
            
            choice = input("\n👉 Select option (1-6): ").strip()
            
            if choice == '1':
                image_path = input("Enter path to MRI image: ").strip()
                result = self.predict_single_image(image_path)
                self.display_prediction_results(result)
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                folder_path = input("Enter folder path: ").strip()
                if os.path.exists(folder_path):
                    image_paths = list(paths.list_images(folder_path))
                    print(f"Found {len(image_paths)} images.")
                    
                    # Ask user preferences
                    show_images = input("Show images during analysis? (y/n): ").lower().startswith('y')
                    
                    print(f"Analyzing {len(image_paths)} images...")
                    for i, image_path in enumerate(image_paths, 1):
                        print(f"\n--- Image {i}/{len(image_paths)} ---")
                        result = self.predict_single_image(image_path)
                        self.display_prediction_results(result, show_image=show_images)
                        
                        if show_images and i < len(image_paths):
                            continue_analysis = input("\nContinue to next image? (y/n): ").lower()
                            if not continue_analysis.startswith('y'):
                                break
                        elif not show_images:
                            print("-" * 40)
                else:
                    print("❌ Folder not found.")
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                self._test_random_dataset_images()
                input("Press Enter to continue...")
                
            elif choice == '4':
                self._visual_batch_analysis()
                input("Press Enter to continue...")
                
            elif choice == '5':
                self._show_model_info()
                input("Press Enter to continue...")
                
            elif choice == '6':
                print("\n👋 Thank you for using the Brain Tumor Detection System!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")
    
    def _visual_batch_analysis(self):
        """Enhanced batch analysis with mandatory image viewing"""
        folder_path = input("Enter folder path for visual analysis: ").strip()
        if not os.path.exists(folder_path):
            print("❌ Folder not found.")
            return
        
        image_paths = list(paths.list_images(folder_path))
        if not image_paths:
            print("❌ No images found in folder.")
            return
            
        print(f"🖼️ Starting visual analysis of {len(image_paths)} images...")
        print("You'll see each image with its analysis results.")
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n🔍 Analyzing image {i}/{len(image_paths)}")
            result = self.predict_single_image(image_path)
            self.display_prediction_results(result, show_image=True)
            
            if i < len(image_paths):
                next_action = input("\nOptions: (n)ext image, (s)kip remaining, (q)uit to menu: ").lower()
                if next_action.startswith('s'):
                    remaining = len(image_paths) - i
                    print(f"⏭️ Skipping remaining {remaining} images...")
                    break
                elif next_action.startswith('q'):
                    break
        
        print(f"✅ Visual analysis complete!")
    
    def _test_random_dataset_images(self):
        """Test on random images from the dataset with enhanced options"""
        dataset_path = "./dataset/brain_tumor_dataset"
        if not os.path.exists(dataset_path):
            print("❌ Dataset not found. Please ensure dataset is in the correct location.")
            return
        
        image_paths = list(paths.list_images(dataset_path))
        if not image_paths:
            print("❌ No images found in dataset.")
            return
        
        # Get user preferences
        try:
            num_images = int(input(f"How many random images to test? (max {len(image_paths)}): ") or "5")
            num_images = min(num_images, len(image_paths))
        except ValueError:
            num_images = 5
            
        show_images = input("Show images during testing? (y/n): ").lower().startswith('y')
        
        random_images = random.sample(image_paths, num_images)
        
        print(f"\n🎲 Testing {num_images} random images from dataset...")
        if show_images:
            print("🖼️ Images will be displayed for each prediction.")
        
        correct_predictions = 0
        
        for i, image_path in enumerate(random_images, 1):
            actual_label = image_path.split(os.path.sep)[-2]
            result = self.predict_single_image(image_path)
            
            if 'error' not in result:
                predicted_label = result['prediction']
                is_correct = actual_label == predicted_label
                accuracy_symbol = "✅" if is_correct else "❌"
                
                if is_correct:
                    correct_predictions += 1
                
                print(f"\n{accuracy_symbol} Image {i}/{num_images}: {os.path.basename(image_path)}")
                print(f"   Actual: {actual_label.upper()} | Predicted: {predicted_label.upper()}")
                print(f"   Confidence: {result['confidence_percentage']:.1f}%")
                print(f"   Risk Level: {result['risk_level']}")
                
                if show_images:
                    self._display_image_with_prediction(result)
                    if i < num_images:
                        input("   Press Enter for next image...")
        
        # Show summary
        accuracy = (correct_predictions / num_images) * 100
        print(f"\n📊 TESTING SUMMARY:")
        print(f"   • Total tested: {num_images}")
        print(f"   • Correct predictions: {correct_predictions}")
        print(f"   • Accuracy: {accuracy:.1f}%")
        
        if accuracy >= 90:
            print("   🎉 Excellent performance!")
        elif accuracy >= 80:
            print("   👍 Good performance!")
        elif accuracy >= 70:
            print("   👌 Fair performance!")
        else:
            print("   🔧 Model may need improvement!")
    
    def _show_model_info(self):
        """Display comprehensive model information"""
        print("\n" + "="*60)
        print("🤖 AI MODEL INFORMATION")
        print("="*60)
        print(f"🏗️  Architecture: Advanced VGG16 Transfer Learning")
        print(f"👨‍💻 Developer: Krish Narula")
        print(f"🎯 Purpose: Brain Tumor Detection in MRI Scans")
        print(f"📊 Parameters: {self.model.count_params():,}")
        print(f"🖼️  Input Size: {self.image_size}")
        print(f"🏷️  Classes: {self.class_names}")
        print(f"🔧 Preprocessing: Advanced medical image enhancement")
        print(f"📈 Features: CLAHE, Gaussian filtering, normalization")
        print("="*60)


def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(
        description="Advanced Brain Tumor Detection Testing Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python brain_tumor_predictor.py --interactive
  python brain_tumor_predictor.py --image path/to/brain_scan.jpg
  python brain_tumor_predictor.py --batch path/to/folder
        """
    )
    
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--image', type=str, help='Analyze single image')
    parser.add_argument('--batch', type=str, help='Batch analyze folder')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = BrainTumorPredictor(args.model)
    
    if args.image:
        result = predictor.predict_single_image(args.image)
        predictor.display_prediction_results(result)
        
    elif args.batch:
        if os.path.exists(args.batch):
            image_paths = list(paths.list_images(args.batch))
            print(f"✅ Found {len(image_paths)} images. Analyzing...")
            for image_path in image_paths:
                result = predictor.predict_single_image(image_path)
                predictor.display_prediction_results(result, show_image=False)
        else:
            print("❌ Batch folder not found.")
        
    elif args.interactive or len(sys.argv) == 1:
        predictor.interactive_testing_interface()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 