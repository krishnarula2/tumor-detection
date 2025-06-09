# 🧠 Advanced Brain Tumor Detection System

**Developed by Krish Narula**

A state-of-the-art deep learning system for automated brain tumor detection using advanced computer vision and transfer learning techniques.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/accuracy-96.15%25-brightgreen.svg)

## 🚀 Project Overview

This project implements a comprehensive brain tumor detection system using advanced deep learning techniques. The system leverages transfer learning with VGG16 architecture, enhanced with custom preprocessing and medical image optimization techniques to achieve high accuracy in detecting brain tumors from MRI scans.

### 🎯 Key Features

- **🤖 Advanced AI Model**: Custom VGG16-based architecture with specialized medical image processing
- **📊 High Accuracy**: Achieves 96.15% accuracy on test dataset
- **🔬 Medical-Grade Processing**: CLAHE enhancement, Gaussian filtering, and specialized preprocessing
- **📈 Comprehensive Analytics**: Detailed confidence analysis and risk assessment
- **🖥️ Interactive Interface**: Professional CLI with real-time prediction capabilities
- **🖼️ Visual Analysis**: Side-by-side image comparison with AI predictions
- **📄 Automated Reporting**: JSON-based analysis reports with timestamps
- **🎨 Advanced Visualization**: Training metrics, confusion matrices, and prediction overlays
- **⚡ Production Ready**: Modular, scalable architecture with proper logging

## 🛠️ Technical Architecture

### Model Architecture
- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Custom Head**: GlobalAveragePooling2D + Dense layers with BatchNormalization
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout layers and early stopping
- **Input Size**: 224×224×3 (RGB images)

### Advanced Features
- **Image Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Data Augmentation**: Rotation, shifting, shearing, zooming, brightness adjustment
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
- **Metrics**: Accuracy, Precision, Recall, F1-Score

## 📁 Project Structure

```
brain-tumor-detection/
├── brain_tumor_classifier.py      # Main training and model building script
├── brain_tumor_predictor.py       # Advanced testing interface
├── brain_tumor_detection.py       # Original comparison script
├── test_brain_tumor.py           # Basic testing script
├── requirements.txt               # Project dependencies
├── README.md                     # Project documentation
├── dataset/                      # Dataset directory
│   └── brain_tumor_dataset/
│       ├── yes/                  # Tumor-positive images
│       └── no/                   # Tumor-negative images
├── models/                       # Saved models directory
├── results/                      # Analysis results and reports
├── logs/                        # Training and system logs
└── .gitignore                   # Git ignore file
```

## 🚀 Installation & Setup

### Prerequisites

```bash
# Ensure Python 3.8+ is installed
python --version

# Clone the repository
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

### Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### Prepare Dataset

1. **Download brain MRI dataset** or use your own
2. **Organize images** in this structure:
   ```
   dataset/brain_tumor_dataset/
   ├── yes/          # Images with tumors
   │   ├── Y1.jpg
   │   ├── Y2.jpg
   │   └── ...
   └── no/           # Images without tumors
       ├── N1.jpg
       ├── N2.jpg
       └── ...
   ```

## 🎓 Training the Model

### Basic Training
```bash
# Train with default settings (25 epochs, batch size 16)
python brain_tumor_classifier.py --train
```

### Advanced Training Options
```bash
# Custom training parameters
python brain_tumor_classifier.py --train --epochs 30 --batch-size 32 --learning-rate 1e-4

# Use custom configuration file
python brain_tumor_classifier.py --train --config custom_config.json
```

### What Happens During Training:
1. **Data Loading**: Automatically loads and preprocesses images
2. **Model Building**: Creates VGG16-based architecture
3. **Training**: Uses data augmentation and callbacks for optimal performance
4. **Evaluation**: Tests on validation set and generates metrics
5. **Saving**: Saves trained model and generates comprehensive report
6. **Visualization**: Creates training plots and saves them

## 🔍 Using the Prediction System

### Interactive Mode (Recommended)
```bash
python brain_tumor_predictor.py --interactive
```

This opens an interactive menu with 6 options:

#### **Option 1: 🔍 Analyze Single MRI Image**
- **Use case**: Analyze one specific brain scan
- **Input**: Path to image file
- **Output**: Detailed analysis with image visualization

**Example:**
```
Enter path to MRI image: ./dataset/brain_tumor_dataset/yes/Y1.jpg
```

#### **Option 2: 📁 Batch Analyze Folder**
- **Use case**: Analyze multiple images quickly
- **Features**: 
  - Asks if you want to see images during analysis
  - Shows progress counter
  - Option to stop analysis early

**Example:**
```
Enter folder path: ./test_images
Show images during analysis? (y/n): y
```

#### **Option 3: 🎲 Test Random Images from Dataset**
- **Use case**: Check model accuracy on random samples
- **Features**:
  - Choose number of images to test
  - Option to display images
  - Shows accuracy statistics
  - Compares actual vs predicted labels

**Example:**
```
How many random images to test? (max 253): 10
Show images during testing? (y/n): y
```

#### **Option 4: 🖼️ Visual Batch Analysis**
- **Use case**: Detailed examination with mandatory image viewing
- **Features**:
  - Shows every image with full analysis
  - Navigation options (next, skip, quit)
  - Perfect for detailed review

#### **Option 5: 📊 Model Performance Summary**
- **Use case**: View model information and specifications

#### **Option 6: ❌ Exit**
- **Use case**: Close the application

### Command Line Usage

#### **Analyze Single Image:**
```bash
# Basic analysis
python brain_tumor_predictor.py --image "path/to/brain_scan.jpg"

# With custom model
python brain_tumor_predictor.py --model "custom_model.h5" --image "scan.jpg"
```

#### **Batch Analysis:**
```bash
# Analyze all images in folder
python brain_tumor_predictor.py --batch "path/to/folder"
```

## 📊 Understanding the Results

### Sample Analysis Output
```
🧠 BRAIN TUMOR ANALYSIS REPORT
================================================================================
📁 Image: brain_scan_001.jpg
⏰ Analysis Time: 2025-01-15T10:30:45
🔬 AI Model: Advanced Brain Tumor Detection System

--------------------------------------------------
📊 ANALYSIS RESULTS
--------------------------------------------------
🔴 Status: TUMOR DETECTED
📈 Confidence: 94.3%
⚠️  Risk Level: HIGH RISK

📋 Detailed Analysis:
   • No Tumor Probability: 5.7%
   • Tumor Present Probability: 94.3%

🏥 Medical Verdict:
   Strong indication of brain tumor presence detected

💡 Recommendation:
   URGENT: Consult neurology specialist immediately for comprehensive evaluation

📊 Confidence Visualization:
   🔴 [██████████████████████████████████████████████████░░░░] 94.3%
```

### Risk Level Interpretation
- **🟢 LOW RISK**: High confidence no tumor (>90%)
- **🟡 LOW-MODERATE RISK**: Moderate confidence (70-90%)
- **🟠 MODERATE RISK**: Tumor detected with good confidence (70-90%)
- **🔴 HIGH RISK**: Tumor detected with high confidence (>90%)
- **⚪ UNCERTAIN**: Low confidence predictions (<70%)

### Image Visualization
The system displays:
- **Left Panel**: Original MRI scan
- **Right Panel**: Enhanced image with AI analysis overlay
- **Title**: Prediction status, confidence, and risk level
- **Colors**: Red for tumor detected, Green for no tumor

## 🧪 Testing with Your Own Images

### Supported Formats
- **File Types**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- **Recommended Size**: Any size (automatically resized to 224x224)
- **Color**: RGB or grayscale (converted automatically)

### Best Practices
1. **Use clear MRI brain scans** for best results
2. **Ensure good image quality** (not blurry or corrupted)
3. **Test multiple images** to understand model behavior
4. **Compare with known results** when possible

### Example Test Commands
```bash
# Test single image
python brain_tumor_predictor.py --image "./my_scan.jpg"

# Test folder of images
python brain_tumor_predictor.py --batch "./my_scans/"

# Interactive testing with visualization
python brain_tumor_predictor.py --interactive
# Then choose option 1 and enter your image path
```

## 📈 Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 96.15% |
| **Precision (Weighted)** | 96.0% |
| **Recall (Weighted)** | 96.0% |
| **F1-Score (Weighted)** | 96.0% |

### Confusion Matrix
```
Actual vs Predicted:
                 Predicted
                No   Yes
Actual    No  [ 9    1 ]
         Yes  [ 0   16 ]

True Negatives: 9   False Positives: 1
False Negatives: 0  True Positives: 16
```

## 🔧 Configuration Options

### Model Parameters
```python
# Default configuration
image_size = (224, 224)      # Input image dimensions
batch_size = 16              # Training batch size
learning_rate = 1e-4         # Learning rate for optimization
epochs = 25                  # Training epochs
patience = 7                 # Early stopping patience
```

### Customization
Create a `config.json` file:
```json
{
    "epochs": 30,
    "batch_size": 32,
    "learning_rate": 0.0001,
    "dense_units": 256,
    "dropout_rate": 0.3
}
```

## 🚨 Troubleshooting

### Common Issues

#### **Model Not Found Error**
```bash
❌ No trained model found. Please train the model first.
```
**Solution**: Run training first:
```bash
python brain_tumor_classifier.py --train
```

#### **Image Loading Error**
```bash
❌ Could not load image from path/to/image.jpg
```
**Solutions**:
- Check file path is correct
- Ensure image format is supported
- Verify file is not corrupted

#### **Memory Issues**
**Solutions**:
- Reduce batch size: `--batch-size 8`
- Close other applications
- Use smaller images

#### **Import Errors**
**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt --upgrade
```

## 🏥 Medical Disclaimer

⚠️ **IMPORTANT**: This system is designed for research and educational purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## 🛡️ Features for Resume/Portfolio

This project demonstrates expertise in:

- **Deep Learning**: Transfer learning, custom architectures, optimization
- **Computer Vision**: Medical image processing, enhancement techniques
- **Software Engineering**: Modular design, logging, configuration management
- **Data Science**: Model evaluation, statistical analysis, visualization
- **Professional Development**: Documentation, testing, CLI interfaces
- **Medical AI**: Healthcare applications, risk assessment, professional reporting

## 📦 Dependencies

```
tensorflow>=2.10.0
scikit-learn>=1.3.0
opencv-python>=4.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.21.0
pandas>=1.3.0
imutils>=0.5.0
pillow>=8.0.0
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

