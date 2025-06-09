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
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## 📁 Project Structure

```
brain-tumor-detection/
├── brain_tumor_classifier.py      # Main training and model building script
├── krish_tumor_tester.py          # Advanced testing interface
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

## 🚀 Quick Start

### Prerequisites

```bash
# Ensure Python 3.8+ is installed
python --version

# Clone the repository
git clone https://github.com/krishnarula/brain-tumor-detection.git
cd brain-tumor-detection
```

### Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
# Train with default configuration
python brain_tumor_classifier.py --train

# Train with custom parameters
python brain_tumor_classifier.py --train --epochs 30 --batch-size 16 --learning-rate 1e-4
```

### Testing and Prediction

```bash
# Interactive testing interface
python krish_tumor_tester.py --interactive

# Analyze single image
python krish_tumor_tester.py --image path/to/brain_scan.jpg

# Batch analysis
python krish_tumor_tester.py --batch path/to/images/folder
```

## 📊 Performance Metrics

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

## 🎨 Sample Output

```
🧠 KRISH NARULA'S BRAIN TUMOR ANALYSIS REPORT
================================================================================
📁 Image: brain_scan_001.jpg
⏰ Analysis Time: 2025-01-15T10:30:45
🔬 AI Model: Krish Narula's Advanced Brain Tumor Detector

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

## 🔧 Configuration

The system uses a flexible configuration system:

```python
@dataclass
class Config:
    # Data parameters
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 16
    test_size: float = 0.15
    validation_size: float = 0.15
    
    # Model parameters
    learning_rate: float = 1e-4
    epochs: int = 25
    patience: int = 7
    
    # Model architecture
    dense_units: int = 128
    dropout_rate: float = 0.5
    use_batch_norm: bool = True
```

## 📈 Training Visualization

The system generates comprehensive training visualizations:

- **Accuracy curves**: Training vs Validation accuracy over epochs
- **Loss curves**: Training vs Validation loss progression
- **Precision/Recall metrics**: Model performance metrics over time
- **Confusion matrices**: Detailed classification results

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
scikit-learn>=1.0.0
opencv-python>=4.5.0
matplotlib>=3.5.0
seaborn>=0.11.0
numpy>=1.21.0
pandas>=1.3.0
imutils>=0.5.0
pillow>=8.0.0
```

## 🔄 Future Enhancements

- [ ] Integration with DICOM medical imaging standards
- [ ] Multi-class tumor type classification
- [ ] Web-based dashboard for real-time analysis
- [ ] Mobile app integration
- [ ] Advanced visualization with 3D rendering
- [ ] Integration with hospital information systems
- [ ] Explainable AI features for medical professionals

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Krish Narula**
- LinkedIn: [Add your LinkedIn]
- Email: [Add your email]
- Portfolio: [Add your portfolio website]

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/krishnarula/brain-tumor-detection/issues).

## ⭐ Show your support

Give a ⭐️ if this project helped you!

---

*Developed with ❤️ by Krish Narula for advancing medical AI research* 