# Image-Classification-using-CNN
# 🐱🐶 Cat vs Dog Image Classifier

A deep learning-based binary image classifier built with TensorFlow/Keras that distinguishes between cats and dogs using Convolutional Neural Networks (CNN).

## ✨ Features

- **Deep CNN Architecture**: 4-layer convolutional neural network for robust feature extraction
- **Data Augmentation**: Automatic image augmentation to prevent overfitting
- **Train/Validation Split**: 80/20 split for reliable model evaluation
- **Visual Predictions**: Display images with prediction results and confidence scores
- **Training Visualization**: Automatic plotting of accuracy and loss curves
- **Model Persistence**: Save and load trained models for future use

## 🔧 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

## 📦 Installation

1. Clone this repository:
```bash
git clone https://github.com/AdityaBawankule1/Image-Classification-using-CNN.git
```

2. Install required packages:
```bash
pip install tensorflow numpy matplotlib
```

## 📁 Dataset Setup

Organize your images in the following structure:

```
PetImages/
├── Cat/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ... (1000 cat images)
└── Dog/
    ├── 1.jpg
    ├── 2.jpg
    └── ... (1000 dog images)
```

**Important Notes:**
- Supported formats: JPG, JPEG, PNG
- Total images: 2000 (1000 cats + 1000 dogs)
- Images will be automatically resized to 150x150 pixels
- The script will automatically split data into 80% training and 20% validation

## 🚀 Usage

### Training the Model

1. Open the script and update the `DATA_DIR` variable:
```python
DATA_DIR = 'path/to/your/image/folder'
```

2. The model will:
   - Load and preprocess images
   - Apply data augmentation
   - Train for 20 epochs
   - Save the model as `cat_dog_classifier.h5`
   - Generate training history plots

### Making Predictions

After training, use the `predict_image()` function:

```python
# Predict with image display
predict_image('path/to/test/image.jpg')

# Predict without image display
predict_image('path/to/test/image.jpg', show_image=False)
```

### Loading a Saved Model

```python
from tensorflow import keras

# Load the saved model
model = keras.models.load_model('cat_dog_classifier.h5')

# Now you can use predict_image() function
```

## 🏗️ Model Architecture

```
Input (150x150x3)
    ↓
Conv2D (32 filters) + ReLU + MaxPooling
    ↓
Conv2D (64 filters) + ReLU + MaxPooling
    ↓
Conv2D (128 filters) + ReLU + MaxPooling
    ↓
Conv2D (128 filters) + ReLU + MaxPooling
    ↓
Flatten
    ↓
Dropout (0.5)
    ↓
Dense (512) + ReLU
    ↓
Dense (1) + Sigmoid
    ↓
Output (0=Cat, 1=Dog)
```

**Total Parameters:** ~7.8M trainable parameters

## 📊 Training Results

Expected performance with 2000 images:

- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 80-90%
- **Training Time**: ~5-10 minutes (depending on hardware)

The training process generates:
- `cat_dog_classifier.h5` - Saved model weights
- `training_history.png` - Accuracy and loss curves

## 🔮 Prediction

The prediction output includes:

```
==================================================
Prediction: DOG
Confidence: 87.35%
Raw Score: 0.8735 (0=Cat, 1=Dog)
==================================================
```

**Interpretation:**
- Score < 0.5 → Cat
- Score > 0.5 → Dog
- Confidence shows how certain the model is

## 🐛 Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError`
- **Solution**: Check that `DATA_DIR` path is correct and folders are named exactly `cats` and `dogs`

**Issue**: Low accuracy (< 70%)
- **Solution**: 
  - Ensure you have enough images (minimum 500 per class)
  - Check image quality
  - Train for more epochs
  - Try data augmentation parameters

**Issue**: Out of memory error
- **Solution**: Reduce `BATCH_SIZE` from 32 to 16 or 8

**Issue**: Overfitting (high training accuracy, low validation accuracy)
- **Solution**: 
  - Increase dropout rate
  - Add more data augmentation
  - Collect more training data

## 🚀 Future Enhancements

- [ ] Transfer learning with pre-trained models (VGG16, ResNet)
- [ ] Batch prediction for multiple images
- [ ] Confusion matrix and detailed metrics
- [ ] Grad-CAM visualization
- [ ] Real-time webcam classification
- [ ] Web-based GUI interface
- [ ] Model export to TensorFlow Lite
- [ ] Multi-class classification (more animal types)

## 📝 Configuration Options

You can customize these parameters in the script:

```python
IMG_SIZE = (150, 150)      # Image dimensions
BATCH_SIZE = 32            # Batch size for training
EPOCHS = 20                # Number of training epochs
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- Dataset inspiration: Kaggle Dogs vs. Cats competition
- Built with TensorFlow and Keras
- Thanks to the open-source community

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ and deep learning**
