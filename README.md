# Pneumonia Detection using CNN

This project aims to detect pneumonia in chest X-ray images using Convolutional Neural Networks (CNNs). The model is trained on a dataset sourced from [describe dataset source].

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Setup](#setup)
5. [References](#references)

## Overview

Pneumonia is a prevalent infectious disease that can be diagnosed through chest X-ray images. This project uses deep learning techniques to automate the detection process, providing a faster and potentially more accurate diagnosis tool.

## Dataset

The dataset used for training, validation, and testing consists of chest X-ray images categorized into two classes:
- NORMAL: Images of healthy lungs.
- PNEUMONIA: Images indicating pneumonia presence.

The dataset is organized into three main directories:
- `train/`: Training set.
- `val/`: Validation set.
- `test/`: Test set.

For confidentiality reasons, the dataset used in this project cannot be publicly shared. However, similar datasets are available for research purposes from sources such as [provide dataset source].

## Model Architecture

The CNN model architecture used for this project is as follows:

```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
```

This architecture comprises convolutional layers with pooling for feature extraction, followed by fully connected layers for classification. The model is compiled with the Adam optimizer and binary cross-entropy loss function.

## Setup

To run the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/reminfrankline/Pneumonia-Detection-using-CNN.git
   cd pneumonia-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your dataset is structured as described in the [Dataset](#dataset) section.

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [OpenCV Documentation](https://opencv.org/)
