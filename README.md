# Dogs vs Cats Image Classification using CNN (TensorFlow/Keras)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Project Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images of cats and dogs. The model is trained on a dataset containing over 24,000 labeled images and achieves approximately **85–90% validation accuracy**.

The project demonstrates a complete deep learning pipeline, including dataset acquisition, preprocessing, model development, training, evaluation, and prediction.

This implementation follows industry-standard machine learning practices and is designed to be modular, scalable, and reproducible.

---

## Dataset

Dataset: Microsoft PetImages (via KaggleHub)

Total images: 24,961

Classes:

* Cat → Label 0
* Dog → Label 1

Structure:

```
PetImages/
├── Cat/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── Dog/
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

Dataset is downloaded automatically using kagglehub.

---

## Project Structure

```
dogs-vs-cats-cnn/
│
├── src/
│   ├── config.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── cnn_model.h5
│
├── outputs/
│   ├── predictions.csv
│   └── plots/
│       └── accuracy.png
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## CNN Architecture

Architecture used:

```
Input Image (150x150x3)
        │
Conv2D (32 filters, ReLU)
        │
MaxPooling2D
        │
Conv2D (64 filters, ReLU)
        │
MaxPooling2D
        │
Conv2D (128 filters, ReLU)
        │
MaxPooling2D
        │
Flatten
        │
Dense (512 units, ReLU)
        │
Dropout (0.5)
        │
Output Dense (1 unit, Sigmoid)
```

Output:

* 0 → Cat
* 1 → Dog

Loss function:

Binary Crossentropy

Optimizer:

Adam Optimizer

Metric:

Accuracy

---

## Machine Learning Pipeline

This project implements a complete ML pipeline:

1. Dataset download using kagglehub
2. Image preprocessing and normalization
3. Corrupted image detection and removal
4. Data augmentation
5. Train-validation split (80/20)
6. CNN architecture design
7. Model training
8. Performance evaluation
9. Model saving
10. Prediction generation

---

## Data Preprocessing

Applied preprocessing techniques:

Normalization:

Pixel values scaled from:

```
0–255 → 0–1
```

Data augmentation:

* Rotation
* Zoom
* Horizontal flip

Purpose:

* Improve generalization
* Reduce overfitting

---

## Overfitting Prevention Techniques

The following methods are used to prevent overfitting:

Dropout layer:

```
Dropout(0.5)
```

Data augmentation

Validation monitoring

Train-validation split

---

## Installation

Clone repository:

```
git clone https://github.com/m4ndil/dog_VS_cat.git
```

---

## Training the Model

Run:

```
python -m src.train
```

This will:

* Download dataset automatically
* Preprocess images
* Train CNN model
* Save trained model
* Generate accuracy plot

Output:

```
models/cnn_model.h5
outputs/plots/accuracy.png
```

---

## Running Predictions

Run:

```
python -m src.predict
```

Output saved at:

```
outputs/predictions.csv
```

Example output:

```
filename,label
Cat/1.jpg,0
Dog/2.jpg,1
```

---

## Model Performance

Typical results:

Accuracy: 85–90%

The model demonstrates good generalization and minimal overfitting.

---

## Training Visualization

Accuracy plot saved at:

```
outputs/plots/accuracy.png
```

Example:

```
Training accuracy increases steadily
Validation accuracy stabilizes around 85–90%
```

This confirms proper convergence and generalization.

---

## Technologies Used

Python

TensorFlow

Keras

NumPy

Pandas

Matplotlib

OpenCV

KaggleHub

---

## Key Learning Outcomes

This project demonstrates:

* CNN architecture design
* Image preprocessing techniques
* Deep learning model training
* Overfitting mitigation techniques
* Performance evaluation
* Production-ready ML project structuring

---

## Future Improvements

Potential improvements include:

Transfer Learning (MobileNetV2, ResNet)

Hyperparameter tuning

Early stopping

Learning rate scheduling

Model deployment using FastAPI or Flask

Web interface integration

---

## License

MIT License

---

## Acknowledgements

Dataset provided by Microsoft and Petfinder.
