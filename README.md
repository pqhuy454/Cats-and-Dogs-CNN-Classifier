# Cats and Dogs Image Classification using CNN

A deep learning project that classifies images of cats and dogs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The project includes image cleaning, dataset preparation, model training, prediction, performance evaluation, and visualization.

## Overview

The project implements an end-to-end image classification pipeline:

- **Data Cleaning:** Detects unreadable and duplicate images before training.
- **Dataset Preparation:** Resizes images to 150 × 150 pixels and splits the dataset into training (70%), validation (15%), and testing (15%) sets.
- **CNN Training:** Builds and trains a CNN using convolutional layers, batch normalization, max pooling, dropout, and data augmentation.
- **Image Classification:** Predicts whether an image contains a cat or a dog using a sigmoid output and a 0.5 classification threshold.
- **Model Evaluation:** Measures Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- **Visualization:** Generates a confusion matrix, decision boundary plot, and calibration curve to analyze model predictions.

## Dataset

The dataset is the Microsoft Cats and Dogs dataset.

- **Download:** [Microsoft Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
- **Classes:** Cat and Dog
- **Image preprocessing:** Resize to 150 × 150 pixels and normalize pixel values to the range [0, 1].
- **Data split:** 70% training, 15% validation, and 15% testing.

After downloading and extracting the dataset, place the original image folders in the following location:

```text
data/
└── PetImages/
    ├── Cat/
    └── Dog/
```

The preprocessing scripts generate the cleaned images and train/validation/test directories automatically.

## Model Architecture

The CNN consists of:

- Convolutional layers with 32, 64, and 128 filters.
- Batch Normalization and Max Pooling layers.
- A fully connected layer with 512 units.
- Dropout for regularization.
- A sigmoid output layer for binary classification.

Training uses the Adam optimizer and binary cross-entropy loss, with Early Stopping and Model Checkpoint callbacks to help control overfitting and save the best model.

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/pqhuy454/Cats-and-Dogs-CNN-Classifier.git
cd Cats-and-Dogs-CNN-Classifier
```

**2. Install dependencies**

Python and a compatible TensorFlow environment are required.

```bash
python -m pip install -r requirements.txt
```

**3. Download and prepare the dataset**

Download the dataset from the Microsoft link above and place `Cat` and `Dog` inside `data/PetImages/`.

## Usage

Run the scripts from the project root in the following order.

**1. Clean the dataset**

```bash
python src/clean_dataset.py
```

Removes unreadable images and duplicates, then saves valid images to `data/PetImages_cleaned/`.

**2. Prepare the dataset**

```bash
python src/prepare_dataset.py
```

Resizes the images and creates the training, validation, and testing splits.

**3. Train the CNN**

```bash
python src/training.py
```

Trains the model and saves the best checkpoint to `models/model_CNN.keras`.

**4. Evaluate predictions**

```bash
python src/predict.py
```

Predicts test images, calculates evaluation metrics, saves per-image predictions, and generates the confusion matrix and classification report.

**5. Generate additional visualizations**

```bash
python src/decision_boundary.py
python src/calibration.py
```

These scripts visualize prediction probabilities and assess the calibration of the model's confidence estimates.

## Experimental Results

The trained CNN was evaluated on **3,747 test images**, with no image-processing failures during evaluation.

| Metric | Cat | Dog |
|---|---:|---:|
| Precision | 0.86 | 0.93 |
| Recall | 0.94 | 0.84 |
| F1-Score | 0.89 | 0.88 |

### Overall Performance

| Metric | Score |
|---|---:|
| Accuracy | **88.98%** |
| Macro Precision | 89.34% |
| Macro Recall | 88.98% |
| Macro F1-Score | 88.95% |

### Confusion Matrix

| Actual / Predicted | Cat | Dog |
|---|---:|---:|
| Cat | 1,756 | 117 |
| Dog | 296 | 1,578 |

The model performs well overall and identifies cats with higher recall than dogs. The 296 dogs misclassified as cats indicate an opportunity to improve dog recognition and reduce false negatives.

## Technologies

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras, CNN
- **Data Processing:** NumPy, Pandas, OpenCV
- **Evaluation:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Version Control:** Git, GitHub

## Future Improvements

- Experiment with deeper CNN architectures and transfer learning.
- Apply hyperparameter tuning and additional data augmentation.
- Improve dog recall and reduce misclassification.
- Compare model performance using additional evaluation metrics.