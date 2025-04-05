This repository contains a complete pipeline for detecting leaf diseases using a MobileNetV2-based deep learning model. The project employs a custom preprocessing method to enhance disease spot visualization and improves accuracy with a data augmentation strategy.

Key Features:

Preprocessing: Custom image preprocessing with a focus on enhancing the visibility of disease spots using color space transformations (LAB) and advanced masking techniques.

Model: Utilizes MobileNetV2 for transfer learning, with a custom classification head to predict leaf diseases. The model is trained on a dataset containing two classes: 'CBD' and 'fungal'.

Training Pipeline: Includes model training, fine-tuning, and evaluation. Implements callbacks like EarlyStopping, ReduceLROnPlateau, and ModelCheckpoint to ensure optimal performance and avoid overfitting.

TFLite Conversion: Converts the trained model to TensorFlow Lite format for deployment on edge devices. Both standard and quantized versions are available.

Evaluation: Provides performance metrics, including accuracy, confusion matrix, and detailed class-wise performance.

Visualization: Includes methods to visualize preprocessing steps and random samples, making it easier to understand the transformations applied to images.

The goal of this project is to build an accurate, efficient, and deployable solution for automated leaf disease detection, suitable for use in precision agriculture.

Key Files:
model_disease_detection_model.h5: Saved Keras model after training.

disease_detection_model.tflite: TensorFlow Lite model for mobile or edge deployment.

disease_detection_model_quantized.tflite: Quantized TensorFlow Lite model for optimized performance.

Requirements:
TensorFlow

OpenCV

Matplotlib

Seaborn

scikit-learn

