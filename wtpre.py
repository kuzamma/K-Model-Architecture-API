import os
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tensorflow.keras.applications import MobileNetV2 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Configuration
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
NUM_CLASSES = 2
CLASS_NAMES = ['CBD', 'fungal']
BASE_PATH = 'dataset'
MODEL_PATH = 'models/disease_detection_model.h5'
TF_LITE_MODEL_PATH = 'models/disease_detection_model.tflite'
TF_LITE_QUANTIZED_PATH = 'models/disease_detection_model_quantized.tflite'

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

def preprocess_leaf_image(image, show_steps=False):
    """Enhanced preprocessing with clear disease spot visualization"""
    if isinstance(image, tf.Tensor):
        image = image.numpy()
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # 1. Resize with aspect ratio preservation
    h, w = image.shape[:2]
    scale = min(INPUT_SHAPE[0]/h, INPUT_SHAPE[1]/w)
    new_h, new_w = int(h*scale), int(w*scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Pad to target size
    pad_h = (INPUT_SHAPE[0] - new_h) // 2
    pad_w = (INPUT_SHAPE[1] - new_w) // 2
    padded = cv2.copyMakeBorder(resized, 
                              pad_h, INPUT_SHAPE[0]-new_h-pad_h,
                              pad_w, INPUT_SHAPE[1]-new_w-pad_w,
                              cv2.BORDER_CONSTANT, 
                              value=[0,0,0])
    
    # 2. Convert to LAB color space
    lab = cv2.cvtColor(padded, cv2.COLOR_RGB2LAB)
    l_channel = lab[:,:,0]
    
    # 3. Create masks (using inverse thresholding for processing)
    _, leaf_mask = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, disease_spots = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    
    # Create visualization version (non-inverted for display)
    disease_spots_visual = cv2.bitwise_not(disease_spots)  # Inverted for clear visualization
    
    # 4. Combine masks
    combined_mask = cv2.bitwise_or(leaf_mask, disease_spots)
    
    # 5. Refine mask
    kernel = np.ones((3,3), np.uint8)
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 6. Find largest contour
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(clean_mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    # 7. Apply mask to LAB image
    masked_lab = cv2.bitwise_and(lab, lab, mask=final_mask)
    
    # 8. Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    masked_lab[:,:,0] = clahe.apply(masked_lab[:,:,0])
    enhanced = cv2.cvtColor(masked_lab, cv2.COLOR_LAB2RGB)
    final = enhanced.astype(np.float32) / 255.0
    
    if show_steps:
        plt.figure(figsize=(15, 10))
        steps = [
            ("1. Original", padded),
            ("2. LAB Color Space", lab),
            ("3. L Channel", l_channel),
           # ("4. Disease Spots (Visual)", disease_spots_visual),  # Using inverted for display
            ("5. Combined Mask", combined_mask),
            ("6. Final Mask", final_mask),
            ("7. Masked LAB", masked_lab),
            ("8. Final Enhanced", final)
        ]
        for i, (title, img) in enumerate(steps, 1):
            plt.subplot(2, 4, i)
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray')
            else:
                plt.imshow(img)
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return final

def visualize_random_sample():
    """Visualize preprocessing with clear disease spot display"""
    # Pick random class and image
    selected_class = random.choice(CLASS_NAMES)
    class_path = os.path.join(BASE_PATH, selected_class)
    sample_image = random.choice(os.listdir(class_path))
    image_path = os.path.join(class_path, sample_image)
    
    print(f"\nVisualizing {selected_class}/{sample_image} with disease spot preservation")
    
    # Load and process
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Couldn't load {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    preprocess_leaf_image(img, show_steps=True)
    
    

def create_data_generators():
    """Create data generators with leaf-specific preprocessing"""
    def leaf_preprocess_wrapper(x):
        processed = tf.py_function(
            lambda img: preprocess_leaf_image(img, show_steps=False),
            [x],
            tf.float32
        )
        processed.set_shape(INPUT_SHAPE)
        return processed
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=leaf_preprocess_wrapper,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2
    )
    
    val_datagen = ImageDataGenerator(
        preprocessing_function=leaf_preprocess_wrapper,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        BASE_PATH,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        BASE_PATH,
        target_size=INPUT_SHAPE[:2],
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def create_model():
    """Create MobileNetV2 model with custom classification head"""
    base_model = MobileNetV2(
        input_shape=INPUT_SHAPE,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

def train_model(model, train_generator, validation_generator):
    """Train the model with callbacks"""
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    return history

def fine_tune_model(model, base_model, train_generator, validation_generator):
    """Fine-tune the model by unfreezing some layers"""
    for layer in base_model.layers[-23:]:
        layer.trainable = True
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=50,
        verbose=1
    )
    return history

def evaluate_model(model, validation_generator):
    """Evaluate model performance"""
    validation_generator.reset()
    y_true = validation_generator.classes
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    accuracy = np.sum(y_pred == y_true) / len(y_true)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    class_accuracies = {}
    for i, class_name in enumerate(CLASS_NAMES):
        class_indices = np.where(y_true == i)[0]
        if len(class_indices) > 0:
            class_acc = np.sum(y_pred[class_indices] == i) / len(class_indices)
            class_accuracies[class_name] = class_acc
            print(f"Accuracy for {class_name}: {class_acc:.4f}")
    
    return accuracy, class_accuracies

def convert_to_tflite(model):
    """Convert model to TensorFlow Lite format"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TF_LITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()
    with open(TF_LITE_QUANTIZED_PATH, 'wb') as f:
        f.write(quantized_tflite_model)
    
    print(f"TFLite models saved to {TF_LITE_MODEL_PATH} and {TF_LITE_QUANTIZED_PATH}")

def test_tflite_model(tflite_model, validation_generator):
    """Test TFLite model on samples"""
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    validation_generator.reset()
    batch_x, batch_y = next(validation_generator)
    
    for i in range(5):
        input_data = np.expand_dims(batch_x[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        
        tflite_prediction = np.argmax(tflite_output[0])
        true_class = np.argmax(batch_y[i])
        
        print(f"Sample {i+1}:")
        print(f"  True: {CLASS_NAMES[true_class]}")
        print(f"  Pred: {CLASS_NAMES[tflite_prediction]}")
        print(f"  Confidence: {tflite_output[0][tflite_prediction]:.4f}\n")

def plot_training_history(history):
    """Plot training metrics"""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    """Main training pipeline"""
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("1. Creating data generators...")
    train_generator, validation_generator = create_data_generators()
    
    print("\n2. Visualizing preprocessing for random sample...")
    visualize_random_sample()
    
    print("\n3. Creating model...")
    model, base_model = create_model()
    model.summary()
    
    print("\n4. Training model (phase 1)...")
    history = train_model(model, train_generator, validation_generator)
    
    print("\n5. Fine-tuning model (phase 2)...")
    history_fine_tune = fine_tune_model(model, base_model, train_generator, validation_generator)
    
    print("\n6. Evaluating model...")
    accuracy, class_accuracies = evaluate_model(model, validation_generator)
    
    print("\n7. Converting to TFLite...")
    convert_to_tflite(model)
    
    print("\n8. Testing TFLite model...")
    with open(TF_LITE_QUANTIZED_PATH, 'rb') as f:
        quantized_tflite_model = f.read()
    test_tflite_model(quantized_tflite_model, validation_generator)
    
    print("\n9. Plotting training history...")
    combined_history = {k: history.history[k] + history_fine_tune.history[k] 
                      for k in history.history}
    plot_training_history(type('', (), {'history': combined_history})())
    
    print("\nTraining complete!")
    print(f"Model saved to {MODEL_PATH}")
    print(f"TFLite models saved to {TF_LITE_MODEL_PATH} and {TF_LITE_QUANTIZED_PATH}")

if __name__ == "__main__":
    main()