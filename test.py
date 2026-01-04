# ============================================
# FINAL COMPLETE SYSTEM: HIGH ACCURACY + BROWSER + PLOTS
# ============================================

import os
# Suppress TensorFlow logs and oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

print("="*60)
print("SYSTEM START: TRAINING & PERFORMANCE VISUALIZATION")
print("="*60)

# ============================================
# 1. DATA LOADING & ROBUST SPLITTING
# ============================================

def load_and_split_data():
    """Loads data and splits it manually to avoid 'test_size' errors"""
    df = pd.read_csv('dataset.csv')
    dataset_files = os.listdir('Dataset')
    images, labels = [], []
    
    for _, row in df.iterrows():
        img_name = str(row['image'])
        actual_path = None
        for ext in ['', '.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            p = os.path.join('Dataset', img_name + ext)
            if os.path.exists(p):
                actual_path = p
                break
        
        if actual_path:
            img = cv2.imread(actual_path)
            if img is not None:
                img = cv2.resize(img, (128, 128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                labels.append(str(row['label']))
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)
    num_classes = len(le.classes_)
    X = np.array(images).astype('float32') / 255.0
    
    X_train, y_train, X_test, y_test = [], [], [], []
    
    for i in range(num_classes):
        idx = np.where(y_encoded == i)[0]
        student_imgs = X[idx]
        if len(idx) >= 2:
            X_test.append(student_imgs[0])
            y_test.append(i)
            X_train.extend(student_imgs[1:])
            y_train.extend([i] * (len(idx) - 1))
        else:
            X_train.append(student_imgs[0])
            y_train.append(i)
            
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)
    
    return np.array(X_train), y_train_cat, np.array(X_test), y_test_cat, le, num_classes

# ============================================
# 2. ADVANCED MODEL (MobileNetV2)
# ============================================

def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(128, 128, 3), include_top=False, weights='imagenet'
    )
    base_model.trainable = False 
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ============================================
# 3. INTERACTIVE BROWSER LOGIC
# ============================================

def test_by_browsing(model, labels):
    """Opens a file explorer to pick images for testing"""
    print("\n" + "-"*40)
    print("CLICK THE POP-UP WINDOW TO SELECT A PHOTO")
    print("-"*40)
    
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    while True:
        file_path = filedialog.askopenfilename(
            title="Select Student Image for Testing",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.JPG *.PNG")]
        )
        
        if not file_path:
            break
            
        img = cv2.imread(file_path)
        img = cv2.resize(img, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img.astype('float32') / 255.0, axis=0)
        
        preds = model.predict(img, verbose=0)[0]
        idx = np.argmax(preds)
        print(f"\n[FILE]: {os.path.basename(file_path)}")
        print(f"[RESULT]: {labels[idx]} ({preds[idx]:.2%})")
        
        if input("\nBrowse for another? (y/n): ").lower() != 'y':
            break

# ============================================
# 4. MAIN RUN WITH PLOTS
# ============================================

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, le, num_classes = load_and_split_data()
    model = build_model(num_classes)
    
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
    )
    
    print(f"\nTraining on {len(X_train)} samples...")
    # CAPTURE HISTORY FOR PLOTTING
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=8),
        validation_data=(X_test, y_test) if len(X_test) > 0 else None,
        epochs=20,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    # --- PLOTTING SECTION ---
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='red')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss', color='green')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_plots.png') # Saves the image to your folder
    plt.show()
    # -----------------------

    model.save('final_trained_model.h5')
    np.save('student_labels.npy', le.classes_)
    
    test_by_browsing(model, le.classes_)