"""
Utilidades comunes para CNN
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def load_data():
    """Carga y preprocesa datos MNIST"""
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    test_images = test_images.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    print(f"Entrenamiento: {train_images.shape}, Prueba: {test_images.shape}")
    return train_images, train_labels, test_images, test_labels

def setup_callbacks(monitor='val_loss', patience=5, min_delta=0.001, prefix='best'):
    """Configura callbacks para entrenamiento"""
    mode = 'min' if 'loss' in monitor else 'max'

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        restore_best_weights=True,
        mode=mode,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        f'{prefix}_{monitor}_p{patience}.keras',
        monitor=monitor,
        save_best_only=True,
        mode=mode,
        verbose=1
    )

    print(f"Early Stopping: {monitor} ({mode}), patience={patience}")
    return [early_stopping, checkpoint]

def display_sample_images(train_images, train_labels):
    """Muestra muestra de imágenes del dataset"""
    labels_numeric = np.argmax(train_labels, axis=1)
    indices = np.random.randint(0, len(train_images), 16)

    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(train_images[indices[i]].reshape(28, 28), cmap='gray')
        plt.title(f'{labels_numeric[indices[i]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Grafica la historia del entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['loss'], label='Train')
    ax1.plot(history.history['val_loss'], label='Val')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='Train')
    ax2.plot(history.history['val_accuracy'], label='Val')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_images, test_labels):
    """Evalúa el modelo y muestra métricas"""
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")

    predictions = model.predict(test_images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)

    cm = confusion_matrix(true_classes, predicted_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes))
    return predictions

def show_predictions(model, test_images, test_labels, num_samples=10):
    """Muestra predicciones del modelo"""
    predictions = model.predict(test_images[:num_samples], verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels[:num_samples], axis=1)

    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
        color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
        plt.title(f'{true_classes[i]} → {predicted_classes[i]}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
