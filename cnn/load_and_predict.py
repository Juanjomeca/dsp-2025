import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import joblib
from utils import load_data

def load_models():
    """Carga los modelos CNN y SVM"""
    cnn_model = load_model('cnn_feature_extractor.keras')
    svm_model = joblib.load('svm_classifier.pkl')
    return cnn_model, svm_model

def extract_features(cnn_model, images):
    """Extrae características con CNN"""
    temp_output = images
    for layer in cnn_model.layers[:-2]:
        temp_output = layer(temp_output)
    return temp_output.numpy() if hasattr(temp_output, 'numpy') else temp_output

def predict(cnn_model, svm_model, images):
    """Predice usando CNN + SVM"""
    features = extract_features(cnn_model, images)
    return svm_model.predict(features)

def test_models():
    """Prueba los modelos cargados"""
    cnn_model, svm_model = load_models()
    _, _, test_images, test_labels = load_data()

    sample_images = test_images[:1000]
    sample_labels = np.argmax(test_labels[:1000], axis=1)
    predictions = predict(cnn_model, svm_model, sample_images)

    accuracy = np.mean(predictions == sample_labels)
    print(f"Accuracy: {accuracy:.4f}")

    plt.figure(figsize=(16, 8))
    for i in range(450):
        plt.subplot(15, 30, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        color = 'green' if predictions[i] == sample_labels[i] else 'red'
        plt.title(f'{sample_labels[i]}→{predictions[i]}', color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_models()
