import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import time
import joblib
from utils import (load_data, setup_callbacks, display_sample_images,
                   plot_training_history, evaluate_model, show_predictions)


def create_cnn_model():
    """Crea CNN completa para pre-entrenar y luego extraer características"""
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_cnn_model_functional():
    """Crea CNN completa (versión funcional) para pre-entrenar y luego extraer características"""

    inputs = Input(shape=(28, 28, 1))

    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_cnn_backbone(model, train_data, val_data, epochs=10):
    """Pre-entrena la CNN completa"""
    train_images, train_labels = train_data
    val_images, val_labels = val_data

    callbacks = setup_callbacks(
        'val_loss', patience=5, min_delta=0.001, prefix='cnn_backbone')

    history = model.fit(
        train_images, train_labels,
        batch_size=128, epochs=epochs,
        validation_data=(val_images, val_labels),
        callbacks=callbacks, verbose=1
    )
    return history


def extract_features(cnn_model, images):
    """Extrae características usando la CNN pre-entrenada (simplificado)"""
    feature_extractor = Model(inputs=cnn_model.inputs,
                              outputs=cnn_model.layers[-3].output)
    features = feature_extractor.predict(images, verbose=0)
    return features


def train_basic_svm_classifier(features, labels):
    """Entrena clasificador SVM con parámetros por defecto"""
    labels_numeric = np.argmax(labels, axis=1) if len(
        labels.shape) > 1 else labels

    svm = SVC()
    svm.fit(features, labels_numeric)
    return svm


def train_svm_classifier(features, labels, use_grid_search=False):
    """Entrena clasificador SVM con las características extraídas"""
    labels_numeric = np.argmax(labels, axis=1) if len(
        labels.shape) > 1 else labels

    if use_grid_search:
        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        svm = SVC(random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(features, labels_numeric)
        print(f"Mejores parámetros: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        svm.fit(features, labels_numeric)
        return svm


def evaluate_and_compare_models(cnn_model, svm_model, test_images, test_labels):
    """Evalúa y compara CNN pura vs CNN-SVM"""
    # Evaluar CNN pura
    print("Evaluando CNN pura:")
    cnn_predictions = evaluate_model(cnn_model, test_images, test_labels)
    cnn_accuracy = accuracy_score(
        np.argmax(test_labels, axis=1), np.argmax(cnn_predictions, axis=1))

    # Evaluar CNN-SVM
    print("\nEvaluando CNN-SVM:")
    test_features = extract_features(cnn_model, test_images)
    svm_predictions = svm_model.predict(test_features)
    test_labels_numeric = np.argmax(test_labels, axis=1)
    svm_accuracy = accuracy_score(test_labels_numeric, svm_predictions)

    print(f"CNN-SVM Accuracy: {svm_accuracy:.4f}")
    print(classification_report(test_labels_numeric, svm_predictions))

    # Comparación
    print(f"\nCOMPARACIÓN:")
    print(f"CNN pura:     {cnn_accuracy:.4f}")
    print(f"CNN-SVM:      {svm_accuracy:.4f}")
    print(f"Diferencia:   {svm_accuracy - cnn_accuracy:+.4f}")

    return cnn_accuracy, svm_accuracy


def show_sample_predictions_cnn_svm(cnn_model, svm_model, test_images, test_labels, num_samples=10):
    """Muestra predicciones de muestra del modelo CNN-SVM"""
    indices = np.random.choice(len(test_images), num_samples, replace=False)
    sample_images = test_images[indices]
    sample_labels = test_labels[indices]

    sample_features = extract_features(cnn_model, sample_images)
    svm_predictions = svm_model.predict(sample_features)
    true_labels = np.argmax(sample_labels, axis=1)

    plt.figure(figsize=(12, 6))
    for i in range(num_samples):
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample_images[i].reshape(28, 28), cmap='gray')
        color = 'green' if svm_predictions[i] == true_labels[i] else 'red'
        plt.title(f'{true_labels[i]} → {svm_predictions[i]}', color=color)
        plt.axis('off')
    plt.suptitle('Predicciones del Modelo CNN-SVM')
    plt.tight_layout()
    plt.show()


def main():
    """Función principal simplificada"""
    print("MODELO HÍBRIDO CNN-SVM PARA MNIST")
    print("="*50)

    # 1. Cargar datos y mostrar muestras
    train_images, train_labels, test_images, test_labels = load_data()
    display_sample_images(train_images, train_labels)

    # 2. Crear y entrenar CNN
    print("\nCreando y entrenando CNN...")
    cnn_model = create_cnn_model_functional()
    cnn_model.summary()

    start_time = time.time()
    history = train_cnn_backbone(
        cnn_model, (train_images, train_labels), (test_images, test_labels))
    cnn_training_time = time.time() - start_time
    plot_training_history(history)

    # 3. Extraer características
    start_time = time.time()
    train_features = extract_features(cnn_model, train_images)
    test_features = extract_features(cnn_model, test_images)
    feature_time = time.time() - start_time

    # 4. Entrenar SVM con parámetros por defecto
    start_time = time.time()
    svm_model = train_svm_classifier(train_features, train_labels)
    svm_training_time = time.time() - start_time

    # 5. Evaluar y comparar modelos
    cnn_accuracy, svm_accuracy = evaluate_and_compare_models(
        cnn_model, svm_model, test_images, test_labels)

    # 6. Mostrar predicciones
    show_sample_predictions_cnn_svm(
        cnn_model, svm_model, test_images, test_labels)
    show_predictions(cnn_model, test_images, test_labels)

    # 7. Resumen final
    total_time = cnn_training_time + feature_time + svm_training_time
    print(f"\nRESUMEN FINAL:")
    print(f"CNN Accuracy:     {cnn_accuracy:.4f}")
    print(f"CNN-SVM Accuracy: {svm_accuracy:.4f}")
    print(f"Mejora:           {svm_accuracy - cnn_accuracy:+.4f}")
    print(f"Tiempo total:     {total_time:.2f}s")

    # 8. Guardar modelos
    cnn_model.save('cnn_feature_extractor.keras')
    joblib.dump(svm_model, 'svm_classifier.pkl')
    print("Modelos guardados: cnn_feature_extractor.keras, svm_classifier.pkl")


if __name__ == "__main__":
    main()
