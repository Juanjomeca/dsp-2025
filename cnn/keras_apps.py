import numpy as np
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from utils import load_data

def resize_and_convert(images):
    """Convierte imágenes MNIST (28x28x1) a RGB (32x32x3)"""
    resized = np.zeros((len(images), 32, 32, 3))
    for i, img in enumerate(images):
        img_resized = np.repeat(img[:, :, 0], 3).reshape(28, 28, 3)
        resized[i, 2:30, 2:30, :] = img_resized
    return resized.astype('float32')


def adapt_mnist_for_vgg():
    train_images, train_labels, test_images, test_labels = load_data()
    train_images_rgb = resize_and_convert(train_images)
    test_images_rgb = resize_and_convert(test_images)
    return train_images_rgb, train_labels, test_images_rgb, test_labels


def train_vgg_transfer_learning():
    train_images, train_labels, test_images, test_labels = adapt_mnist_for_vgg()

    # Convertir labels ai estan en formato one hot
    if len(train_labels.shape) > 1:
        train_labels_num = np.argmax(train_labels, axis=1)
        test_labels_num = np.argmax(test_labels, axis=1)
    else:
        train_labels_num = train_labels
        test_labels_num = test_labels

    train_labels_cat = to_categorical(train_labels_num, 10)
    test_labels_cat = to_categorical(test_labels_num, 10)

    train_subset = train_images[:5000]
    train_labels_subset = train_labels_cat[:5000]
    test_subset = test_images[:1000]
    test_labels_subset = test_labels_cat[:1000]

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    print("\nEntrenando modelo Transfer Learning...")
    model.fit(train_subset, train_labels_subset, epochs=5, batch_size=64, validation_split=0.1, verbose=1)

    loss, acc = model.evaluate(test_subset, test_labels_subset, verbose=0)
    print(f"Transfer Learning Accuracy: {acc:.4f}")
    return acc


def train_vgg_svm():
    feature_extractor = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    x = GlobalAveragePooling2D()(feature_extractor.output)
    feature_extractor = Model(inputs=feature_extractor.input, outputs=x)

    for layer in feature_extractor.layers:
        layer.trainable = False

    train_images, train_labels, test_images, test_labels = adapt_mnist_for_vgg()
    train_subset = train_images[:5000]
    train_labels_subset = np.argmax(train_labels[:5000], axis=1)
    test_subset = test_images[:1000]
    test_labels_subset = np.argmax(test_labels[:1000], axis=1)

    print("\nExtrayendo características con VGG16...")
    train_features = feature_extractor.predict(train_subset, verbose=0)
    test_features = feature_extractor.predict(test_subset, verbose=0)

    svm = SVC(kernel='rbf', C=1, gamma='scale')
    svm.fit(train_features, train_labels_subset)
    predictions = svm.predict(test_features)

    accuracy = accuracy_score(test_labels_subset, predictions)
    print(f"VGG16-SVM Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    print("=== COMPARACIÓN: VGG16 en MNIST ===")

    acc_transfer = train_vgg_transfer_learning()
    acc_svm = train_vgg_svm()

    print("\n=== RESULTADOS COMPARATIVOS ===")
    print(f"Transfer Learning (VGG + Dense): {acc_transfer:.4f}")
    print(f"VGG + SVM (Feature Extraction):  {acc_svm:.4f}")
