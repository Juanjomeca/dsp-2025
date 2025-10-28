from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from utils import load_data, setup_callbacks, display_sample_images, \
                plot_training_history, evaluate_model, show_predictions

def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
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

def train_model(model, train_data, test_data, monitor='val_loss', patience=5, min_delta=0.001):
    train_images, train_labels = train_data
    test_images, test_labels = test_data

    callbacks = setup_callbacks(monitor, patience, min_delta)

    history = model.fit(
        train_images, train_labels,
        batch_size=128,
        epochs=50,
        validation_data=(test_images, test_labels),
        callbacks=callbacks,
        verbose=1
    )
    return history

def main():
    monitor = 'val_loss'
    patience = 5
    min_delta = 0.001

    # Pipeline
    train_images, train_labels, test_images, test_labels = load_data()
    display_sample_images(train_images, train_labels)

    model = create_cnn_model()
    print("\nModel Summary:")
    model.summary()

    history = train_model(
        model,
        (train_images, train_labels),
        (test_images, test_labels),
        monitor, patience, min_delta
    )

    plot_training_history(history)
    evaluate_model(model, test_images, test_labels)
    show_predictions(model, test_images, test_labels)

    model.save('last_cnn_model.keras')
    print(f"\nModel saved. Config: {monitor}, patience={patience}")

if __name__ == "__main__":
    main()
