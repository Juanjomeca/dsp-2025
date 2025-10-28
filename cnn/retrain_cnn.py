from keras.models import load_model
from utils import load_data, setup_callbacks
import os

def main():
    model_to_load = 'best_val_loss_p5.keras'

    train_images, train_labels, test_images, test_labels = load_data()

    if os.path.exists(model_to_load):
        model = load_model(model_to_load)
        print(f"Modelo cargado: {model_to_load}")
    else:
        print(f"Error: Modelo {model_to_load} no encontrado.")
        return

    loss_before, acc_before = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Antes: Accuracy={acc_before:.4f}, Loss={loss_before:.4f}")

    callbacks = setup_callbacks('val_loss', patience=7, prefix='retrained')

    model.optimizer.learning_rate.assign(0.0001) # SOLO SI SE QUIERE UN AJUSTE FINO

    model.fit(
        train_images, train_labels,
        batch_size=128,
        epochs=20,
        validation_data=(test_images, test_labels),
        callbacks=callbacks,
        verbose=1
    )

    loss_after, acc_after = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Después: Accuracy={acc_after:.4f}, Loss={loss_after:.4f}")
    print(f"Mejora: {acc_after-acc_before:+.4f}")

    model.save('cnn_mnist_retrained.keras')
    print("Guardado: cnn_mnist_retrained.keras")

if __name__ == "__main__":
    main()
