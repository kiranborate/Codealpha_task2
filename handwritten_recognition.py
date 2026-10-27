"""MNIST handwritten digit recognition with a simple CNN."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
tf.keras.utils.set_random_seed(42)

# MNIST has 60,000 training and 10,000 test grayscale images of digits 0–9.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(
    x_train, y_train, epochs=5, batch_size=128,
    validation_split=0.1, verbose=1
)
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
prob = model.predict(x_test, verbose=0)
pred = np.argmax(prob, axis=1)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")
print(classification_report(y_test, pred, digits=4))

with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump({"test_accuracy": float(test_accuracy), "test_loss": float(test_loss)}, f, indent=2)

fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred)).plot(ax=ax, colorbar=False)
ax.set_title("MNIST CNN — Confusion Matrix")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].squeeze(), cmap="gray")
    ax.set_title(f"True: {y_test[i]} | Pred: {pred[i]}")
    ax.axis("off")
fig.suptitle("Sample MNIST Predictions")
fig.tight_layout()
fig.savefig(OUT / "sample_predictions.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(history.history["accuracy"], label="Training accuracy")
ax.plot(history.history["val_accuracy"], label="Validation accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Training History")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "training_history.png", dpi=200)
plt.close(fig)

model.save(OUT / "mnist_cnn.keras")
print(f"Saved model, metrics and plots to: {OUT}")
