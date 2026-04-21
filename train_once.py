import tensorflow as tf
import numpy as np
import os

print("🚀 Training CNN Model... (5 mins)")

# Create models folder
os.makedirs('models', exist_ok=True)

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess
X_train_cnn = x_train.reshape(-1, 28, 28, 1) / 255.0
X_test_cnn = x_test.reshape(-1, 28, 28, 1) / 255.0

# Build CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(X_train_cnn, y_train, epochs=5, batch_size=128, verbose=1)

# Test accuracy
test_loss, test_acc = model.evaluate(X_test_cnn, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# Save model
model.save('models/digit_cnn_model.h5')
print("🎉 Model saved! Run: streamlit run app.py")