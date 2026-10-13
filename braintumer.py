import os
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


# Dataset folder
dataset_dir = "brain_dataset/brain_tumor_dataset"


# Load and prepare images
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)


# Training data
train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode="binary",
    subset="training"
)


# Validation data
validation_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=(128, 128),
    batch_size=16,
    class_mode="binary",
    subset="validation",
    shuffle=False
)


# Create CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(1, activation="sigmoid")
])


# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# Train model
model.fit(
    train_data,
    epochs=10,
    validation_data=validation_data
)


# Save model
model.save("brain_tumor_model.keras")

print("Model training completed!")
print("Model saved successfully!")