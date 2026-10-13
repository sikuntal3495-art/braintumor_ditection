import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# Load trained model
model = tf.keras.models.load_model("brain_tumor_model.keras")

# Load validation data
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

validation_data = datagen.flow_from_directory(
    "brain_dataset/brain_tumor_dataset",
    target_size=(128, 128),
    batch_size=16,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# User image path
image_path = input("Enter MRI image path: ").strip().strip('"')

# Load image
img = load_img(image_path, target_size=(128, 128))

# Display user image
plt.imshow(img)
plt.title("User MRI Image")
plt.axis("off")
plt.show()

# Convert image into array
img_array = img_to_array(img)
img_array = img_array / 255.0
img_array = tf.expand_dims(img_array, axis=0)

# Predict single image
prediction = model.predict(img_array, verbose=0)

# Display result
if prediction[0][0] >= 0.5:
    print("Tumor detected")
else:
    print("No tumor detected")

# Actual labels from validation data
y_true = validation_data.classes

# Predict all validation images
y_prob = model.predict(validation_data, verbose=0)

# Convert probabilities into class labels
y_pred = (y_prob >= 0.5).astype(int).flatten()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)
plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Tumor", "Tumor"],
    yticklabels=["No Tumor", "Tumor"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Brain Tumor Confusion Matrix")

plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["No Tumor", "Tumor"]
))