import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ---------------- CONFIG ----------------
DATASET_DIR = "dataset"  # folder containing your class folders
IMG_SIZE = (128, 128)
BATCH_SIZE = 4  # small batch size for few images
EPOCHS = 5       # train a few epochs for demo

# ---------------- DATA AUGMENTATION ----------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% for validation
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True
)

num_classes = len(train_generator.class_indices)
print("Classes:", train_generator.class_indices)
print("Number of classes:", num_classes)

# ---------------- BUILD MODEL ----------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------- TRAIN MODEL ----------------
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS
)

# ---------------- SAVE MODEL ----------------
model.save("plant_model.h5")
print("Model saved as plant_model.h5")