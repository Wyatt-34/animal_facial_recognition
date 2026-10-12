# Import TensorFlow
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img, img_to_array

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

#Load Data set 

label_csv = r"C:\Users\wah31\OneDrive\Documents\Data_Ai_Kaggle\animal_facial_recognition\train_labels.csv"
training_images = r"C:\Users\wah31\OneDrive\Documents\Data_Ai_Kaggle\enge-707-2026-s-2-cat-and-dog-facial-feature-classification-ch\train\train"


df = pd.read_csv(label_csv, dtype={'image_id': str})

class_names = [
    'CatEye',
    'DogEye',
    'CatFace',
    'DogFace',
    'CatMouth',
    'DogMouth',
    'CatNose',
    'DogNose'
]

class_to_number = {
    name: i for i, name in enumerate(class_names)
}

df['label'] = df['class'].map(class_to_number)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=123,
    stratify=df['label']
)

IMAGE_SIZE = (128, 128)

def load_image(image_id, label):
    image_path = tf.strings.join([
        training_images,
        os.sep,
        image_id,
        '.jpg'
    ])

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = image / 255.0

    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_df['image_id'].values,
        train_df['label'].values
    )
)

test_dataset = tf.data.Dataset.from_tensor_slices(
    (
        test_df['image_id'].values,
        test_df['label'].values
    )
)

train_dataset = train_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

BATCH_SIZE = 32

train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

plt.figure(figsize=(8, 8))

for images, labels in train_dataset.take(1):
    for i in range(min(25, len(images))):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i]])

plt.show()

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.4),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])


model = models.Sequential([

    data_augmentation,
    layers.Conv2D(32,(3, 3),activation='relu',input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64,(3, 3),activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64,(3, 3),activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64,(3, 3),activation='relu'),

    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(8, activation='softmax')
])

model.summary()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False
    ),
    metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]

EPOCHS = 30

history = model.fit(
    train_dataset,
    epochs= EPOCHS,
    validation_data=test_dataset,
    callbacks =callbacks
)

from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_save_path = (
    r"C:\Users\wah31\OneDrive\Documents\Data_Ai_Kaggle"
    r"\animal_facial_recognition\saved_models"
    rf"\CNN_Base_{EPOCHS}_{timestamp}.keras"
)


model.save(model_save_path)

plt.plot(
    history.history['accuracy'],
    label='accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='val_accuracy'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.show()

test_loss, test_acc = model.evaluate(
    test_dataset,
    verbose=2
)
y_true = []
y_prob = []

for images, labels in test_dataset:

    probabilities =  model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_prob.extend(probabilities.numpy())

y_true = np.array(y_true)
y_prob = np.array(y_prob)

y_pred = np.argmax(
    y_prob,
    axis=1
)


# Find misclassified images
misclassified_indices = np.where(y_true != y_pred)[0]

# Reset index so it matches y_true and y_pred
test_results = test_df.reset_index(drop=True)

# Create a DataFrame containing the misclassified images
misclassified_df = pd.DataFrame({
    'image_id': test_results.iloc[misclassified_indices]['image_id'].values,
    'true_label': [
        class_names[i] for i in y_true[misclassified_indices]
    ],
    'predicted_label': [
        class_names[i] for i in y_pred[misclassified_indices]
    ]
})

# Export to CSV
misclassified_df.to_csv(
    'misclassified_image_ids.csv',
    index=False
)

print(
    f"Exported {len(misclassified_df)} misclassified images "
    f"to misclassified_image_ids.csv"
)



cm = confusion_matrix(
    y_true,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(10, 8)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    cmap='Blues',
    xticks_rotation=45
)

plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print('\nClassification Report:\n')

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

y_true_bin = label_binarize(
    y_true,
    classes=np.arange(len(class_names))
)

print('\nROC-AUC for Each Class:')

for i, class_name in enumerate(class_names):
    class_auc = roc_auc_score(
        y_true_bin[:, i],
        y_prob[:, i]
    )

    print(
        f'{class_name}: {class_auc:.4f}'
    )

overall_auc = roc_auc_score(
    y_true_bin,
    y_prob,
    multi_class='ovr',
    average='macro'
)

print(
    f'\nOverall ROC-AUC (Macro OvR): {overall_auc:.4f}'
)

plt.figure(figsize=(10, 8))

for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(
        y_true_bin[:, i],
        y_prob[:, i]
    )

    class_auc = auc(
        fpr,
        tpr
    )

    plt.plot(
        fpr,
        tpr,
        label=f'{class_name} (AUC = {class_auc:.3f})'
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    label='Random classifier'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - One-vs-Rest')

plt.legend(
    loc='lower right',
    fontsize=9
)

plt.grid()
plt.tight_layout()
plt.show()
print('Test Accuracy is', test_acc)
