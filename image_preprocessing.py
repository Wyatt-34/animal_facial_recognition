import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import load_img, img_to_array

def preprocessTrainingData():
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

    return train_dataset, test_dataset