import pandas as pd
import numpy as np

# Load dataset
print()
df = pd.read_csv("animal_facial_recognition/train_labels.csv")

print(df["class"].nunique())
print(df["class"].unique())

print(df["image_id"].nunique())


