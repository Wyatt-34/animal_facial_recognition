import pandas as pd

# Read the input CSV file
input_csv = r"C:\Users\wah31\OneDrive\Documents\Data_Ai_Kaggle\animal_facial_recognition\test_results\xception_test_predictions.csv"
df = pd.read_csv(input_csv)

# Remove '.jpg' from the image_id column
df["image_id"] = df["image_id"].astype(str).str.replace(".jpg", "", regex=False)

# Remove the confidence column (3rd column)
df = df.drop(columns=["confidence"])

# Save the modified CSV file
df.to_csv("xception_text_submission.csv", index=False)

print("CSV file processed successfully!")