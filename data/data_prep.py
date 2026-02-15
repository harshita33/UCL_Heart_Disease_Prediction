import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
import os

#Input Data Path
RAW_FILE_NAME = "heart_disease_uci.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# LOAD DATA
df = pd.read_csv(RAW_FILE_NAME)

print("Original shape:", df.shape)
print("Original columns:", list(df.columns))


# TARGET CONVERSION (Binary)
# num: 0 -> no disease and num: 1,2,3,4 -> disease
df["target"] = (df["num"] > 0).astype(int)

# Encode categorical columns
categorical_cols = [
    "sex", "cp", "restecg", "slope", "thal"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

print(df.dtypes)

# DROP UNNECESSARY COLUMNS
columns_to_drop = ["id", "dataset", "num"]
df.drop(columns=columns_to_drop, inplace=True)

print("After cleaning shape:", df.shape)
print("Final columns:", list(df.columns))


# CREATE OUTPUT DIRECTORY
os.makedirs(OUTPUT_DIR, exist_ok=True)


# TRAIN-TEST SPLIT (STRATIFIED)
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["target"]
)

# SAVE FILES IN SAME DIRECTORY
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\n✅ Data preparation completed successfully!")
print("Train file:", train_path, " | Shape:", train_df.shape)
print("Test file:", test_path, " | Shape:", test_df.shape)

