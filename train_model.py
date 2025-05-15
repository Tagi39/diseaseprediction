import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv("dataset.csv")

# Fill empty cells with empty string and combine all symptoms into a list
df.fillna("", inplace=True)

# Combine symptom columns into list
symptom_columns = [col for col in df.columns if "Symptom" in col]
df["symptoms"] = df[symptom_columns].values.tolist()
df["symptoms"] = df["symptoms"].apply(lambda x: [symptom.strip() for symptom in x if symptom])

# Use MultiLabelBinarizer to convert symptoms to binary features
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptoms"])
y = df["Disease"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model and symptom encoder
with open("model.pkl", "wb") as f:
    pickle.dump((model, mlb), f)

print("Model trained and saved successfully.")