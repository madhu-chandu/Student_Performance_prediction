# main.py
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import os

# Create folders if not exist
os.makedirs("../output", exist_ok=True)
os.makedirs("../visuals", exist_ok=True)

# Load dataset
df = pd.read_csv("../data/StudentsPerformance.csv")
print("Dataset Loaded ✅\n")

# Clean column names
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
print("Columns cleaned:", df.columns.tolist())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# === Encode categorical columns using separate LabelEncoders ===
categorical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    # Save each encoder
    joblib.dump(le, f"../output/le_{col}.pkl")

print("Encoders saved ✅")

# Exploratory Data Analysis
plt.figure(figsize=(8, 4))
sns.histplot(df['math_score'], kde=True)
plt.title('Math Score Distribution')
plt.savefig("../visuals/math_score_dist.png")  # Save the plot
plt.close()
print("Distribution plot saved ✅")

# Model Building
X = df.drop(['math_score'], axis=1)
y = df['math_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
print(f"\nModel R² Score: {r2:.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

# Save the trained model
joblib.dump(model, "../output/math_score_model.pkl")
print("Model saved to output/ folder ✅")
