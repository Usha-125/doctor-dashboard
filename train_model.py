import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- 1. Load Data ---
print("Loading data...")
try:
    df_raw = pd.read_csv('demo.csv')
except FileNotFoundError:
    print("Error: 'demo.csv' not found. Make sure the file is in the same directory.")
    sys.exit(1)

# --- Sanity check: required columns ---
required_cols = ['Patient_ID', 'Measured_Angle_Flex_Sensor', 'R_flex_Ohms', 'Recovery_Status']
missing = [c for c in required_cols if c not in df_raw.columns]

if missing:
    print(f"Error: The following required columns are missing in 'demo.csv': {missing}")
    print("Columns found:", list(df_raw.columns))
    sys.exit(1)

# --- 2. Feature Engineering (Aggregation) ---
# The recovery status is based on the maximum angle for a patient.
print("Aggregating features per Patient_ID...")

# Calculate the maximum angle and minimum resistance (other summary stats are good too)
df_agg = (
    df_raw
    .groupby('Patient_ID')
    .agg(
        Max_Angle=('Measured_Angle_Flex_Sensor', 'max'),
        Min_Resistance=('R_flex_Ohms', 'min'),
        Recovery_Status=('Recovery_Status', 'first')  # Get the status label
    )
    .reset_index()
)

print("\nAggregated Data Head:")
print(df_agg[['Patient_ID', 'Max_Angle', 'Min_Resistance', 'Recovery_Status']].head())
print("-" * 40)

# --- 3. Define Features (X) and Target (y) ---
X = df_agg[['Max_Angle', 'Min_Resistance']]
y = df_agg['Recovery_Status']

# --- Simple sanity check on labels ---
print("Label distribution (per Recovery_Status):")
print(y.value_counts())
print("-" * 40)

if y.nunique() < 2:
    print("Error: Only one class found in 'Recovery_Status'.")
    print("At least two different classes are needed to train a classifier.")
    sys.exit(1)

# --- 4. Split Data for Training and Testing ---
print("Splitting data into train and test sets...")

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y  # This preserves class distribution
    )
except ValueError as e:
    # Fallback: if stratify fails because some classes are too small
    print(f"Warning during stratified split: {e}")
    print("Falling back to non-stratified train_test_split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=None
    )

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("-" * 40)

# --- 5. Train the Random Forest Model ---
print("Training Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Training complete.")
print("-" * 40)

# --- 6. Evaluate the Model ---
print("Evaluating model performance on the Test Set...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 7. Feature Importance ---
importance = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance_df)
print("-" * 40)

# --- 8. Example Prediction ---
print("Example: Predicting recovery status for a new patient...")

# New patient data: Max_Angle = 72.0, Min_Resistance = 45000.0
new_data = pd.DataFrame(
    [[72.0, 45000.0]],
    columns=['Max_Angle', 'Min_Resistance']
)

prediction = model.predict(new_data)
print(f"Input: Max Angle = 72.0°, Min Resistance = 45000.0 Ohms")
print(f"Predicted Recovery Status: {prediction[0]}")
