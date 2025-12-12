# patient_dashboard.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Doctor Dashboard — Patient Recovery", layout="wide")

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_data(path="demo.csv"):
    df = pd.read_csv(path)
    # Standardize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]
    # Parse timestamp robustly
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    else:
        df['Timestamp'] = pd.NaT
    # Ensure numeric types
    for col in ['R_flex_Ohms', 'Measured_Angle_Flex_Sensor']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Patient_ID as string
    if 'Patient_ID' in df.columns:
        df['Patient_ID'] = df['Patient_ID'].astype(str)
    return df

@st.cache_data
def load_model(path="model.pkl"):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        # Return None if model not available or incompatible
        return None

def rule_label(angle):
    if pd.isna(angle):
        return "Unknown"
    if angle >= 85.0:
        return "Recovered"
    elif angle >= 60.0:
        return "Keep_under_observation"
    else:
        return "Not_yet_recovered"

def aggregated_features_for_patient(df_patient):
    # features used in training: Max_Angle, Min_Resistance (based on your train script)
    max_angle = df_patient['Measured_Angle_Flex_Sensor'].max(skipna=True)
    min_res = df_patient['R_flex_Ohms'].min(skipna=True)
    return {'Max_Angle': max_angle, 'Min_Resistance': min_res}

# -------------------------
# UI
# -------------------------
st.title("Doctor Dashboard — Patient Recovery")

col1, col2 = st.columns([2,1])

with col2:
    data_path = st.text_input("CSV path", value="demo.csv")
    model_path = st.text_input("Model path (joblib)", value="model.pkl")
    st.markdown("**Legend:** Rule-based labels use angle thresholds (≥85 Recovered, 60–84.9 Observe, ≤59.9 Not recovered)")

df = load_data(data_path)
model = load_model(model_path)

if df.empty:
    st.error("No data loaded. Check the CSV path and file content.")
    st.stop()

# patient list
patient_ids = sorted(df['Patient_ID'].unique())
selected = st.selectbox("Select Patient ID", patient_ids)

patient_df = df[df['Patient_ID'] == selected].copy()
if patient_df.empty:
    st.warning("No data for this patient.")
    st.stop()

st.subheader(f"Patient: {selected} — {len(patient_df)} rows")

# show table (first 200 rows)
st.dataframe(patient_df.head(200))

# compute per-row rule label
patient_df['rule_condition'] = patient_df['Measured_Angle_Flex_Sensor'].apply(rule_label)

# model aggregated prediction (if model exists)
if model is not None:
    try:
        # If the loaded model expects features in model.feature_names_in_ (sklearn)
        feat_names = getattr(model, "feature_names_in_", None)
        agg = aggregated_features_for_patient(patient_df)
        if feat_names is not None and set(feat_names).issubset(set(['Max_Angle','Min_Resistance'])):
            # prepare DataFrame in correct order
            Xagg = pd.DataFrame([agg])[list(feat_names)]
            aggregated_pred = model.predict(Xagg)[0]
            # try to get probability if available
            try:
                prob = model.predict_proba(Xagg).max()
            except Exception:
                prob = None
        else:
            # fallback: try to predict if model works on per-row values
            # (unlikely because you trained aggregated model, so prefer aggregated)
            X_try = patient_df[['Measured_Angle_Flex_Sensor','R_flex_Ohms']].rename(
                columns={'Measured_Angle_Flex_Sensor':'Max_Angle','R_flex_Ohms':'Min_Resistance'}
            ).fillna(method='ffill').head(1)  # try first row
            aggregated_pred = model.predict(pd.DataFrame([agg]))[0]
            prob = None
    except Exception as e:
        aggregated_pred = None
        prob = None
else:
    aggregated_pred = None
    prob = None

# Top-level summary
st.markdown("### Summary")
colA, colB, colC = st.columns(3)
colA.metric("Rows (samples)", len(patient_df))
colB.metric("Max Angle (deg)", f"{patient_df['Measured_Angle_Flex_Sensor'].max():.2f}")
colC.metric("Min Resistance (Ohms)", f"{patient_df['R_flex_Ohms'].min():.1f}")

if aggregated_pred is not None:
    st.success(f"Model aggregated prediction for patient: **{aggregated_pred}**" + (f" (prob {prob:.2f})" if prob is not None else ""))

# -------------------------
# Plots
# -------------------------
st.markdown("### Angle over time")
fig, ax = plt.subplots(figsize=(10,3.5))
# choose x axis
if patient_df['Timestamp'].notna().any():
    patient_df = patient_df.sort_values('Timestamp')
    x = patient_df['Timestamp']
    xlabel = "Timestamp"
else:
    patient_df = patient_df.reset_index().reset_index().rename(columns={'index':'sample_idx'})
    x = patient_df['sample_idx']
    xlabel = "Sample index"

ax.plot(x, patient_df['Measured_Angle_Flex_Sensor'], marker='o', linewidth=1)
ax.set_xlabel(xlabel)
ax.set_ylabel("Measured Angle (deg)")
ax.set_title("Measured Angle Over Time")

# annotate labels above points (small fontsize)
for xi, angle, label in zip(x, patient_df['Measured_Angle_Flex_Sensor'], patient_df['rule_condition']):
    ax.annotate(label.replace("_"," "), (xi, angle), textcoords="offset points", xytext=(0,6), ha='center', fontsize=8)

st.pyplot(fig)

# Show resistance plot if available
if 'R_flex_Ohms' in patient_df.columns:
    st.markdown("### Resistance (Ohms) over time")
    fig2, ax2 = plt.subplots(figsize=(10,2.7))
    ax2.plot(x, patient_df['R_flex_Ohms'], marker='o', linewidth=1)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("R_flex_Ohms")
    st.pyplot(fig2)

# Improvement metric: slope (degrees per sample)
try:
    angle_vals = patient_df['Measured_Angle_Flex_Sensor'].dropna().astype(float).values
    if len(angle_vals) >= 2:
        slope = (angle_vals[-1] - angle_vals[0]) / (len(angle_vals)-1)
        st.markdown(f"**Improvement (avg Δ angle per sample):** {slope:.3f}°/sample")
        # also show first and last
        st.markdown(f"First angle: {angle_vals[0]:.2f}°, Last angle: {angle_vals[-1]:.2f}°")
    else:
        st.markdown("Not enough points to compute improvement slope.")
except Exception:
    pass

# allow export of patient filtered CSV
csv_out = patient_df.to_csv(index=False)
st.download_button("Download patient CSV", csv_out, file_name=f"patient_{selected}.csv")

# show raw aggregated features used by model
st.markdown("### Aggregated features (for model)")
agg_feats = aggregated_features_for_patient(patient_df)
st.json(agg_feats)

# optionally show a simple table of rule-condition counts
st.markdown("### Rule-condition counts (this patient's rows)")
st.table(patient_df['rule_condition'].value_counts().rename_axis('Condition').reset_index(name='count'))
