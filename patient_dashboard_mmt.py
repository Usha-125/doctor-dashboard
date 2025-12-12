# patient_dashboard_mmt_fixed.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO

st.set_page_config(page_title="Patient Recovery + MMT Assign", layout="wide")

# -------------------------
# Load & helpers
# -------------------------
@st.cache_data
def load_data(path="demo.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    else:
        df['Timestamp'] = pd.NaT
    for col in ['R_flex_Ohms', 'Measured_Angle_Flex_Sensor']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Patient_ID'] = df['Patient_ID'].astype(str)
    # optional fields
    for opt in ['Patient_Name', 'DOB', 'Doctor_Name']:
        if opt not in df.columns:
            df[opt] = None
    return df

def rule_label(angle: float):
    if pd.isna(angle):
        return "Unknown"
    if angle >= 85.0:
        return "Recovered"
    elif angle >= 60.0:
        return "Observation"
    else:
        return "Not_Recovered"

def build_plot(patient_df):
    # choose x axis
    if patient_df['Timestamp'].notna().any():
        patient_df = patient_df.sort_values('Timestamp').reset_index(drop=True)
        x = patient_df['Timestamp']
    else:
        patient_df = patient_df.reset_index().reset_index().rename(columns={'index':'sample_idx'})
        x = patient_df['sample_idx']

    patient_df['rolling'] = patient_df['Measured_Angle_Flex_Sensor'].rolling(window=3, min_periods=1).mean()
    valid_idx = ~patient_df['Measured_Angle_Flex_Sensor'].isna()
    if valid_idx.sum() >= 2:
        xs = np.arange(valid_idx.sum())
        ys = patient_df.loc[valid_idx, 'Measured_Angle_Flex_Sensor'].values
        p = np.polyfit(xs, ys, 1)
        trend_line = p[0] * np.arange(len(patient_df)) + p[1]
    else:
        trend_line = np.full(len(patient_df), np.nan)

    patient_df['rule'] = patient_df['Measured_Angle_Flex_Sensor'].apply(rule_label)
    color_map = {"Recovered": "green", "Observation": "orange", "Not_Recovered": "red", "Unknown": "gray"}
    patient_df['color'] = patient_df['rule'].map(color_map).fillna("gray")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=patient_df['Measured_Angle_Flex_Sensor'],
        mode='markers+lines', name='Angle',
        marker=dict(color=patient_df['color'], size=8),
        line=dict(color='rgba(0,0,0,0.12)', width=1),
        hovertemplate="Angle: %{y:.1f}<br>%{x}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=patient_df['rolling'],
        mode='lines', name='Rolling Avg (3)',
        line=dict(color='blue', dash='dash'),
        hovertemplate="Roll avg: %{y:.2f}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=trend_line,
        mode='lines', name='Trend',
        line=dict(color='purple')
    ))
    # threshold bands
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=85, y1=180, fillcolor="rgba(0,200,0,0.07)", line_width=0)
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=60, y1=85, fillcolor="rgba(255,165,0,0.05)", line_width=0)
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-20, y1=60, fillcolor="rgba(255,0,0,0.03)", line_width=0)

    fig.update_layout(template="plotly_white",
                      margin=dict(l=30,r=20,t=30,b=30),
                      xaxis_title="Timestamp / Sample",
                      yaxis_title="Measured Angle (°)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig, patient_df

# MMT reference table (from image you provided)
@st.cache_data
def reference_table():
    df = pd.DataFrame([
        {"MMT": 0, "cycles": 1, "break_min": "More than 3 min", "sets": 1, "speed%": "0–20%"},
        {"MMT": 1, "cycles": 2, "break_min": "2-3", "sets": 2, "speed%": "21–40%"},
        {"MMT": 2, "cycles": 3, "break_min": "1-2", "sets": 3, "speed%": "41–55%"},
        {"MMT": 3, "cycles": 4, "break_min": "30-60 sec", "sets": 4, "speed%": "56–70%"},
        {"MMT": 4, "cycles": "5+", "break_min": "Less than 30 sec", "sets": "5+", "speed%": "71–90%"},
        {"MMT": 5, "cycles": "6+", "break_min": "—", "sets": "6+", "speed%": "91–120%"},
    ])
    return df

# Saving assignments locally
ASSIGN_CSV = "mmt_assignments.csv"
def save_assignment(record: dict):
    df = pd.DataFrame([record])
    try:
        # append if exists
        old = pd.read_csv(ASSIGN_CSV)
        new = pd.concat([old, df], ignore_index=True)
        new.to_csv(ASSIGN_CSV, index=False)
    except FileNotFoundError:
        df.to_csv(ASSIGN_CSV, index=False)

def load_assignments():
    try:
        return pd.read_csv(ASSIGN_CSV)
    except FileNotFoundError:
        return pd.DataFrame(columns=["timestamp","Patient_ID","Assigned_MMT","Suggested_MMT","cycles","break_min","sets","speed_percent","notes","assigned_by"])

# -------------------------
# UI layout
# -------------------------
st.title("Patient Recovery + MMT Assignment")

# left: patient graph & table, right: sidebar controls
df = load_data("demo.csv")
if df.empty:
    st.error("No data loaded: place demo.csv in the folder with expected columns.")
    st.stop()

patient_ids = sorted(df['Patient_ID'].unique())
selected = st.selectbox("Select Patient ID", patient_ids)

patient_df = df[df['Patient_ID'] == selected].copy()
if patient_df.empty:
    st.warning("No rows for selected patient")
    st.stop()

fig, patient_df = build_plot(patient_df)
st.plotly_chart(fig, use_container_width=True)

# quick row table and rule label counts
patient_df['rule'] = patient_df['Measured_Angle_Flex_Sensor'].apply(rule_label)
with st.expander("Show patient readings (latest 20)"):
    st.dataframe(patient_df.sort_values('Timestamp', ascending=False).head(20)[['Timestamp','R_flex_Ohms','Measured_Angle_Flex_Sensor','rule']])

# Compute suggested MMT based on last available angle
angle_vals = patient_df['Measured_Angle_Flex_Sensor'].dropna().astype(float).values
if len(angle_vals) == 0:
    st.warning("No angle data available for this patient to suggest MMT.")
    last_angle = np.nan
else:
    last_angle = angle_vals[-1]

# convert last_angle to speed% heuristic (clamped)
if np.isnan(last_angle):
    speed_percent = np.nan
else:
    # heuristic: treat angle as proxy for % (0->0, 120->120)
    speed_percent = max(0, min(120, last_angle))
# choose MMT bucket from speed% ranges
def suggest_mmt_from_speed(sp):
    if np.isnan(sp):
        return None
    if sp <= 20:
        return 0
    elif sp <= 40:
        return 1
    elif sp <= 55:
        return 2
    elif sp <= 70:
        return 3
    elif sp <= 90:
        return 4
    else:
        return 5

suggested_mmt = suggest_mmt_from_speed(speed_percent)

# Right sidebar: assignment & reference
ref_df = reference_table()
with st.sidebar:
    st.header("MMT Assignment")
    st.markdown("**Suggested MMT** (heuristic):")
    st.metric("Suggested MMT", str(suggested_mmt) if suggested_mmt is not None else "N/A")
    st.markdown("Last angle: " + (f"{last_angle:.2f}°" if not np.isnan(last_angle) else "N/A"))
    st.markdown("Estimated speed% (proxy): " + (f"{speed_percent:.1f}%" if not np.isnan(speed_percent) else "N/A"))

    st.markdown("---")
    # show/hide reference table via button
    if st.button("Check reference table"):
        st.markdown("#### Reference Table (MMT → cycles / break / sets / speed%)")
        st.dataframe(ref_df)

    st.markdown("### Assign MMT")
    assigned_by = st.text_input("Assigned by (name)", value="Clinician")
    assigned_mmt = st.selectbox("Pick MMT to assign", options=[0,1,2,3,4,5], index=int(suggested_mmt) if suggested_mmt is not None else 0)
    # pre-fill cycles/sets/break/speed from reference table
    sel_row = ref_df[ref_df['MMT'] == assigned_mmt].iloc[0]
    cycles_default = sel_row['cycles']
    break_default = sel_row['break_min']
    sets_default = sel_row['sets']
    speed_default = sel_row['speed%']

    cycles = st.text_input("cycles", value=str(cycles_default))
    break_min = st.text_input("break(min)", value=str(break_default))
    sets = st.text_input("sets", value=str(sets_default))
    speed_pc = st.text_input("speed%", value=str(speed_default))
    notes = st.text_area("Notes (optional)", value="")

    if st.button("Save MMT assignment"):
        rec = {
            "timestamp": datetime.now().isoformat(),
            "Patient_ID": selected,
            "Assigned_MMT": int(assigned_mmt),
            "Suggested_MMT": int(suggested_mmt) if suggested_mmt is not None else "",
            "cycles": cycles,
            "break_min": break_min,
            "sets": sets,
            "speed_percent": speed_pc,
            "notes": notes,
            "assigned_by": assigned_by
        }
        save_assignment(rec)
        st.success(f"Saved assignment for {selected}: MMT {assigned_mmt}")
        # no experimental_rerun() — Streamlit will re-run automatically after button click

    st.markdown("---")
    st.markdown("Assignment history (latest 10):")
    history = load_assignments()
    if not history.empty:
        st.dataframe(history[history['Patient_ID'] == selected].sort_values('timestamp', ascending=False).head(10))
    else:
        st.write("No assignments yet.")

# below main area: show overall patient KPIs
st.markdown("### KPIs")
cols = st.columns(4)
n_records = len(patient_df)
first_angle = angle_vals[0] if len(angle_vals) >= 1 else np.nan
last_angle = last_angle
improvement = (last_angle - first_angle) if (not np.isnan(first_angle) and not np.isnan(last_angle)) else np.nan
avg_delta = (improvement / (n_records - 1)) if n_records > 1 and not np.isnan(improvement) else np.nan
cols[0].metric("Records", n_records)
cols[1].metric("First angle (°)", f"{first_angle:.2f}" if not np.isnan(first_angle) else "N/A")
cols[2].metric("Last angle (°)", f"{last_angle:.2f}" if not np.isnan(last_angle) else "N/A")
cols[3].metric("Avg Δ / sample", f"{avg_delta:.3f}" if not np.isnan(avg_delta) else "N/A")

# Show last saved assignment (if any)
st.markdown("### Latest assignment (this patient)")
hist = load_assignments()
if not hist.empty and selected in hist['Patient_ID'].values:
    last = hist[hist['Patient_ID'] == selected].sort_values('timestamp', ascending=False).iloc[0]
    st.write(f"Assigned MMT: **{last['Assigned_MMT']}** by {last.get('assigned_by','')}, at {last['timestamp']}")
    st.write("Details:", dict(last[['cycles','break_min','sets','speed_percent','notes']]))
else:
    st.info("No previous assignment found for this patient.")

# small help text
st.info("Suggestion heuristic uses the *last measured angle* as a proxy to speed% and picks MMT buckets. You can override and save custom assignments for each patient.")
