# patient_dashboard_improved.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

st.set_page_config(page_title="Patient Recovery — Clean Dashboard", layout="wide", initial_sidebar_state="expanded")

# -------------------------
# Helpers
# -------------------------
@st.cache_data
def load_data(path="demo.csv"):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    # Parse timestamp
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    else:
        df['Timestamp'] = pd.NaT
    # Numeric
    for col in ['R_flex_Ohms', 'Measured_Angle_Flex_Sensor']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Patient_ID'] = df['Patient_ID'].astype(str)
    return df

@st.cache_data
def load_model(path="model.pkl"):
    try:
        return joblib.load(path)
    except Exception:
        return None

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
    # x axis: Timestamp or sample index
    if patient_df['Timestamp'].notna().any():
        patient_df = patient_df.sort_values('Timestamp').reset_index(drop=True)
        x = patient_df['Timestamp']
        x_title = "Timestamp"
    else:
        patient_df = patient_df.reset_index().reset_index().rename(columns={'index':'sample_idx'})
        x = patient_df['sample_idx']
        x_title = "Sample"

    # Rolling average
    patient_df['rolling'] = patient_df['Measured_Angle_Flex_Sensor'].rolling(window=3, min_periods=1).mean()

    # Trend (linear fit)
    valid_idx = ~patient_df['Measured_Angle_Flex_Sensor'].isna()
    if valid_idx.sum() >= 2:
        xs = np.arange(valid_idx.sum())
        ys = patient_df.loc[valid_idx, 'Measured_Angle_Flex_Sensor'].values
        p = np.polyfit(xs, ys, 1)
        trend_line = p[0] * np.arange(len(patient_df)) + p[1]
    else:
        trend_line = np.full(len(patient_df), np.nan)

    # marker color from rule label
    patient_df['rule'] = patient_df['Measured_Angle_Flex_Sensor'].apply(rule_label)
    color_map = {"Recovered": "green", "Observation": "orange", "Not_Recovered": "red", "Unknown": "gray"}
    patient_df['color'] = patient_df['rule'].map(color_map).fillna("gray")

    fig = go.Figure()

    # scatter points colored by rule
    fig.add_trace(go.Scatter(
        x=x,
        y=patient_df['Measured_Angle_Flex_Sensor'],
        mode='markers+lines',
        name='Angle',
        marker=dict(color=patient_df['color'], size=8),
        line=dict(color='rgba(0,0,0,0.15)', width=1),
        hovertemplate="Angle: %{y}<br>%{x}<extra></extra>"
    ))

    # rolling avg
    fig.add_trace(go.Scatter(
        x=x,
        y=patient_df['rolling'],
        mode='lines',
        name='Rolling Avg (3)',
        line=dict(color='blue', dash='dash'),
        hovertemplate="Rolling avg: %{y:.2f}<br>%{x}<extra></extra>"
    ))

    # trend
    fig.add_trace(go.Scatter(
        x=x,
        y=trend_line,
        mode='lines',
        name='Trend (linear)',
        line=dict(color='purple'),
        hovertemplate="Trend: %{y:.2f}<br>%{x}<extra></extra>"
    ))

    # aesthetic
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title=x_title,
        yaxis_title="Measured Angle (deg)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # add threshold bands (colored background)
    fig.add_shape(type="rect",
                  xref="paper", yref="y",
                  x0=0, x1=1, y0=85, y1=180,
                  fillcolor="rgba(0,200,0,0.07)", line_width=0)
    fig.add_shape(type="rect",
                  xref="paper", yref="y",
                  x0=0, x1=1, y0=60, y1=85,
                  fillcolor="rgba(255,165,0,0.05)", line_width=0)
    fig.add_shape(type="rect",
                  xref="paper", yref="y",
                  x0=0, x1=1, y0=-10, y1=60,
                  fillcolor="rgba(255,0,0,0.03)", line_width=0)

    return fig, patient_df

def create_pdf_report(patient_id, patient_df, fig):
    # convert plot to PNG bytes (requires kaleido)
    img_bytes = fig.to_image(format="png", width=1000, height=400, scale=2)
    buf = BytesIO(img_bytes)

    # create a PDF with reportlab, insert PNG and text
    outbuf = BytesIO()
    c = canvas.Canvas(outbuf, pagesize=A4)
    w, h = A4

    margin = 20 * mm
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, h - margin - 10, f"Patient Report — {patient_id}")

    c.setFont("Helvetica", 11)
    # KPIs
    first_angle = patient_df['Measured_Angle_Flex_Sensor'].dropna().iloc[0] if not patient_df['Measured_Angle_Flex_Sensor'].dropna().empty else np.nan
    last_angle = patient_df['Measured_Angle_Flex_Sensor'].dropna().iloc[-1] if not patient_df['Measured_Angle_Flex_Sensor'].dropna().empty else np.nan
    n_records = len(patient_df)
    improvement = (last_angle - first_angle) if (not np.isnan(first_angle) and not np.isnan(last_angle)) else np.nan

    text_y = h - margin - 40
    c.drawString(margin, text_y, f"Records: {n_records}")
    c.drawString(margin + 200, text_y, f"First angle: {first_angle:.2f}°" if not np.isnan(first_angle) else "First angle: N/A")
    c.drawString(margin + 400, text_y, f"Last angle: {last_angle:.2f}°" if not np.isnan(last_angle) else "Last angle: N/A")
    text_y -= 16
    c.drawString(margin, text_y, f"Improvement (Δ): {improvement:.2f}°" if not np.isnan(improvement) else "Improvement: N/A")

    # place plot image
    # save buffer temporarily
    imgbuf = BytesIO(img_bytes)
    # draw image at a fixed position
    img_x = margin
    img_w = w - 2 * margin
    img_h = 80 * mm
    img_y = text_y - img_h - 10 * mm
    c.drawImage(ImageReader(imgbuf), img_x, img_y, width=img_w, height=img_h)

    # Footer note
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(margin, 15 * mm, "Generated by Recovery Dashboard — rule-based thresholds shown on chart")

    c.showPage()
    c.save()
    outbuf.seek(0)
    return outbuf

# small helper for reportlab image conversion
from reportlab.lib.utils import ImageReader

# -------------------------
# Sidebar / Inputs
# -------------------------
with st.sidebar:
    st.header("Data & Model")
    data_path = st.text_input("CSV path", value="demo.csv")
    model_path = st.text_input("Model path (joblib)", value="model.pkl")
    st.markdown("**Display options**")
    smoothing = st.checkbox("Show rolling average (window=3)", value=True)
    show_trend = st.checkbox("Show linear trend line", value=True)
    st.markdown("---")
    st.markdown("Click **Generate Report** to produce a downloadable PDF with the plot and summary.")

# -------------------------
# Load
# -------------------------
df = load_data(data_path)
model = load_model(model_path)  # optional: used only to show aggregated pred if available

if df.empty:
    st.error("No data loaded. Check demo.csv path.")
    st.stop()

patient_ids = sorted(df['Patient_ID'].unique())
selected = st.selectbox("Select Patient ID", patient_ids)

patient_df = df[df['Patient_ID'] == selected].copy()
if patient_df.empty:
    st.warning("No data for this patient.")
    st.stop()

# Build plot and compute metrics
fig, patient_df = build_plot(patient_df)

# Top KPIs layout
st.markdown("### Patient Summary")
k1, k2, k3, k4 = st.columns([1,1,1,1])

first_angle = patient_df['Measured_Angle_Flex_Sensor'].dropna().iloc[0] if not patient_df['Measured_Angle_Flex_Sensor'].dropna().empty else np.nan
last_angle = patient_df['Measured_Angle_Flex_Sensor'].dropna().iloc[-1] if not patient_df['Measured_Angle_Flex_Sensor'].dropna().empty else np.nan
n_records = len(patient_df)
improvement_per_sample = (last_angle - first_angle) / (n_records - 1) if n_records > 1 and not np.isnan(first_angle) and not np.isnan(last_angle) else np.nan

k1.metric("Records", n_records)
k2.metric("First angle (°)", f"{first_angle:.2f}" if not np.isnan(first_angle) else "N/A")
k3.metric("Last angle (°)", f"{last_angle:.2f}" if not np.isnan(last_angle) else "N/A")
k4.metric("Avg Δ per sample", f"{improvement_per_sample:.3f}" if not np.isnan(improvement_per_sample) else "N/A")

# Optional model aggregated prediction (if available)
if model is not None:
    try:
        # compute aggregated features (same as training script)
        agg = {
            'Max_Angle': patient_df['Measured_Angle_Flex_Sensor'].max(skipna=True),
            'Min_Resistance': patient_df['R_flex_Ohms'].min(skipna=True)
        }
        pred = model.predict(pd.DataFrame([agg]))[0]
        st.success(f"Model aggregated prediction: **{pred}**")
    except Exception:
        pass

# Display plot (big)
st.plotly_chart(fig, use_container_width=True)

# show small table and controls
with st.expander("Show patient data table"):
    st.dataframe(patient_df[['Timestamp','R_flex_Ohms','Measured_Angle_Flex_Sensor','rule']].reset_index(drop=True))

# Export buttons
export_col1, export_col2, export_col3 = st.columns(3)
with export_col1:
    csv = patient_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, file_name=f"patient_{selected}.csv", mime="text/csv")
with export_col2:
    # Export HTML report (simple)
    html_content = f"<h1>Patient Report - {selected}</h1>"
    html_content += f"<p>Records: {n_records}</p>"
    html_content += patient_df.to_html(index=False)
    st.download_button("Download HTML report", html_content, file_name=f"patient_{selected}.html", mime="text/html")
with export_col3:
    # Create PDF on demand
    if st.button("Generate PDF Report"):
        try:
            pdf_buf = create_pdf_report(selected, patient_df, fig)
            st.success("PDF generated — ready to download")
            st.download_button("Download PDF", data=pdf_buf.getvalue(), file_name=f"patient_{selected}_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Failed to create PDF: {e}")
