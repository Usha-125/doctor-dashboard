# patient_dashboard_final.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from PIL import Image

st.set_page_config(page_title="Mayuri Hospital — Patient Recovery", layout="wide")

# ---------- CONFIG / FILES ----------
DATA_CSV = "demo.csv"
ASSIGN_CSV = "mmt_assignments.csv"
LOGO_PATH = "clinic_logo.png"  # optional logo file in folder

CLINIC_NAME = "Mayuri Hospital"
CLINIC_TAGLINE = "Orthopedic Excellence — Powered by Technology"
PAGE_HEADLINE = "Intelligent, real-time insights empowering better patient recovery."

# doctor info for report
DOCTOR_NAME = "Dr. Kaver S A"
DOCTOR_SPECIALTY = "Orthopedic Hand Surgeon"
DOCTOR_EXP = "10 yrs"

# ---------- HELPERS ----------
@st.cache_data
def load_data(path=DATA_CSV):
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
    for opt in ['Patient_Name','DOB','Doctor_Name']:
        if opt not in df.columns:
            df[opt] = None
    return df

def load_assignments(path=ASSIGN_CSV):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=["timestamp","Patient_ID","Assigned_MMT","Suggested_MMT","cycles","break_min","sets","speed_percent","notes","assigned_by"])

def save_assignment(record: dict, path=ASSIGN_CSV):
    df = pd.DataFrame([record])
    try:
        old = pd.read_csv(path)
        new = pd.concat([old, df], ignore_index=True)
        new.to_csv(path, index=False)
    except FileNotFoundError:
        df.to_csv(path, index=False)

def rule_label(angle):
    if pd.isna(angle):
        return "Unknown"
    if angle >= 85.0:
        return "Recovered"
    elif angle >= 60.0:
        return "Keep under observation"
    else:
        return "Not Yet Recovered"

def build_plot(patient_df):
    # prepare x axis
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
    color_map = {"Recovered":"green","Keep under observation":"orange","Not yet recovered":"red","Unknown":"gray"}
    patient_df['color'] = patient_df['rule'].map(color_map).fillna("gray")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=patient_df['Measured_Angle_Flex_Sensor'], mode='markers+lines', name='Angle',
                             marker=dict(color=patient_df['color'], size=8),
                             line=dict(color='rgba(0,0,0,0.12)', width=1),
                             hovertemplate="Angle: %{y:.1f}<br>%{x}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=patient_df['rolling'], mode='lines', name='Rolling Avg (3)', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=x, y=trend_line, mode='lines', name='Trend', line=dict(color='purple')))

    # add colored threshold bands
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=85, y1=180, fillcolor="rgba(0,200,0,0.06)", line_width=0)
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=60, y1=85, fillcolor="rgba(255,165,0,0.04)", line_width=0)
    fig.add_shape(type="rect", xref="paper", yref="y", x0=0, x1=1, y0=-20, y1=60, fillcolor="rgba(255,0,0,0.03)", line_width=0)

    fig.update_layout(template="plotly_white", margin=dict(l=20,r=20,t=20,b=20),
                      xaxis_title="Timestamp / Sample", yaxis_title="Measured Angle (deg)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig, patient_df

def create_pdf_report(patient_id, patient_df, fig, clinic_name=CLINIC_NAME, clinic_tag=CLINIC_TAGLINE, headline=PAGE_HEADLINE,
                      doc_name=DOCTOR_NAME, doc_specialty=DOCTOR_SPECIALTY, doc_exp=DOCTOR_EXP, logo_path=LOGO_PATH):
    # convert fig to png (kaleido required)
    img_bytes = fig.to_image(format="png", width=1200, height=420, scale=2)
    img_buf = BytesIO(img_bytes)

    out = BytesIO()
    c = canvas.Canvas(out, pagesize=A4)
    W, H = A4
    margin = 18 * mm
    y = H - margin

    # Header block
    if logo_path:
        try:
            logo_img = Image.open(logo_path)
            max_h = 20 * mm
            w_ratio = logo_img.width / logo_img.height
            logo_w = max_h * w_ratio
            logo_h = max_h
            c.drawImage(ImageReader(logo_img), margin, y - logo_h, width=logo_w, height=logo_h, mask='auto')
        except Exception:
            pass

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(W/2, y - 8, clinic_name)
    c.setFont("Helvetica", 10)
    c.drawCentredString(W/2, y - 24, clinic_tag)
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(W/2, y - 38, headline)
    y -= 54

    # Doctor info box (right side)
    
    box_h = 16 * mm   # Slightly increased to allow top padding + 2 text lines
    c.roundRect(margin, y - box_h, W - 2*margin, box_h, radius=6, stroke=1, fill=0)

# Padding from top of box (leave one empty line)
    top_padding = 6   # Adjust as needed (6–7 works well)

# Draw text with padding
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin + 6, y - top_padding - 6, f"Doctor Name: {doc_name}")

# Adjust this for line spacing between the two lines
    line_spacing = 12   # try 10–14 depending on spacing you want

    c.setFont("Helvetica", 8.5)
    c.drawString(margin + 6, y - top_padding - 6 - line_spacing,
             f"{doc_specialty}   |   Experience: {doc_exp}")


# Update Y position
    y -= box_h + 6


    # Reduced spacing below
    #y -= box_h + 6
    #bottom_pad = 8 
    line_space = 10    # adjust this number to control space
    y -= line_space
    #box over
    # Patient summary area (KPIs + current condition and latest assignment)
    angle_vals = patient_df['Measured_Angle_Flex_Sensor'].dropna().astype(float).values
    n_records = len(patient_df)
    first_angle = angle_vals[0] if len(angle_vals) >= 1 else np.nan
    last_angle = angle_vals[-1] if len(angle_vals) >= 1 else np.nan
    improvement = (last_angle - first_angle) if (not np.isnan(first_angle) and not np.isnan(last_angle)) else np.nan
    avg_delta = (improvement / (n_records - 1)) if n_records > 1 and not np.isnan(improvement) else np.nan

    # KPIs table (left) and latest assignment (right)
    assignments = load_assignments()
    last_assign = None
    if not assignments.empty and patient_id in assignments['Patient_ID'].values:
        last_assign = assignments[assignments['Patient_ID'] == patient_id].sort_values('timestamp', ascending=False).iloc[0]

    # Draw KPIs as text
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin, y, "Patient Summary")
    c.setFont("Helvetica", 9)
    y -= 12
    c.drawString(margin, y, f"Patient ID: {patient_id}")
    c.drawString(margin + 220, y, f"Records: {n_records}")
    y -= 12
    c.drawString(margin, y, f"First angle (°): {first_angle:.2f}" if not np.isnan(first_angle) else "First angle (°): N/A")
    c.drawString(margin + 220, y, f"Last angle (°): {last_angle:.2f}" if not np.isnan(last_angle) else "Last angle (°): N/A")
    y -= 12
    c.drawString(margin, y, f"Avg Δ / sample: {avg_delta:.3f}" if not np.isnan(avg_delta) else "Avg Δ / sample: N/A")
    # Current condition (from last reading)
    current_condition = rule_label(last_angle) if not np.isnan(last_angle) else "Unknown"
    c.drawString(margin + 220, y, f"Current condition: {current_condition}")
    y -= 18

    # Latest assignment block
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin, y, "Latest MMT Assignment")
    y -= 12
    c.setFont("Helvetica", 9)
    if last_assign is not None:
        c.drawString(margin, y, f"Assigned MMT: {int(last_assign['Assigned_MMT'])}    By: {last_assign.get('assigned_by','-')}    On: {last_assign['timestamp']}")
        y -= 12
        c.drawString(margin, y, f"Details: cycles={last_assign.get('cycles','-')}, break={last_assign.get('break_min','-')}, sets={last_assign.get('sets','-')}, speed={last_assign.get('speed_percent','-')}")
        y -= 18
    else:
        c.drawString(margin, y, "No assignment found")
        y -= 18

    # Insert plot image
    img_reader = ImageReader(img_buf)
    img_w = W - 2*margin
    img_h = 85 * mm
    c.drawImage(img_reader, margin, y - img_h, width=img_w, height=img_h)
    y -= img_h + 10

    # Recent readings table
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin, y, "Recent readings (latest 10)")
    y -= 12
    recent = patient_df.sort_values('Timestamp', ascending=False).head(10)[['Timestamp','R_flex_Ohms','Measured_Angle_Flex_Sensor','rule']].reset_index(drop=True)
    col_w = (W - 2*margin) / 4
    c.setFont("Helvetica-Bold", 9)
    c.drawString(margin + 2, y, "Timestamp")
    c.drawString(margin + col_w + 4, y, "Resistance (Ω)")
    c.drawString(margin + 2*col_w + 8, y, "Angle (°)")
    c.drawString(margin + 3*col_w + 12, y, "Condition")
    y -= 12
    c.setFont("Helvetica", 9)
    for idx, row in recent.iterrows():
        ts = row['Timestamp']
        ts_s = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M") if pd.notna(ts) else "-"
        r = f"{row['R_flex_Ohms']:.1f}" if pd.notna(row['R_flex_Ohms']) else "-"
        ang = f"{row['Measured_Angle_Flex_Sensor']:.1f}" if pd.notna(row['Measured_Angle_Flex_Sensor']) else "-"
        cond = str(row['rule']).replace("_"," ")
        c.drawString(margin + 2, y, ts_s)
        c.drawString(margin + col_w + 4, y, r)
        c.drawString(margin + 2*col_w + 8, y, ang)
        c.drawString(margin + 3*col_w + 12, y, cond)
        y -= 12
        if y < 30*mm:
            break

    # Footer
    c.setFont("Helvetica-Oblique", 8)
    c.drawCentredString(W/2, 12*mm, "Generated by Mayuri Hospital Recovery Dashboard")
    c.showPage()
    c.save()
    out.seek(0)
    return out

# ---------- UI layout ----------
st.markdown(f"## {CLINIC_NAME} — {CLINIC_TAGLINE}", unsafe_allow_html=True)
st.markdown(f"**{PAGE_HEADLINE}**")

# Load data
df = load_data()
if df.empty:
    st.error("No data available — put demo.csv in this folder with expected columns.")
    st.stop()

patient_ids = sorted(df['Patient_ID'].unique())
selected = st.selectbox("Select Patient ID", patient_ids)

patient_df = df[df['Patient_ID'] == selected].copy()
if patient_df.empty:
    st.warning("No rows for selected patient.")
    st.stop()

# KPIs first (table)
angle_vals = patient_df['Measured_Angle_Flex_Sensor'].dropna().astype(float).values
n_records = len(patient_df)
first_angle = angle_vals[0] if len(angle_vals) >= 1 else np.nan
last_angle = angle_vals[-1] if len(angle_vals) >= 1 else np.nan
improvement = (last_angle - first_angle) if (not np.isnan(first_angle) and not np.isnan(last_angle)) else np.nan
avg_delta = (improvement / (n_records - 1)) if n_records > 1 and not np.isnan(improvement) else np.nan

kpi_df = pd.DataFrame({
    "Metric": ["Records", "First angle (°)", "Last angle (°)", "Avg Δ / sample"],
    "Value": [n_records,
              f"{first_angle:.2f}" if not np.isnan(first_angle) else "N/A",
              f"{last_angle:.2f}" if not np.isnan(last_angle) else "N/A",
              f"{avg_delta:.3f}" if not np.isnan(avg_delta) else "N/A"]
})

# Latest MMT assignment table
assignments = load_assignments()
if not assignments.empty and selected in assignments['Patient_ID'].values:
    last_assign = assignments[assignments['Patient_ID'] == selected].sort_values('timestamp', ascending=False).iloc[0]
    assign_df = pd.DataFrame({
        "Field": ["Assigned MMT", "Assigned by", "Assigned on", "Details"],
        "Value": [int(last_assign['Assigned_MMT']), last_assign.get('assigned_by','-'), last_assign['timestamp'],
                  f"cycles={last_assign.get('cycles','-')}, break={last_assign.get('break_min','-')}, sets={last_assign.get('sets','-')}, speed={last_assign.get('speed_percent','-')}"]
    })
else:
    assign_df = pd.DataFrame({"Field": ["Latest MMT Assignment"], "Value": ["No assignment found"]})

col_left, col_right = st.columns([1,1])
with col_left:
    st.table(kpi_df.set_index("Metric"))
with col_right:
    st.subheader("Latest Assignment")
    st.table(assign_df.set_index("Field"))

# Graph next
fig, patient_df = build_plot(patient_df)
st.plotly_chart(fig, use_container_width=True)

# Current status: show latest condition and counts table
if not np.isnan(last_angle):
    current = rule_label(last_angle)
else:
    current = "Unknown"
counts = patient_df['Measured_Angle_Flex_Sensor'].apply(lambda a: rule_label(a)).value_counts().reindex(
    ["Recovered", "Keep under observation", "Not yet recovered", "Unknown"], fill_value=0)

st.markdown("### Current status")
st.write(f"**Current condition:** {current}")
st.table(pd.DataFrame({"Condition": counts.index, "Count": counts.values}).set_index("Condition"))

# Sidebar: MMT assignment and reference (unchanged)
ref_df = pd.DataFrame([
    {"MMT": 0, "cycles": 1, "break_min": "More than 3 min", "sets": 1, "speed%": "0–20%"},
    {"MMT": 1, "cycles": 2, "break_min": "2-3", "sets": 2, "speed%": "21–40%"},
    {"MMT": 2, "cycles": 3, "break_min": "1-2", "sets": 3, "speed%": "41–55%"},
    {"MMT": 3, "cycles": 4, "break_min": "30-60 sec", "sets": 4, "speed%": "56–70%"},
    {"MMT": 4, "cycles": "5+", "break_min": "Less than 30 sec", "sets": "5+", "speed%": "71–90%"},
    {"MMT": 5, "cycles": "6+", "break_min": "—", "sets": "6+", "speed%": "91–120%"},
])

with st.sidebar:
    st.header("MMT Assignment")
    # suggested MMT from last angle
    if np.isnan(last_angle):
        suggested = None
    else:
        sp = max(0, min(120, last_angle))
        def sug(sp):
            if sp <= 20: return 0
            if sp <= 40: return 1
            if sp <= 55: return 2
            if sp <= 70: return 3
            if sp <= 90: return 4
            return 5
        suggested = sug(sp)
    st.metric("Suggested MMT", suggested if suggested is not None else "N/A")
    if st.button("Check reference table"):
        st.dataframe(ref_df)

    st.markdown("Assign / Save MMT")
    assigned_by = st.text_input("Assigned by", value="Clinician")
    assigned_mmt = st.selectbox("Assigned MMT", [0,1,2,3,4,5], index=int(suggested) if suggested is not None else 0)
    sel = ref_df[ref_df['MMT']==assigned_mmt].iloc[0]
    cycles = st.text_input("cycles", value=str(sel['cycles']))
    break_min = st.text_input("break(min)", value=str(sel['break_min']))
    sets = st.text_input("sets", value=str(sel['sets']))
    speed_pc = st.text_input("speed%", value=str(sel['speed%']))
    notes = st.text_area("Notes (optional)", value="")

    if st.button("Save assignment"):
        rec = {
            "timestamp": datetime.now().isoformat(),
            "Patient_ID": selected,
            "Assigned_MMT": int(assigned_mmt),
            "Suggested_MMT": int(suggested) if suggested is not None else "",
            "cycles": cycles,
            "break_min": break_min,
            "sets": sets,
            "speed_percent": speed_pc,
            "notes": notes,
            "assigned_by": assigned_by
        }
        save_assignment(rec)
        st.success("Assignment saved.")

    st.markdown("---")
    st.markdown("Assignment history (latest 10)")
    hist = load_assignments()
    if not hist.empty and selected in hist['Patient_ID'].values:
        st.dataframe(hist[hist['Patient_ID']==selected].sort_values('timestamp', ascending=False).head(10))
    else:
        st.write("No assignments yet.")

# Report section & generation
st.markdown("## Report")
st.markdown("The report contains clinic header, doctor info, patient KPIs, latest assignment, graph and recent readings.")

st.subheader("Recent readings (latest 10)")
st.dataframe(patient_df.sort_values('Timestamp', ascending=False).head(10)[['Timestamp','R_flex_Ohms','Measured_Angle_Flex_Sensor','rule']].reset_index(drop=True))

if st.button("Generate PDF report (clinic format)"):
    try:
        pdf_buf = create_pdf_report(selected, patient_df, fig)
        st.success("PDF generated — download below")
        st.download_button("Download PDF", data=pdf_buf.getvalue(), file_name=f"{selected}_mayuri_report.pdf", mime="application/pdf")
    except Exception as e:
        st.error(f"Failed to create PDF: {e}")

st.caption("Layout: KPIs → Latest Assignment → Graph → Current Status → Report. Sidebar: MMT assignment controls & reference table.")
