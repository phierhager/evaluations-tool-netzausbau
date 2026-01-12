import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import io

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="PhyLFlex Auditor (D2.3)",
    page_icon="⚖️",
    layout="wide"
)

# --- CONFIGURATION ---
RESULTS_DIR = "simulation_results"

# ==========================================
# 1. FILE & METADATA HANDLER
# ==========================================

def parse_result_file(filepath):
    """
    Reads a file. Separates the Metadata Header (lines starting with #)
    from the CSV Data.
    """
    metadata = {}
    csv_buffer = io.StringIO()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_metadata = False
    for line in lines:
        # Parse Metadata Section
        if line.strip() == "# METADATA_START":
            in_metadata = True
            continue
        if line.strip() == "# METADATA_END":
            in_metadata = False
            continue
            
        if in_metadata:
            # Parse "Key: Value" lines
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip().replace("#", "").strip()] = val.strip()
        else:
            # Append everything else to CSV buffer
            csv_buffer.write(line)
            
    # Load CSV data from buffer
    csv_buffer.seek(0)
    try:
        df = pd.read_csv(csv_buffer)
        # Parse Timestamp if it exists
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    except pd.errors.EmptyDataError:
        df = pd.DataFrame() # Handle empty files
        
    return metadata, df

def get_available_results():
    """Scans the directory and returns a list of file info dicts."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv")]
    results_list = []
    
    for f in files:
        path = os.path.join(RESULTS_DIR, f)
        meta, _ = parse_result_file(path) # Peek at metadata only
        
        # Create a display name based on Metadata, fallback to filename
        display_name = meta.get("Scenario", f) + " - " + meta.get("Strategy", "")
        if not display_name.strip(" - "):
            display_name = f.replace(".csv", "").replace("_", " ").title()
            
        results_list.append({
            "filename": f,
            "path": path,
            "display_name": display_name,
            "metadata": meta
        })
        
    return results_list

# ==========================================
# 2. EVALUATION LOGIC (D2.3)
# ==========================================

def evaluate_economics(df, metadata):
    """
    Calculates CAPEX and OPEX based on the raw physical data.
    """
    stats = {
        "min_voltage": df["Min Voltage"].min() if "Min Voltage" in df else 230,
        "max_trafo": df["Trafo Load"].max() if "Trafo Load" in df else 0,
        "total_curtailment": df["Curtailed Energy"].sum() if "Curtailed Energy" in df else 0,
        "intervention_count": (df["Curtailed Energy"] > 0).sum() if "Curtailed Energy" in df else 0
    }
    
    # --- LOGIC: Grid Expansion (CAPEX) ---
    # Trigger: Voltage < 207V or Trafo > 100%
    needs_expansion = (stats["min_voltage"] < 207) or (stats["max_trafo"] > 100)
    
    capex = 0
    if needs_expansion:
        # Cost Assumptions (could be moved to sidebar)
        cable_cost = 80000 # €/km (Assumption for calculation)
        trafo_cost = 25000 # €
        capex = cable_cost * 1.5 + trafo_cost # Dummy distance
        
    # --- LOGIC: Intervention Costs (OPEX) ---
    # Cost: 0.10 € per kWh curtailed (Compensation)
    opex = stats["total_curtailment"] * 0.10 * 20 * 365 # 20 Years projection
    
    return stats, capex, opex, needs_expansion

# ==========================================
# 3. USER INTERFACE
# ==========================================

def main():
    # --- SIDEBAR: FILE SELECTOR ---
    st.sidebar.header("📂 Result Browser")
    
    available_runs = get_available_results()
    
    if not available_runs:
        st.error(f"No result files found in `{RESULTS_DIR}/`.")
        st.info("Please run the AP3 Simulation Grid to generate CSV files.")
        st.stop()
        
    # Dropdown Menu
    selected_option = st.sidebar.selectbox(
        "Select Simulation Run",
        options=available_runs,
        format_func=lambda x: x["display_name"]
    )
    
    # Load Full Data for Selection
    meta, df = parse_result_file(selected_option["path"])
    
    # --- HEADER: CONTEXT ---
    st.title("PhyLFlex Grid Auditor")
    st.markdown("### 📋 Context Report")
    
    # Display Metadata nicely
    col1, col2, col3 = st.columns(3)
    col1.info(f"**Scenario:**\n{meta.get('Scenario', 'Unknown')}")
    col2.info(f"**Strategy:**\n{meta.get('Strategy', 'Unknown')}")
    col3.info(f"**Hardware:**\n{meta.get('Hardware', 'Unknown')}")

    if df.empty:
        st.warning("Selected file contains no data rows.")
        st.stop()

    # --- CALCULATIONS ---
    stats, capex, opex, failed = evaluate_economics(df, meta)
    
    # --- DASHBOARD: KPIS ---
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    
    k1.metric("Min Voltage", f"{stats['min_voltage']:.1f} V", 
              delta=f"{stats['min_voltage']-207:.1f} V", 
              delta_color="normal" if stats['min_voltage'] >= 207 else "inverse")
              
    k2.metric("Trafo Load", f"{stats['max_trafo']:.0f} %",
              delta_color="inverse" if stats['max_trafo'] > 100 else "normal")
              
    k3.metric("Curtailment (Energy)", f"{stats['total_curtailment']:.1f} kWh",
              help="Total energy lost due to §14a interventions")
              
    k4.metric("Intervention Count", f"{stats['intervention_count']} Events",
              help="Number of 15-min intervals where dimming occurred")
              
    # --- DECISION MATRIX ---
    st.subheader("💰 Economic Evaluation (20 Years)")
    
    c_col1, c_col2 = st.columns([1, 2])
    
    with c_col1:
        if failed:
            st.error("### ❌ Grid Failure")
            st.markdown("Software flexibility was **insufficient**.")
            st.markdown("**Physical Reinforcement Required.**")
        elif capex == 0 and opex > 0:
            st.warning("### ⚠️ Intervention Heavy")
            st.markdown("Grid is stable, but relies on **frequent curtailment**.")
        else:
            st.success("### ✅ Optimal Operation")
            st.markdown("Grid stable with minimal intervention.")
            
    with c_col2:
        # Cost Stack Bar
        fig_cost = go.Figure(data=[
            go.Bar(name='Grid Expansion (CAPEX)', x=['Total Cost'], y=[capex], marker_color='#EF553B'),
            go.Bar(name='Curtailment Compensation (OPEX)', x=['Total Cost'], y=[opex], marker_color='#FFA15A')
        ])
        fig_cost.update_layout(barmode='stack', height=250, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_cost, use_container_width=True)

    # --- DETAILED CHARTS ---
    st.subheader("📈 Time Series Inspection")
    
    tab1, tab2 = st.tabs(["Voltage & Physics", "Interventions"])
    
    with tab1:
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=df["Timestamp"], y=df["Min Voltage"], name="Grid Voltage", line=dict(color="#636EFA")))
        fig_v.add_hline(y=207, line_dash="dash", line_color="red", annotation_text="Limit (207V)")
        st.plotly_chart(fig_v, use_container_width=True)
        
    with tab2:
        if "Curtailed Energy" in df.columns:
            fig_c = go.Figure()
            # Bar chart for interventions
            fig_c.add_trace(go.Bar(x=df["Timestamp"], y=df["Curtailed Energy"], name="Curtailed kWh", marker_color="orange"))
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("No Curtailment data in file.")

if __name__ == "__main__":
    main()