import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- SEITE KONFIGURIEREN ---
st.set_page_config(
    page_title="PhyLFlex Professional Evaluation Tool (AP 2.2)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. PHYSIKALISCHE MODELLE & KLASSEN
# ==========================================

class CableType:
    """
    Definiert physikalische Eigenschaften von Erdkabeln (NAYY).
    Werte basieren auf Standard-Kabeldatenbüchern.
    """
    def __init__(self, name, r_ohm_km, x_ohm_km, i_max_a):
        self.name = name
        self.r = r_ohm_km  # Wirkwiderstand
        self.x = x_ohm_km  # Induktiver Blindwiderstand
        self.i_max = i_max_a # Thermische Stromgrenze

# Kabel-Bibliothek
CABLES = {
    "Altbestand (NAYY 4x50)": CableType("NAYY 4x50", 0.641, 0.083, 142),
    "Standard (NAYY 4x150)": CableType("NAYY 4x150", 0.206, 0.080, 270),
    "Verstärkt (NAYY 4x240)": CableType("NAYY 4x240", 0.125, 0.079, 355)
}

class GridScenario:
    def __init__(self, trafo_kva, cable_type_key, length_km, households):
        self.trafo_limit_kva = trafo_kva
        self.cable = CABLES[cable_type_key]
        self.length_km = length_km
        self.households = households
        self.u_n = 400.0 # Niederspannung Phase-Phase
        
        # Impedanz der Strecke zum kritischen Knoten (Worst Case: Strangende)
        self.R_line = self.cable.r * length_km
        self.X_line = self.cable.x * length_km

    def calculate_voltage_drop(self, p_kw, q_kvar):
        """
        Berechnet den Spannungsabfall über die Längsimpedanz.
        Formel (vereinfacht für Längsspannungsfall): dU approx (P*R + Q*X) / Un
        Rückgabe: Spannung am Knoten in Volt.
        """
        # P in Watt, Q in Var
        p_w = p_kw * 1000
        q_var = q_kvar * 1000
        
        # Spannungsfall delta_u
        delta_u = (p_w * self.R_line + q_var * self.X_line) / self.u_n
        
        # Strangspannung (L-N) ist eigentlich 230V, wir rechnen hier auf 400V Ebene 
        # und skalieren für die Ausgabe auf 230V (Strang)
        u_node_400 = self.u_n - delta_u
        
        # Umrechnung auf Phasenspannung (L-N) für den Hausanschluss
        u_node_230 = u_node_400 / np.sqrt(3)
        return u_node_230

# ==========================================
# 2. SIMULATIONS-KERN (ZEITREIHEN)
# ==========================================

class SimulationEngine:
    def __init__(self, grid: GridScenario, params):
        self.grid = grid
        self.p = params
        self.steps = 96 # 15 min Auflösung (24h)
        self.dt = 0.25  # Stunden pro Step
        
    def generate_base_profiles(self):
        """Erzeugt Wetter, Preise und Grundlasten."""
        np.random.seed(42)
        t = np.linspace(0, 24, self.steps)
        
        # --- PREIS (§41a) ---
        # Dynamischer Tarif: Teuer Morgens/Abends, Günstig Nachts/Mittags
        # Zufallskomponente simulierte Volatilität
        price = 30 + 10 * np.sin((t-7)*np.pi/12) - 15 * np.exp(-(t-13)**2/10)
        price += np.random.normal(0, 2, self.steps)
        
        # --- SONNE (PV) ---
        pv_curve = np.maximum(0, np.sin((t-6)*np.pi/14))
        # Bewölkung simulieren (Rauschen)
        pv_curve *= np.random.uniform(0.8, 1.0, self.steps)
        
        # --- LAST (H0) ---
        # Grundlast Haushalt (ohne EV/WP)
        baseload = (np.sin((t-6)*np.pi/12)**2 * 0.4 + 0.15) 
        
        return t, price, pv_curve, baseload

    def run(self):
        t, price, pv_curve, baseload_curve = self.generate_base_profiles()
        
        # Ergebnisse Container
        results = {
            "Time": t,
            "Price": price,
            "Load_Total": [],
            "Voltage": [],
            "Trafo_Load_Pct": [],
            "Curtailment_kW": [],
            "EV_SoC": [], # State of Charge Durchschnitt
        }
        
        # Aggregierte Parameter
        n_ev = int(self.grid.households * (self.p['ev_share']/100))
        n_pv = int(self.grid.households * (self.p['pv_share']/100))
        n_gems = int(n_ev * (self.p['gems_adoption']/100)) # GEMS nur relevant für steuerbare Lasten (EV)
        n_dumb = n_ev - n_gems
        
        # EV Simulation State
        # Wir simulieren eine "Große Batterie" als Aggregat aller EVs
        ev_battery_cap_total = n_ev * 50 # 50kWh pro Auto
        ev_soc_current = ev_battery_cap_total * 0.4 # Start bei 40%
        
        # Wann kommen die Autos an? (Rush Hour 16:00 - 19:00)
        arrival_profile = np.exp(-(t-17.5)**2/4) 
        # Normalize arrival to represent plugged-in capacity factor
        connected_factor = np.clip(arrival_profile / arrival_profile.max(), 0.1, 1.0)
        
        # GEMS Strategie-Logik Vorbereitung
        # Wir berechnen den optimalen Ladeplan für GEMS Autos im Voraus (Day-Ahead Optimierung)
        # Strategie: Wenn Preis niedrig, lade viel.
        if self.p['strategy'] == "Markt (§41a)":
            # Je billiger, desto höher der Target-Power-Factor
            gems_intent_profile = np.interp(price, (price.min(), price.max()), (1.0, 0.0)) 
        elif self.p['strategy'] == "Netzdienlich (GEMS)":
            # Antizyklisch zur Grundlast und PV
            net_base = baseload_curve * self.grid.households - (pv_curve * n_pv * 8)
            # Ziel: Flat line. Wo net_base hoch ist, lade wenig.
            gems_intent_profile = np.interp(net_base, (net_base.min(), net_base.max()), (1.0, 0.0))
        else:
            # Dummes Laden
            gems_intent_profile = connected_factor # Laden sobald angesteckt
            
        
        # --- ZEITSCHRITT SIMULATION ---
        for i in range(self.steps):
            
            # 1. BASIS WERTE
            p_base_total = baseload_curve[i] * self.grid.households
            p_pv_total = pv_curve[i] * n_pv * 8 # 8 kWp Schnitt
            
            # 2. EV LADUNG (UNGESTEUERT / DUMM)
            # Dumme Autos laden einfach wenn sie da sind (connected_factor) mit 11kW
            # Bis voll (vereinfacht: wir nehmen an sie laden immer nachmittags nach)
            p_ev_dumb = n_dumb * 11.0 * connected_factor[i]
            
            # 3. EV LADUNG (GEMS / SMART)
            # GEMS Autos laden nach Strategie, aber limitiert durch Anschlussleistung (11kW)
            p_ev_smart = n_gems * 11.0 * gems_intent_profile[i] * connected_factor[i]
            
            # SoC Update (Energie in die Autos)
            # Limitieren: Wir können nicht mehr laden als Batteriekapazität
            total_charge_power = p_ev_dumb + p_ev_smart
            max_charge_energy = ev_battery_cap_total - ev_soc_current
            
            if total_charge_power * self.dt > max_charge_energy:
                total_charge_power = max_charge_energy / self.dt
                
            ev_soc_current += total_charge_power * self.dt
            
            # 4. NETZLAST VOR REGLUNG
            # Last = Base + EVs - PV
            p_load_net = p_base_total + total_charge_power - p_pv_total
            
            # Blindleistungsschätzung (cos phi 0.95 ind. für Haushalt, 1.0 für PV/EV)
            # Q = P * tan(acos(phi))
            q_load_net = p_base_total * 0.33 # ca cos phi 0.95
            
            # 5. PHYSIK CHECK (SPANNUNG & TRAFO)
            u_node = self.grid.calculate_voltage_drop(p_load_net, q_load_net)
            trafo_load_kva = np.sqrt(p_load_net**2 + q_load_net**2)
            
            # 6. §14a EINGRIFF (CURTAILMENT)
            curtailment = 0
            
            # Kriterien für Eingriff
            u_min = 230 * 0.9 # -10% Limit (207V)
            trafo_overload = trafo_load_kva > self.grid.trafo_limit_kva
            voltage_violation = u_node < u_min
            
            if (trafo_overload or voltage_violation) and p_load_net > 0:
                # Wir müssen dimmen!
                # VNB sendet Dimm-Signal.
                # Max erlaubte Last berechnen
                
                # A) Durch Trafo limitiert
                p_max_trafo = np.sqrt(self.grid.trafo_limit_kva**2 - q_load_net**2)
                
                # B) Durch Spannung limitiert (Auflösen der Spannungsformel nach P)
                # delta_u_max = Un_phase - U_min
                # delta_u_line = (Un_line - U_min_line * sqrt(3)) ... vereinfacht:
                # Wir nutzen die berechnete Node Spannung und iterieren linear zurück
                # delta_u_allow = 23V
                # P_allow = (delta_u_allow * Un * sqrt(3) - Q*X) / R  <-- Näherung
                delta_u_max_v = (230 - u_min) * np.sqrt(3) # auf 400V Basis
                p_max_volt = ((delta_u_max_v * 400) - (q_load_net * 1000 * self.grid.X_line)) / (self.grid.R_line * 1000)
                
                p_limit = min(p_max_trafo, p_max_volt)
                
                if p_load_net > p_limit:
                    needed_reduction = p_load_net - p_limit
                    # Wir können nur die EVs dimmen (SteuVE), nicht den Haushalt
                    # Max Reduktion = EV_Power - (Anzahl_Autos * 4.2 kW Mindestleistung)
                    # Wenn wir mehr reduzieren müssten, bricht das Netz zusammen (Blackout Sim)
                    
                    min_guaranteed_power = (n_ev * 4.2)
                    dimmable_power = max(0, total_charge_power - min_guaranteed_power)
                    
                    actual_reduction = min(needed_reduction, dimmable_power)
                    
                    curtailment = actual_reduction
                    p_load_net -= curtailment
                    
                    # SoC Korrektur (Energie wurde nicht geladen)
                    ev_soc_current -= curtailment * self.dt
                    
                    # Physik neu berechnen mit gedimmter Last
                    u_node = self.grid.calculate_voltage_drop(p_load_net, q_load_net)
                    trafo_load_kva = np.sqrt(p_load_net**2 + q_load_net**2)

            # 7. DATEN SPEICHERN
            results["Load_Total"].append(p_load_net)
            results["Voltage"].append(u_node)
            results["Trafo_Load_Pct"].append((trafo_load_kva / self.grid.trafo_limit_kva)*100)
            results["Curtailment_kW"].append(curtailment)
            results["EV_SoC"].append(ev_soc_current / ev_battery_cap_total * 100)

        return pd.DataFrame(results)

# ==========================================
# 3. KOSTEN-KALKULATOR (AP 2.2)
# ==========================================

def calculate_economics(df, grid_sc, params):
    # 1. Analyse der Netzprobleme
    sum_curtailment = df["Curtailment_kW"].sum() * 0.25 # kWh
    
    # Spannungs-Extrema
    min_voltage = df["Voltage"].min()
    max_voltage = df["Voltage"].max() # <--- NEU: Das Maximum prüfen
    
    max_trafo = df["Trafo_Load_Pct"].max()
    
    # 2. Status Bewertung
    grid_upgrade_needed = False
    grid_capex = 0
    problem_reason = ""
    
    # CHECK: Unterspannung (-10%) ODER Überspannung (+10%) ODER Trafo-Überlast
    if min_voltage < 207:
        grid_upgrade_needed = True
        problem_reason = "Unterspannung (Last)"
    elif max_voltage > 253: # <--- NEU: Überspannungs-Check (230V + 10%)
        grid_upgrade_needed = True
        problem_reason = "Überspannung (PV)"
    elif max_trafo > 100:
        grid_upgrade_needed = True
        problem_reason = "Trafo Überlast"

    # Kostenberechnung (wie gehabt)
    if grid_upgrade_needed:
        cable_cost = grid_sc.length_km * 80000 
        trafo_cost = 25000 
        grid_capex = cable_cost + trafo_cost

    gems_unit_cost = 400 
    gems_capex = (grid_sc.households * (params['gems_adoption']/100)) * gems_unit_cost
    opex_curtailment = sum_curtailment * 365 * 20 * 1.00 
    
    return {
        "Grid_CAPEX": grid_capex,
        "GEMS_CAPEX": gems_capex,
        "OPEX_Loss": opex_curtailment,
        "Total_Cost": grid_capex + gems_capex + opex_curtailment,
        "Needed_Grid": grid_upgrade_needed,
        "Reason": problem_reason, # Grund für den Fehler speichern
        "Stats": (min_voltage, max_voltage, max_trafo, sum_curtailment)
    }

# ==========================================
# 4. FRONTEND (STREAMLIT UI)
# ==========================================

def main():
    # --- SIDEBAR EINGABEN ---
    st.sidebar.title("⚙️ Simulation Setup")
    
    st.sidebar.subheader("Netztopologie")
    grid_type = st.sidebar.selectbox("Siedlungsstruktur", ["Ländlich (Dorf)", "Suburban (Vorstadt)", "Urban (Stadt)"])
    
    if grid_type == "Ländlich (Dorf)":
        h_count = 60
        g_len = 2.5 # km
        g_trafo = 250
        c_type = "Altbestand (NAYY 4x50)"
    elif grid_type == "Suburban (Vorstadt)":
        h_count = 120
        g_len = 1.0
        g_trafo = 400
        c_type = "Standard (NAYY 4x150)"
    else:
        h_count = 200
        g_len = 0.5
        g_trafo = 630
        c_type = "Verstärkt (NAYY 4x240)"
        
    # Manuelle Overrides (Advanced)
    with st.sidebar.expander("Erweiterte Netz-Parameter"):
        g_len = st.number_input("Stranglänge (km)", 0.1, 10.0, g_len)
        g_trafo = st.number_input("Trafo Limit (kVA)", 100, 1000, g_trafo)
        c_type = st.selectbox("Kabeltyp", list(CABLES.keys()), index=1)

    scenario = GridScenario(g_trafo, c_type, g_len, h_count)

    st.sidebar.subheader("Prosumer & Markt")
    pv_share = st.sidebar.slider("PV Durchdringung (%)", 0, 100, 50)
    ev_share = st.sidebar.slider("EV Durchdringung (%)", 0, 100, 30)
    
    st.sidebar.subheader("GEMS & Regulierung")
    gems_adopt = st.sidebar.slider("GEMS Verbreitung (%)", 0, 100, 0, help="Anteil der EVs mit intelligentem Lademanagement")
    strategy = st.sidebar.radio("GEMS Strategie", ["Ungesteuert", "Markt (§41a)", "Netzdienlich (GEMS)"])
    
    params = {
        "pv_share": pv_share,
        "ev_share": ev_share,
        "gems_adoption": gems_adopt,
        "strategy": strategy
    }

    # --- SIMULATION STARTEN ---
    engine = SimulationEngine(scenario, params)
    df = engine.run()
    costs = calculate_economics(df, scenario, params)

    # --- DASHBOARD OUTPUT ---
    st.title("PhyLFlex Eval-Tool (AP 2.2)")
    st.markdown(f"**Szenario:** {grid_type} | **Kabel:** {c_type} ({g_len}km) | **Strategie:** {strategy}")
    
    # KPI ROW
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    u_min = costs["Stats"][0]
    load_max = costs["Stats"][1]
    curtail_sum = costs["Stats"][2]
    
    kpi1.metric("Min. Spannung (Strang)", f"{u_min:.1f} V", delta=f"{u_min-207:.1f} V Puffer", delta_color="normal" if u_min > 207 else "inverse")
    kpi2.metric("Max. Trafo Last", f"{load_max:.0f} %", delta_color="inverse" if load_max > 100 else "normal")
    kpi3.metric("§14a Eingriffe (24h)", f"{curtail_sum:.1f} kWh", help="Abgeregelte Energie durch Notfallmaßnahmen")
    
    savings = 0
    if costs["Needed_Grid"]:
        status_msg = "❌ Netzausbau erforderlich!"
        color = "red"
    else:
        status_msg = "✅ Netz stabil"
        color = "green"
        
    kpi4.markdown(f"<h3 style='color:{color};'>{status_msg}</h3>", unsafe_allow_html=True)

    # --- TABS ---
    tab_phys, tab_econ, tab_conflict = st.tabs(["⚡ Netzphysik & §14a", "💰 Ökonomische Bewertung", "⚖️ Konflikt-Analyse"])

    with tab_phys:
        # Plot 1: Last und Grenzen
        fig_load = go.Figure()
        fig_load.add_trace(go.Scatter(x=df["Time"], y=df["Load_Total"], name="Netzlast (kW)", fill='tozeroy', line=dict(color='#636EFA')))
        fig_load.add_hline(y=scenario.trafo_limit_kva, line_dash="dash", line_color="red", annotation_text="Trafo Limit")
        
        # Plot Curtailment Events
        curtail_events = df[df["Curtailment_kW"] > 0]
        fig_load.add_trace(go.Scatter(x=curtail_events["Time"], y=curtail_events["Load_Total"], mode="markers", marker=dict(color="red", symbol="x", size=10), name="§14a Dimmung"))
        
        fig_load.update_layout(title="Lastprofil am Ortsnetztransformator", xaxis_title="Tageszeit (h)", yaxis_title="Leistung (kW)")
        st.plotly_chart(fig_load, use_container_width=True)
        
        # Plot 2: Spannung
        fig_volt = go.Figure()
        fig_volt.add_trace(go.Scatter(x=df["Time"], y=df["Voltage"], name="Spannung (Kritischer Knoten)", line=dict(color='orange')))
        fig_volt.add_hline(y=207, line_dash="dot", line_color="red", annotation_text="-10% Limit (207V)")
        fig_volt.add_hline(y=230, line_dash="dot", line_color="green", annotation_text="Nennspannung (230V)")
        fig_volt.update_layout(title="Spannungsqualität (Power Quality)", yaxis_title="Spannung (V)", yaxis_range=[190, 245])
        st.plotly_chart(fig_volt, use_container_width=True)

    with tab_econ:
        st.subheader("Investitionsentscheidung (CAPEX/OPEX)")
        
        c1, c2 = st.columns(2)
        
        # Waterfall Chart
        fig_waterfall = go.Figure(go.Waterfall(
            name = "Kosten", orientation = "v",
            measure = ["relative", "relative", "relative", "total"],
            x = ["GEMS Invest", "Netz Ausbau", "Komfortverlust (OPEX)", "Gesamtkosten"],
            textposition = "outside",
            text = [f"{costs['GEMS_CAPEX']/1000:.0f}k", f"{costs['Grid_CAPEX']/1000:.0f}k", f"{costs['OPEX_Loss']/1000:.0f}k", f"{costs['Total_Cost']/1000:.0f}k"],
            y = [costs['GEMS_CAPEX'], costs['Grid_CAPEX'], costs['OPEX_Loss'], costs['Total_Cost']],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(title="Kostenstruktur Szenario (20 Jahre)", yaxis_title="Kosten (€)")
        c1.plotly_chart(fig_waterfall, use_container_width=True)
        
        c2.info(f"""
        **Analyse:**
        * **GEMS Strategie:** {strategy}
        * **Kosten Netzausbau:** {costs['Grid_CAPEX']:,.0f} €
        * **Kosten GEMS:** {costs['GEMS_CAPEX']:,.0f} €
        
        **Fazit:**
        {'GEMS verhindert den Netzausbau erfolgreich!' if costs['Grid_CAPEX'] == 0 and strategy != 'Ungesteuert' else 'Trotz Maßnahmen ist Netzausbau nötig oder GEMS ist nicht aktiv.'}
        """)

    with tab_conflict:
        st.subheader("Systemkonflikt: Preis vs. Netz")
        st.markdown("Visualisierung des Konflikts zwischen Marktanreiz (billiger Strom) und Netzstabilität.")
        
        fig_conf = make_subplots(specs=[[{"secondary_y": True}]])
        
        # EV Ladeverhalten
        # Wir berechnen das durchschnittliche Laden eines einzelnen Autos
        ev_profile = (df["Load_Total"] - (df["Price"]*0 + np.min(df["Load_Total"]))) # Näherung
        
        fig_conf.add_trace(go.Scatter(x=df["Time"], y=df["Load_Total"], name="Netzlast", fill='tozeroy'), secondary_y=False)
        fig_conf.add_trace(go.Scatter(x=df["Time"], y=df["Price"], name="Strompreis (Dynamisch)", line=dict(color='green', width=3)), secondary_y=True)
        
        fig_conf.update_yaxes(title_text="Last (kW)", secondary_y=False)
        fig_conf.update_yaxes(title_text="Preis (ct/kWh)", secondary_y=True)
        fig_conf.update_layout(title="Korrelation: Führt niedriger Preis zu Lastspitzen?")
        
        st.plotly_chart(fig_conf, use_container_width=True)
        
        st.warning("Hinweis: Wenn 'Markt (§41a)' gewählt ist, korrelieren die Lastspitzen oft mit den niedrigsten Preisen. Das provoziert §14a Eingriffe.")

if __name__ == "__main__":
    main()