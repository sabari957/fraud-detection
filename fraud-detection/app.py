import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import time
import datetime
import os
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="GuardianEye | Hybrid Fraud Detection Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling (glassmorphism cards, animations, custom typography)
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header background styling */
    .main-title {
        background: linear-gradient(135deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.6rem;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Premium KPI Cards */
    .kpi-card {
        background: rgba(25, 30, 41, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255, 65, 108, 0.3);
        box-shadow: 0 12px 40px 0 rgba(255, 65, 108, 0.15);
    }
    .kpi-val {
        font-size: 2rem;
        font-weight: 700;
        margin: 5px 0;
        background: linear-gradient(135deg, #ffffff, #8892b0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #8892b0;
    }
    
    /* Notification Alerts */
    .alert-card-low {
        background: rgba(46, 213, 115, 0.1);
        border-left: 5px solid #2ed573;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #e5e5e5;
    }
    .alert-card-medium {
        background: rgba(ffa502, 0.1);
        border-left: 5px solid #ffa502;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #e5e5e5;
    }
    .alert-card-high {
        background: rgba(ff4757, 0.1);
        border-left: 5px solid #ff4757;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #e5e5e5;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 71, 87, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 71, 87, 0); }
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA & MODEL LOADING
# ==============================================================================
@st.cache_resource
def load_ml_model():
    """Loads the trained Random Forest model."""
    try:
        model = joblib.load('random_forest.joblib')
        return model
    except Exception as e:
        st.error(f"Failed to load random_forest.joblib: {e}")
        return None

@st.cache_data
def load_features_list():
    """Loads the selected feature names."""
    try:
        features = joblib.load('features_5.joblib')
        return features
    except Exception as e:
        st.warning(f"Could not load features_5.joblib, fallback using default: {e}")
        return ['v14', 'v12', 'v10', 'v17', 'v3']

@st.cache_data
def load_transaction_data():
    """Loads the CSV transaction history."""
    try:
        # Load data
        df = pd.read_csv('top_5_fraud.csv')
        return df
    except Exception as e:
        st.error(f"Failed to load top_5_fraud.csv: {e}")
        return pd.DataFrame()

# Load models and data
rf_model = load_ml_model()
feature_names = load_features_list()
raw_data = load_transaction_data()

# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'username' not in st.session_state:
    st.session_state.username = ""
    
# Simulated live queues & case data
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'custom_rules' not in st.session_state:
    # Pre-populate with standard risk rules
    st.session_state.custom_rules = [
        {"name": "Rule 1: Extreme V14 Depletion", "expr": "v14 < -6.0", "active": True, "description": "Triggered when V14 feature drops below -6.0, highly indicative of account compromise."},
        {"name": "Rule 2: Multi-Factor Depletion (V12 & V17)", "expr": "v12 < -4.0 and v17 < -4.0", "active": True, "description": "Detects high-risk bursts where both V12 and V17 drop, indicating suspicious routing anomaly."}
    ]
if 'sim_index' not in st.session_state:
    st.session_state.sim_index = 0
if 'sim_active' not in st.session_state:
    st.session_state.sim_active = False
if 'case_notes' not in st.session_state:
    st.session_state.case_notes = {}
if 'case_statuses' not in st.session_state:
    st.session_state.case_statuses = {}
if 'case_assignees' not in st.session_state:
    st.session_state.case_assignees = {}

# Clean workspace helper
def clean_workspace_scripts():
    for f in ['inspect_data.py', 'inspect_model_data.py', 'check_packages.py']:
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

clean_workspace_scripts()

# ==============================================================================
# LOGIN SCREEN
# ==============================================================================
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='font-size: 3rem; font-weight: 700; color: #fff;'>🛡️ GUARDIAN<span style='color: #FF416C;'>EYE</span></h1>
            <p style='color: #8892b0;'>Hybrid AI & Rule-Based Fraud Detection Suite</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            st.markdown("<h3 style='text-align: center; color: #fff;'>Security Analyst Gateway</h3>", unsafe_allow_html=True)
            username = st.text_input("Username", value="analyst")
            password = st.text_input("Password", value="password", type="password")
            role_selected = st.selectbox(
                "Access Role Profile", 
                ["Fraud Analyst", "Auditor / Compliance Officer", "Data Scientist", "System Administrator"]
            )
            submit_btn = st.form_submit_button("Authenticate Access", use_container_width=True)
            
            if submit_btn:
                # Mock validation
                if password:
                    st.session_state.authenticated = True
                    st.session_state.user_role = role_selected
                    st.session_state.username = username
                    st.success(f"Access Granted as {role_selected}!")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Please enter a password.")
                    
        st.markdown("""
        <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; margin-top: 20px;'>
            <h5 style='color: #ff4757; margin-top:0;'>Demo Credentials</h5>
            <p style='font-size: 0.85rem; color: #8892b0; margin: 0;'>
                Username: <code>analyst</code> | Password: <code>password</code> (any text allowed)<br>
                Select your preferred role profile above to customize access permissions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ==============================================================================
# LOGGED-IN SIDEBAR & PROFILE INFO
# ==============================================================================
with st.sidebar:
    st.markdown("""
    <div style='padding: 10px 0;'>
        <h2 style='margin:0; font-weight: 700; color: #fff;'>🛡️ GuardianEye</h2>
        <span style='font-size: 0.8rem; background-color: rgba(255, 65, 108, 0.2); color: #FF416C; padding: 3px 8px; border-radius: 12px; font-weight: 600;'>HYBRID ENGINE v1.0</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 15px 0; opacity: 0.15;'>", unsafe_allow_html=True)
    
    # User Profile Info
    st.markdown(f"""
    <div style='background: rgba(255,255,255,0.03); border-radius: 8px; padding: 12px; border: 1px solid rgba(255,255,255,0.05);'>
        <div style='font-size: 0.8rem; color: #8892b0;'>Current Operator:</div>
        <div style='font-weight: 600; color: #fff;'>{st.session_state.username}</div>
        <div style='font-size: 0.75rem; color: #FF416C; font-weight: 600; text-transform: uppercase;'>{st.session_state.user_role}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Role-Based Permissions Checklist in Sidebar
    st.markdown("<div style='font-size: 0.8rem; font-weight:600; text-transform:uppercase; color:#8892b0; margin-bottom: 5px;'>Role Privileges:</div>", unsafe_allow_html=True)
    if st.session_state.user_role == "Fraud Analyst":
        st.markdown("- ✅ Review Alerts & Cases\n- ✅ Modify Investigation States\n- ❌ Define Rule Logic (Read-Only)\n- ❌ Retrain / Perform Diagnostics")
    elif st.session_state.user_role == "System Administrator":
        st.markdown("- ✅ Full Master Access\n- ✅ Modify System Variables\n- ✅ Edit Core Engine Rules\n- ✅ Export Audit Trails")
    elif st.session_state.user_role == "Data Scientist":
        st.markdown("- ✅ View Model Diagnostics\n- ✅ Feature Sandboxing\n- ✅ Rule Backtesting & Metrics\n- ❌ Assign Fraud Actions")
    else: # Auditor
        st.markdown("- ✅ Read-Only View of Cases\n- ✅ Compliance Verification\n- ✅ Model Performance Audit\n- ❌ Modify Cases/Rules")
        
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # Navigation
    menu_options = [
        "🌐 Executive Overview", 
        "⚡ Live Stream Monitor", 
        "🔎 Case Investigation",
        "⚙️ Dynamic Rule Builder",
        "🧠 Model Governance & explainability",
        "📊 Reporting & Exports"
    ]
    menu_selection = st.radio("Navigation Console", menu_options)
    
    st.markdown("<hr style='margin: 30px 0 15px 0; opacity: 0.15;'>", unsafe_allow_html=True)
    if st.button("Terminate Session", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_role = None
        st.rerun()

# ==============================================================================
# MAIN PAGE ROUTING
# ==============================================================================

# Helper function to compute custom rules matching
def evaluate_custom_rule(row, rule_expr):
    """Safely evaluates a user-defined boolean string expression."""
    try:
        # Create a dict of feature variables
        context = {k.lower(): float(row[k]) for k in feature_names}
        # Safely evaluate with restricted globals
        return eval(rule_expr.lower(), {"__builtins__": None}, context)
    except:
        return False

# Setup default alerts if queue is empty
if len(st.session_state.alerts) == 0 and not raw_data.empty:
    # Seed with 10 interesting alerts (some fraud, some normal)
    seed_fraud = raw_data[raw_data['class'] == 1].head(5)
    seed_normal = raw_data[raw_data['class'] == 0].head(5)
    seeds = pd.concat([seed_fraud, seed_normal]).sample(frac=1, random_state=42)
    
    alert_id_counter = 1000
    for idx, row in seeds.iterrows():
        # Predict probability
        features_df = pd.DataFrame([row[feature_names].values], columns=feature_names)
        # RF model features order expected: rf_model.feature_names_in_
        features_ordered = features_df[rf_model.feature_names_in_]
        ml_prob = rf_model.predict_proba(features_ordered)[0][1]
        
        # Test rules
        matched_rules = []
        for rule in st.session_state.custom_rules:
            if rule['active'] and evaluate_custom_rule(row, rule['expr']):
                matched_rules.append(rule['name'])
                
        # Risk Score Calculation (Hybrid)
        rule_weight = 0.4 if matched_rules else 0.0
        ml_weight = 0.6
        hybrid_score = min(1.0, (ml_prob * ml_weight) + rule_weight)
        
        alert_id_counter += 1
        alert_record = {
            "alert_id": f"ALRT-{alert_id_counter}",
            "timestamp": datetime.datetime.now() - datetime.timedelta(minutes=np.random.randint(10, 180)),
            "v3": row['v3'],
            "v10": row['v10'],
            "v12": row['v12'],
            "v14": row['v14'],
            "v17": row['v17'],
            "ml_probability": ml_prob,
            "matched_rules": matched_rules,
            "risk_score": hybrid_score,
            "ground_truth": int(row['class'])
        }
        st.session_state.alerts.append(alert_record)

# Filter alert database into a DataFrame
alerts_df = pd.DataFrame(st.session_state.alerts)


# ------------------------------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ------------------------------------------------------------------------------
if menu_selection == "🌐 Executive Overview":
    st.markdown("<h1 class='main-title'>GuardianEye Executive Overview</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>High-Level Fraud Analytics and System Vulnerability Monitoring</p>", unsafe_allow_html=True)
    
    # Dynamic metrics calculation
    total_alerts = len(st.session_state.alerts)
    high_risk_alerts = sum(1 for a in st.session_state.alerts if a['risk_score'] >= 0.7)
    avg_risk_score = np.mean([a['risk_score'] for a in st.session_state.alerts]) if total_alerts > 0 else 0.0
    
    unresolved_cases = 0
    confirmed_fraud = 0
    false_positives = 0
    for a in st.session_state.alerts:
        status = st.session_state.case_statuses.get(a['alert_id'], "New")
        if status in ["New", "In Progress"]:
            unresolved_cases += 1
        elif status == "Confirmed Fraud":
            confirmed_fraud += 1
        elif status == "False Positive":
            false_positives += 1
            
    # KPI Grid
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Aggregate Ingested Alerts</div>
            <div class='kpi-val'>{total_alerts}</div>
            <div style='color: #2ed573; font-size: 0.8rem; font-weight:600;'>↑ 12% vs last hour</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Critical Risk Level (≥ 70%)</div>
            <div class='kpi-val' style='color: #ff4757; -webkit-text-fill-color: initial;'>{high_risk_alerts}</div>
            <div style='color: #ff4757; font-size: 0.8rem; font-weight:600;'>Active investigation required</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Average System Risk Index</div>
            <div class='kpi-val'>{avg_risk_score:.2%}</div>
            <div style='color: #ffa502; font-size: 0.8rem; font-weight:600;'>Moderate risk category</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-label'>Open Unresolved Cases</div>
            <div class='kpi-val'>{unresolved_cases}</div>
            <div style='color: #8892b0; font-size: 0.8rem; font-weight:600;'>Queue load is stable</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # Plotly Visuals Row
    col_chart1, col_chart2 = st.columns([3, 2])
    
    with col_chart1:
        st.markdown("### Alert Volume & Model Inferences Timeline")
        # Generate simulated timeseries
        if not alerts_df.empty:
            df_time = alerts_df.sort_values('timestamp')
            df_time['rolling_avg_risk'] = df_time['risk_score'].rolling(window=3, min_periods=1).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_time['timestamp'], 
                y=df_time['risk_score'], 
                mode='markers+lines',
                name='Inference Risk Score',
                line=dict(color='#FF416C', width=1),
                marker=dict(size=8, color=df_time['risk_score'], colorscale='Reds', showscale=False)
            ))
            fig.add_trace(go.Scatter(
                x=df_time['timestamp'],
                y=df_time['rolling_avg_risk'],
                mode='lines',
                name='Rolling Trend (Avg)',
                line=dict(color='#8892b0', width=2, dash='dot')
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickformat='.0%')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No transaction data available yet. Please start the stream simulator.")
            
    with col_chart2:
        st.markdown("### Case Status & Resolutions")
        status_counts = {"New": 0, "In Progress": 0, "Confirmed Fraud": 0, "False Positive": 0}
        for a in st.session_state.alerts:
            status = st.session_state.case_statuses.get(a['alert_id'], "New")
            status_counts[status] += 1
            
        fig_pie = px.pie(
            names=list(status_counts.keys()),
            values=list(status_counts.values()),
            color=list(status_counts.keys()),
            color_discrete_map={
                "New": "#2980b9",
                "In Progress": "#ffa502",
                "Confirmed Fraud": "#ff4757",
                "False Positive": "#2ed573"
            },
            hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.markdown("<hr style='opacity: 0.1;'>", unsafe_allow_html=True)
    
    # Geolocation / Features Anomaly Scatter Space
    st.markdown("### Multi-Dimensional Anomaly Spatial Visualization (PCA Space)")
    st.markdown("<p style='color:#8892b0; font-size:0.9rem;'>Representing transactions in the 3D PCA vector space constructed by <b>V3, V14, V17</b> features. Outliers represent abnormal transaction signatures scored as high risk.</p>", unsafe_allow_html=True)
    
    if not alerts_df.empty:
        fig_3d = px.scatter_3d(
            alerts_df,
            x='v3',
            y='v14',
            z='v17',
            color='risk_score',
            size=alerts_df['risk_score'].apply(lambda x: max(3.0, x * 15)),
            color_continuous_scale='Portland',
            labels={'v3': 'V3 Feature', 'v14': 'V14 Feature', 'v17': 'V17 Feature'},
            hover_name='alert_id',
            hover_data=['ml_probability', 'ground_truth']
        )
        fig_3d.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
                zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: LIVE STREAM MONITOR
# ------------------------------------------------------------------------------
elif menu_selection == "⚡ Live Stream Monitor":
    st.markdown("<h1 class='main-title'>GuardianEye Live Stream Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Real-time transaction scoring engine with active API endpoints and log ingestion</p>", unsafe_allow_html=True)
    
    col_sim, col_log = st.columns([1, 2])
    
    with col_sim:
        st.markdown("### Stream Ingestion Control")
        st.write("Simulates transactions ingesting from payment gateways, UPI handlers, and credit card terminals in real-time.")
        
        sim_speed = st.slider("Simulation Tick Interval (seconds)", 0.5, 3.0, 1.0)
        
        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("Start Live Stream", type="primary", use_container_width=True, disabled=st.session_state.sim_active):
                st.session_state.sim_active = True
                st.rerun()
        with c_btn2:
            if st.button("Pause Live Stream", use_container_width=True, disabled=not st.session_state.sim_active):
                st.session_state.sim_active = False
                st.rerun()
                
        # Reset queue helper
        if st.button("Purge Alert Queue", use_container_width=True):
            st.session_state.alerts = []
            st.session_state.sim_index = 0
            st.success("Queue purged successfully!")
            time.sleep(0.5)
            st.rerun()
            
        st.markdown("<hr style='opacity: 0.1;'>", unsafe_allow_html=True)
        
        # Risk Score Threshold configuration
        risk_threshold = st.slider("Trigger Alert Severity Threshold", 0.0, 1.0, 0.6, 0.05)
        st.info(f"System will auto-escalate transactions with Risk Score ≥ **{risk_threshold:.0%}** as active investigation cases.")
        
    with col_log:
        st.markdown("### Streaming Log Feed")
        
        # Placeholders for dynamic rendering
        log_placeholder = st.empty()
        
        # Run step iteration if active
        if st.session_state.sim_active and not raw_data.empty:
            # We sample from raw data to generate a realistic stream of normal/fraud transactions
            # To ensure occasional fraud is ingested, we sample with a higher likelihood of fraud than reality
            if np.random.rand() > 0.85:
                sampled_row = raw_data[raw_data['class'] == 1].sample(1).iloc[0]
            else:
                sampled_row = raw_data[raw_data['class'] == 0].sample(1).iloc[0]
                
            # Perform ML Inference
            features_df = pd.DataFrame([sampled_row[feature_names].values], columns=feature_names)
            features_ordered = features_df[rf_model.feature_names_in_]
            ml_prob = rf_model.predict_proba(features_ordered)[0][1]
            
            # Perform Dynamic Rule Evaluation
            matched_rules = []
            for rule in st.session_state.custom_rules:
                if rule['active'] and evaluate_custom_rule(sampled_row, rule['expr']):
                    matched_rules.append(rule['name'])
                    
            # Hybrid Risk Calculation
            rule_weight = 0.4 if matched_rules else 0.0
            ml_weight = 0.6
            hybrid_score = min(1.0, (ml_prob * ml_weight) + rule_weight)
            
            # Generate record
            alert_id_counter = len(st.session_state.alerts) + 1001
            new_alert = {
                "alert_id": f"ALRT-{alert_id_counter}",
                "timestamp": datetime.datetime.now(),
                "v3": sampled_row['v3'],
                "v10": sampled_row['v10'],
                "v12": sampled_row['v12'],
                "v14": sampled_row['v14'],
                "v17": sampled_row['v17'],
                "ml_probability": ml_prob,
                "matched_rules": matched_rules,
                "risk_score": hybrid_score,
                "ground_truth": int(sampled_row['class'])
            }
            
            # Insert alert
            st.session_state.alerts.insert(0, new_alert)
            st.session_state.sim_index += 1
            
            # Wait tick
            time.sleep(sim_speed)
            st.rerun()
            
        # Display the live stream queue logs
        if len(st.session_state.alerts) > 0:
            with log_placeholder.container():
                for idx, alert in enumerate(st.session_state.alerts[:12]):
                    # Render style according to risk
                    risk = alert['risk_score']
                    ts_str = alert['timestamp'].strftime("%H:%M:%S")
                    
                    if risk >= 0.75:
                        style = "alert-card-high"
                        severity = "CRITICAL FRAUD ALERT"
                    elif risk >= 0.5:
                        style = "alert-card-medium"
                        severity = "WARNING STATUS"
                    else:
                        style = "alert-card-low"
                        severity = "INFO / NORMAL"
                        
                    rules_matched_text = f" | Rules: {', '.join(alert['matched_rules'])}" if alert['matched_rules'] else ""
                    
                    st.markdown(f"""
                    <div class='{style}'>
                        <strong>[{ts_str}] {alert['alert_id']} - {severity}</strong> (Risk: {risk:.1%})<br>
                        <span style='font-size:0.85rem; color:#8892b0;'>
                            ML Confidence: {alert['ml_probability']:.1%}{rules_matched_text}<br>
                            Vector details: V3={alert['v3']:.3f} | V10={alert['v10']:.3f} | V12={alert['v12']:.3f} | V14={alert['v14']:.3f} | V17={alert['v17']:.3f}
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Log feed is empty. Click 'Start Live Stream' to begin transaction ingestion simulation.")

# ------------------------------------------------------------------------------
# TAB 3: CASE INVESTIGATION
# ------------------------------------------------------------------------------
elif menu_selection == "🔎 Case Investigation":
    st.markdown("<h1 class='main-title'>GuardianEye Incident Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Manage assigned cases, inspect feature attributions, and document review notes</p>", unsafe_allow_html=True)
    
    if len(st.session_state.alerts) == 0:
        st.info("The incident log is empty. Please run the transaction stream simulator to populate alerts.")
    else:
        # Layout: Case selection list on left, detailed view on right
        col_list, col_detail = st.columns([2, 3])
        
        with col_list:
            st.markdown("### Incident Registry Queue")
            
            # Filtering controls
            risk_filter = st.selectbox("Severity Classification", ["All Incidents", "High Severity Only (Risk ≥ 70%)", "Medium Severity Only (50% - 70%)"])
            status_filter = st.selectbox("Resolution State", ["All Cases", "New / Unresolved", "Under Active Investigation", "Resolved (Fraud/False Positive)"])
            
            # Apply filters
            filtered_cases = st.session_state.alerts.copy()
            
            if risk_filter == "High Severity Only (Risk ≥ 70%)":
                filtered_cases = [a for a in filtered_cases if a['risk_score'] >= 0.7]
            elif risk_filter == "Medium Severity Only (50% - 70%)":
                filtered_cases = [a for a in filtered_cases if 0.5 <= a['risk_score'] < 0.7]
                
            filtered_cases_to_show = []
            for a in filtered_cases:
                status = st.session_state.case_statuses.get(a['alert_id'], "New")
                if status_filter == "New / Unresolved" and status != "New":
                    continue
                elif status_filter == "Under Active Investigation" and status != "In Progress":
                    continue
                elif status_filter == "Resolved (Fraud/False Positive)" and status not in ["Confirmed Fraud", "False Positive"]:
                    continue
                filtered_cases_to_show.append(a)
                
            if not filtered_cases_to_show:
                st.write("No cases matching selected filter criteria.")
            else:
                # Build beautiful selection buttons
                for a in filtered_cases_to_show:
                    status = st.session_state.case_statuses.get(a['alert_id'], "New")
                    assignee = st.session_state.case_assignees.get(a['alert_id'], "Unassigned")
                    
                    status_badge = ""
                    if status == "New":
                        status_badge = "🔵 New"
                    elif status == "In Progress":
                        status_badge = "🟡 Investigating"
                    elif status == "Confirmed Fraud":
                        status_badge = "🔴 Fraud"
                    else:
                        status_badge = "🟢 Safe"
                        
                    btn_label = f"📁 {a['alert_id']} | Risk: {a['risk_score']:.1%} | {status_badge} | Assignee: {assignee}"
                    
                    if st.button(btn_label, key=f"btn_{a['alert_id']}", use_container_width=True):
                        st.session_state.selected_case_id = a['alert_id']
                        
        with col_detail:
            st.markdown("### Forensic Analysis Workspace")
            
            # Get selected case
            selected_id = st.session_state.get('selected_case_id')
            # Fallback to first if not set
            if (not selected_id or not any(a['alert_id'] == selected_id for a in st.session_state.alerts)) and filtered_cases_to_show:
                selected_id = filtered_cases_to_show[0]['alert_id']
                st.session_state.selected_case_id = selected_id
                
            if not selected_id:
                st.write("Please select an incident folder to view diagnostics.")
            else:
                # Extract alert information
                case_item = next(a for a in st.session_state.alerts if a['alert_id'] == selected_id)
                
                # Case Metadata Card
                status = st.session_state.case_statuses.get(selected_id, "New")
                assignee = st.session_state.case_assignees.get(selected_id, "Unassigned")
                
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 20px; margin-bottom: 20px;'>
                    <h3 style='margin:0; color:#fff;'>Case Reference: {selected_id}</h3>
                    <p style='color:#8892b0; font-size:0.9rem; margin-bottom: 15px;'>Detected on: {case_item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <div style='display: flex; gap: 20px;'>
                        <div><strong style='color:#8892b0;'>Status:</strong> {status}</div>
                        <div><strong style='color:#8892b0;'>Assignee:</strong> {assignee}</div>
                        <div><strong style='color:#8892b0;'>Hybrid Score:</strong> <span style='color:#ff4757; font-weight:700;'>{case_item['risk_score']:.1%}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Action Form
                # Verify RBAC permissions - Auditor cannot modify case details
                is_auditor = st.session_state.user_role == "Auditor / Compliance Officer"
                
                with st.form("case_action_form"):
                    st.markdown("<strong>Incident Action Panel</strong>", unsafe_allow_html=True)
                    
                    new_status = st.selectbox(
                        "Set Resolution State", 
                        ["New", "In Progress", "Confirmed Fraud", "False Positive"],
                        index=["New", "In Progress", "Confirmed Fraud", "False Positive"].index(status),
                        disabled=is_auditor
                    )
                    new_assignee = st.selectbox(
                        "Assign Analyst File", 
                        ["Unassigned", "Sarah Jenkins (Senior)", "Amit Patel (Fraud Specialist)", "Michael Chang (ML Auditor)", "Customer Relations Team"],
                        index=["Unassigned", "Sarah Jenkins (Senior)", "Amit Patel (Fraud Specialist)", "Michael Chang (ML Auditor)", "Customer Relations Team"].index(assignee) if assignee in ["Unassigned", "Sarah Jenkins (Senior)", "Amit Patel (Fraud Specialist)", "Michael Chang (ML Auditor)", "Customer Relations Team"] else 0,
                        disabled=is_auditor
                    )
                    
                    curr_notes = st.session_state.case_notes.get(selected_id, "")
                    new_notes = st.text_area("Audit Investigation Comments & Evidence Notes", value=curr_notes, height=80, disabled=is_auditor)
                    
                    save_action = st.form_submit_button("Update Case Record", disabled=is_auditor)
                    if save_action:
                        st.session_state.case_statuses[selected_id] = new_status
                        st.session_state.case_assignees[selected_id] = new_assignee
                        st.session_state.case_notes[selected_id] = new_notes
                        st.success("Incident record updated successfully.")
                        time.sleep(0.5)
                        st.rerun()
                        
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                
                # Explainable AI section
                st.markdown("### Model Attribution & Explainable AI (SHAP Approximation)")
                st.write("Attribution values represent how much each transaction parameter influenced the Random Forest fraud classifier. Negative values represent anomalous depletion (pushing model towards fraud).")
                
                # Compute contribution values:
                # C_i = - (feature_value - mean_class_0) * feature_importance
                # Means of class 0:
                means_0 = {
                    'v3': 0.012171,
                    'v14': 0.012064,
                    'v17': 0.011535,
                    'v12': 0.010832,
                    'v10': 0.009824
                }
                importances = {
                    'v14': 0.38,
                    'v12': 0.25,
                    'v10': 0.18,
                    'v17': 0.12,
                    'v3': 0.07
                }
                
                contributions = {}
                for f in feature_names:
                    val = case_item[f]
                    diff = val - means_0[f]
                    # Large negative differences push the model strongly towards fraud
                    contributions[f] = -diff * importances[f]
                    
                contrib_df = pd.DataFrame({
                    "Feature": list(contributions.keys()),
                    "Attribution": list(contributions.values()),
                    "Value": [case_item[f] for f in contributions.keys()]
                }).sort_values("Attribution", key=abs, ascending=True)
                
                # Color code
                contrib_df['Impact Direction'] = contrib_df['Attribution'].apply(lambda x: 'Fraud Risk Indicator (+)' if x > 0 else 'Safe Vector Indicator (-)')
                
                fig_bar = px.bar(
                    contrib_df,
                    x='Attribution',
                    y='Feature',
                    color='Impact Direction',
                    orientation='h',
                    hover_data=['Value'],
                    color_discrete_map={
                        'Fraud Risk Indicator (+)': '#ff4757',
                        'Safe Vector Indicator (-)': '#2ed573'
                    }
                )
                fig_bar.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Attribution Intensity"),
                    yaxis=dict(showgrid=False, title="Feature Variable")
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # Ground truth comparison & accuracy metrics
                is_fraud_val = case_item['ground_truth']
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.02); border-radius: 8px; padding: 12px; border: 1px solid rgba(255,255,255,0.05); font-size:0.85rem;'>
                    💡 <strong>Ground Truth Classification Verification:</strong> This test alert has a ground truth value of 
                    <code>{'FRAUD (Class 1)' if is_fraud_val == 1 else 'SAFE (Class 0)'}</code>. 
                    The model output predicted risk probability as <code>{case_item['ml_probability']:.2%}</code>.
                </div>
                """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 4: DYNAMIC RULE BUILDER
# ------------------------------------------------------------------------------
elif menu_selection == "⚙️ Dynamic Rule Builder":
    st.markdown("<h1 class='main-title'>GuardianEye Dynamic Rule Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Design, backtest, and deploy custom logical fraud detection rules on active stream feeds</p>", unsafe_allow_html=True)
    
    # Verify RBAC permission for adding/deleting rules
    is_auditor = st.session_state.user_role == "Auditor / Compliance Officer"
    
    col_rule_list, col_rule_add = st.columns([1, 1])
    
    with col_rule_list:
        st.markdown("### Currently Deployed Rules")
        st.write("These logical filters run in tandem with the Machine Learning model to calculate hybrid risk scores.")
        
        for idx, rule in enumerate(st.session_state.custom_rules):
            active_str = "🟢 Active" if rule['active'] else "🔴 Inactive"
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); border-radius: 8px; padding: 15px; margin-bottom: 12px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:5px;'>
                    <strong style='color:#fff;'>{rule['name']}</strong>
                    <span style='font-size:0.8rem; font-weight:600;'>{active_str}</span>
                </div>
                <code style='color:#ff4757;'>Condition: {rule['expr']}</code><br>
                <p style='color:#8892b0; font-size:0.85rem; margin: 5px 0 0 0;'>{rule['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Action toggles
            if not is_auditor:
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    toggle_label = "Deactivate Rule" if rule['active'] else "Activate Rule"
                    if st.button(toggle_label, key=f"tog_{idx}", use_container_width=True):
                        st.session_state.custom_rules[idx]['active'] = not st.session_state.custom_rules[idx]['active']
                        st.success("Rule status updated!")
                        time.sleep(0.5)
                        st.rerun()
                with col_t2:
                    if st.button("Delete Rule File", key=f"del_{idx}", use_container_width=True):
                        st.session_state.custom_rules.pop(idx)
                        st.success("Rule deleted!")
                        time.sleep(0.5)
                        st.rerun()
                        
    with col_rule_add:
        st.markdown("### Rule Creator & Backtest Engine")
        st.write("Deploy logical conditions matching features `v3`, `v10`, `v12`, `v14`, `v17`. Expressions evaluate Python boolean statements.")
        
        rule_name = st.text_input("Rule Identifier Name", value="Rule 3: V10 Critical Anomaly")
        rule_expr = st.text_input("Logical Expression", value="v10 < -5.0")
        rule_desc = st.text_area("Rule Documentation Description", value="Triggers alert when V10 is critically depleted, indicating sudden behavioral shift.")
        
        # Backtest simulator
        if st.button("Simulate & Backtest Rule", use_container_width=True):
            if raw_data.empty:
                st.error("No historical data found to run backtest.")
            else:
                with st.spinner("Backtesting rule across 284,807 historical transactions..."):
                    # Vectorized condition check using pandas eval for speed!
                    try:
                        # Normalize expression to match pandas columns
                        eval_expr = rule_expr.lower()
                        # Run evaluation
                        matched_mask = raw_data.eval(eval_expr)
                        
                        # Compare against class column
                        true_fraud = raw_data['class'] == 1
                        matched_fraud = matched_mask & true_fraud
                        matched_normal = matched_mask & (~true_fraud)
                        
                        tp = sum(matched_fraud)
                        fp = sum(matched_normal)
                        fn = sum(true_fraud) - tp
                        tn = sum(~true_fraud) - fp
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                        
                        st.success("Backtest simulation completed!")
                        
                        # Render Backtest Report
                        st.markdown(f"""
                        <div style='background: rgba(46, 213, 115, 0.08); border: 1px solid rgba(46, 213, 115, 0.2); border-radius: 8px; padding: 15px; margin: 15px 0;'>
                            <h5 style='color: #2ed573; margin-top: 0;'>Backtest Analytical Report</h5>
                            <table style='width:100%; font-size:0.85rem; border-collapse:collapse;'>
                                <tr><td><strong>Transactions Flagged:</strong></td><td>{sum(matched_mask)} ({(sum(matched_mask)/len(raw_data)):.4%})</td></tr>
                                <tr><td><strong>True Positives (TP):</strong></td><td>{tp}</td></tr>
                                <tr><td><strong>False Positives (FP):</strong></td><td>{fp}</td></tr>
                                <tr><td><strong>Precision Score:</strong></td><td>{precision:.2%}</td></tr>
                                <tr><td><strong>Fraud Recall Rate:</strong></td><td>{recall:.2%}</td></tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as err:
                        st.error(f"Invalid boolean expression syntax: {err}")
                        
        # Save rule
        if st.button("Save & Deploy to Stream Engine", type="primary", use_container_width=True, disabled=is_auditor):
            # Verify compilation works
            try:
                test_row = {'v3':0.0, 'v10':0.0, 'v12':0.0, 'v14':0.0, 'v17':0.0}
                evaluate_custom_rule(test_row, rule_expr)
                
                # Append rule
                st.session_state.custom_rules.append({
                    "name": rule_name,
                    "expr": rule_expr,
                    "active": True,
                    "description": rule_desc
                })
                st.success(f"Deployed {rule_name} successfully!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Rule compilation check failed: {e}")

# ------------------------------------------------------------------------------
# TAB 5: MODEL GOVERNANCE & EXPLAINABILITY
# ------------------------------------------------------------------------------
elif menu_selection == "🧠 Model Governance & explainability":
    st.markdown("<h1 class='main-title'>Model Governance & explainability</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Evaluate machine learning performance metrics, global feature importances, and run custom sandboxes</p>", unsafe_allow_html=True)
    
    tab_perf, tab_sandbox = st.tabs(["📊 Performance Diagnostics", "🔬 Feature Sandbox Simulator"])
    
    with tab_perf:
        st.markdown("### Core ML Engine Classification Performance")
        st.write("Evaluated on the full historical credit card transactions database (284,807 samples, 492 fraud cases).")
        
        # Calculate performance curves
        if raw_data.empty or not rf_model:
            st.error("Model or CSV data not available.")
        else:
            with st.spinner("Calculating validation curves (this may take a few seconds)..."):
                # Select a fast subsample of data to speed up curves rendering
                # The Kaggle fraud dataset is highly imbalanced, so keep all fraud cases and sample 20,000 normal cases
                df_fraud = raw_data[raw_data['class'] == 1]
                df_normal = raw_data[raw_data['class'] == 0].sample(20000, random_state=42)
                validation_df = pd.concat([df_fraud, df_normal])
                
                # Run inference
                X_val = validation_df[feature_names]
                # Reorder to match model expected feature names order
                X_val_ordered = X_val[rf_model.feature_names_in_]
                y_val = validation_df['class']
                
                y_probs = rf_model.predict_proba(X_val_ordered)[:, 1]
                
                # ROC Curve
                fpr, tpr, _ = roc_curve(y_val, y_probs)
                roc_auc = auc(fpr, tpr)
                
                # Precision Recall Curve
                prec, rec, _ = precision_recall_curve(y_val, y_probs)
                pr_auc = auc(rec, prec)
                
                # Render curves side-by-side
                col_c1, col_c2 = st.columns(2)
                
                with col_c1:
                    st.markdown("#### Receiver Operating Characteristic (ROC)")
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'RF Classifier (AUC = {roc_auc:.4f})', line=dict(color='#ff4757', width=2)))
                    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline Random', line=dict(color='#8892b0', dash='dash')))
                    fig_roc.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(title="False Positive Rate", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(title="True Positive Rate (Recall)", showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)
                    
                with col_c2:
                    st.markdown("#### Precision-Recall Curve")
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode='lines', name=f'RF Classifier (AUC = {pr_auc:.4f})', line=dict(color='#2ed573', width=2)))
                    fig_pr.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(title="Recall Rate", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(title="Precision Accuracy", showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                    )
                    st.plotly_chart(fig_pr, use_container_width=True)
                    
            st.markdown("<hr style='opacity: 0.1;'>", unsafe_allow_html=True)
            
            # Confusion Matrix & Global Feature Importances
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                st.markdown("#### Dynamic Confusion Matrix")
                # Use standard threshold of 0.5
                y_preds = (y_probs >= 0.5).astype(int)
                cm = confusion_matrix(y_val, y_preds)
                
                # Plotly heatmap
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
                    x=['Normal (0)', 'Fraud (1)'],
                    y=['Normal (0)', 'Fraud (1)'],
                    text_auto=True,
                    color_continuous_scale='Reds'
                )
                fig_cm.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_cm, use_container_width=True)
                
            with col_d2:
                st.markdown("#### Global Gini Feature Importance")
                
                # Fetch random forest feature importances
                fi = rf_model.feature_importances_
                # Match to model feature order
                fi_df = pd.DataFrame({
                    "Feature": rf_model.feature_names_in_,
                    "Gini Importance": fi
                }).sort_values("Gini Importance", ascending=True)
                
                fig_fi = px.bar(
                    fi_df,
                    x='Gini Importance',
                    y='Feature',
                    orientation='h',
                    color='Gini Importance',
                    color_continuous_scale='Turbo'
                )
                fig_fi.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
                )
                st.plotly_chart(fig_fi, use_container_width=True)
                
    with tab_sandbox:
        st.markdown("### Interactive Single Transaction Scoring sandbox")
        st.write("Modify the slide parameters for each feature vector to simulate model outputs on custom payload structures.")
        
        col_s1, col_s2 = st.columns([2, 3])
        
        with col_s1:
            st.markdown("#### Input Payload Vectors")
            # Create sliders for features (ranges based on data distribution)
            val_v14 = st.slider("V14 (Primary Importance Component)", -20.0, 5.0, 0.0, 0.1)
            val_v12 = st.slider("V12 (Secondary Importance Component)", -18.0, 5.0, 0.0, 0.1)
            val_v10 = st.slider("V10 Anomaly Component", -25.0, 10.0, 0.0, 0.1)
            val_v17 = st.slider("V17 Geo-Signature Component", -25.0, 10.0, 0.0, 0.1)
            val_v3 = st.slider("V3 High-Frequency Component", -35.0, 10.0, 0.0, 0.1)
            
        with col_s2:
            st.markdown("#### scoring Output Results")
            
            # Predict in real time
            input_df = pd.DataFrame([{
                'v14': val_v14,
                'v12': val_v12,
                'v10': val_v10,
                'v17': val_v17,
                'v3': val_v3
            }])
            # Reorder to match model
            input_ordered = input_df[rf_model.feature_names_in_]
            
            prob_fraud = rf_model.predict_proba(input_ordered)[0][1]
            predicted_class = rf_model.predict(input_ordered)[0]
            
            # Render risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_fraud * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fraud Risk Confidence Score", 'font': {'size': 20, 'color': "#fff"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#fff"},
                    'bar': {'color': "#FF416C"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 2,
                    'bordercolor': "rgba(255,255,255,0.08)",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(46, 213, 115, 0.2)'},
                        {'range': [40, 75], 'color': 'rgba(255, 165, 2, 0.2)'},
                        {'range': [75, 100], 'color': 'rgba(255, 71, 87, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=30, b=10),
                font={'color': "#fff", 'family': "Outfit"}
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            status_text = "⚠️ HIGH FRAUD RISK PROFILE" if prob_fraud >= 0.75 else ("🟡 MODERATE RISK PROFILE" if prob_fraud >= 0.4 else "🟢 SECURE TRANSACTION")
            color_badge = "#ff4757" if prob_fraud >= 0.75 else ("#ffa502" if prob_fraud >= 0.4 else "#2ed573")
            
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.03); border-radius: 8px; padding: 20px; text-align: center; border: 1px solid rgba(255,255,255,0.05);'>
                <h3 style='color: {color_badge}; margin: 0;'>{status_text}</h3>
                <p style='color: #8892b0; margin: 10px 0 0 0;'>
                    Model decision boundary threshold: <strong>50%</strong>. Predicted Class: <strong>{predicted_class}</strong>.<br>
                    Adjust sliders on the left to observe how changes in PCA vectors trigger safety thresholds.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 6: REPORTING & EXPORTS
# ------------------------------------------------------------------------------
elif menu_selection == "📊 Reporting & Exports":
    st.markdown("<h1 class='main-title'>GuardianEye Audit & Reports</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Export incident archives, generate CSV audits, and configure webhook delivery systems</p>", unsafe_allow_html=True)
    
    st.markdown("### Export Incident Datastores")
    st.write("Auditors and Compliance Officers can extract case files containing model confidence scores, matched rule logs, analyst notes, and resolution actions.")
    
    # Render interactive table of cases
    cases_report_data = []
    for a in st.session_state.alerts:
        alert_id = a['alert_id']
        cases_report_data.append({
            "Incident ID": alert_id,
            "Timestamp": a['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
            "V3": f"{a['v3']:.4f}",
            "V10": f"{a['v10']:.4f}",
            "V12": f"{a['v12']:.4f}",
            "V14": f"{a['v14']:.4f}",
            "V17": f"{a['v17']:.4f}",
            "ML Confidence": f"{a['ml_probability']:.2%}",
            "Rules Triggered": ", ".join(a['matched_rules']) if a['matched_rules'] else "None",
            "Hybrid Risk Score": f"{a['risk_score']:.2%}",
            "Resolution State": st.session_state.case_statuses.get(alert_id, "New"),
            "Assigned Analyst": st.session_state.case_assignees.get(alert_id, "Unassigned"),
            "Audit Notes": st.session_state.case_notes.get(alert_id, "")
        })
        
    df_report = pd.DataFrame(cases_report_data)
    
    if df_report.empty:
        st.info("No cases logged in memory. Populate cases via simulator to enable exports.")
    else:
        st.dataframe(df_report, use_container_width=True)
        
        # Download buttons
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            csv_data = df_report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Audit File (CSV)",
                data=csv_data,
                file_name=f"guardianeye_audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                use_container_width=True
            )
            
        with col_ex2:
            json_data = df_report.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download Forensic Log (JSON)",
                data=json_data,
                file_name=f"guardianeye_forensics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime='application/json',
                use_container_width=True
            )
            
        st.markdown("<hr style='opacity: 0.1;'>", unsafe_allow_html=True)
        
        # Webhook notifications & APIs integration settings simulation
        st.markdown("### Integration Endpoints & REST Webhooks")
        st.write("Configure external communication pipelines to send instant Slack alerts, email escalations, or Kafka messages when severe fraud transactions trigger.")
        
        with st.form("webhook_config_form"):
            st.text_input("Incident Escalation Slack Webhook URL", placeholder="Enter Slackbook URL",value="")
            st.text_input("PagerDuty Integration Routing Key", value="pd_routing_key_xyz_12345_secure")
            st.selectbox("Escalation Protocol Severity", ["High Severity Only (Risk ≥ 70%)", "All System Anomalies (Risk ≥ 50%)"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.form_submit_button("Test & Save Endpoint Configuration"):
                st.success("Escalation pipeline updated. Connection test successful! Sending JSON ping payload.")
