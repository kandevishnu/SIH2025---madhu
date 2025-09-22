"""
Full Student-friendly Hackathon Prototype
AI-Driven Community Microgrid
Dynamic, interactive, latest Streamlit compatible
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Community Microgrid — Prototype", layout="wide")

# ---------------------------
# Utility functions
# ---------------------------
def simulate_household_history(villages, households_per_village=4, hours=48):
    rows = []
    start = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(hours=hours)
    for v in villages:
        for h in range(1, households_per_village + 1):
            hh = f"{v}_HH{h:02d}"
            has_rooftop = np.random.choice([0,1], p=[0.4,0.6])
            base = np.random.uniform(0.3,1.2)
            for i in range(hours):
                ts = start + timedelta(hours=i)
                hour = ts.hour
                cons = base*(0.6 + 0.6*np.sin((hour-7)/24*2*np.pi))*np.random.uniform(0.9,1.2)
                if 18 <= hour <= 22:
                    cons *= 1.5
                solar = 0.0
                if 6 <= hour <= 17 and has_rooftop:
                    solar = base*np.random.uniform(0.5,1.5)
                rows.append({
                    'village': v, 'household': hh, 'datetime': ts, 'hour': hour,
                    'consumption_kwh': round(cons, 3), 'solar_kwh': round(solar, 3),
                    'has_rooftop': has_rooftop
                })
    return pd.DataFrame(rows)

def aggregate_village_hourly(df, battery_capacity=100.0):
    df_v = df.groupby(['village','datetime']).agg(
        consumption_kwh=('consumption_kwh','sum'),
        solar_kwh=('solar_kwh','sum')
    ).reset_index().sort_values(['village','datetime'])
    results = []
    for v in df_v['village'].unique():
        vdf = df_v[df_v['village']==v].copy().reset_index(drop=True)
        battery = battery_capacity*0.5
        for _, r in vdf.iterrows():
            needed = r['consumption_kwh']
            solar_used = min(needed, r['solar_kwh'])
            needed -= solar_used
            discharge = min(needed, battery)
            battery -= discharge
            needed -= discharge
            grid_import = needed
            battery_used_this_hour = discharge
            results.append({
                'village':v,'datetime':r['datetime'],'consumption_kwh':r['consumption_kwh'],
                'solar_kwh':r['solar_kwh'],'battery_level_kwh':round(battery,3),
                'battery_used_kwh':round(battery_used_this_hour,3),
                'grid_import_kwh':round(grid_import,3)
            })
    return pd.DataFrame(results)

def compute_credits(df_households):
    agg = df_households.groupby(['household','village']).agg(
        total_solar=('solar_kwh','sum'),
        avg_consumption=('consumption_kwh','mean'),
        has_rooftop=('has_rooftop','max')
    ).reset_index()
    agg['score'] = agg['total_solar']*2 + agg['has_rooftop']*3 - agg['avg_consumption']*0.5
    mn, mx = agg['score'].min(), agg['score'].max()
    agg['credits'] = ((agg['score']-mn)/(mx-mn)*100).round(0).astype(int) if mx-mn>0 else 10
    return agg.sort_values('credits',ascending=False)

def train_models(village_hourly):
    models = {}
    for v in village_hourly['village'].unique():
        vdf = village_hourly[village_hourly['village']==v].copy()
        vdf['hour'] = vdf['datetime'].dt.hour
        if len(vdf)<24:
            models[v] = None
            continue
        X = vdf[['hour','solar_kwh']]
        y = vdf['consumption_kwh']
        rf = RandomForestRegressor(n_estimators=40, random_state=42)
        rf.fit(X,y)
        models[v] = rf
    return models

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.title("Controls")
num_villages = st.sidebar.slider("Villages",1,4,2)
households = st.sidebar.slider("Households per village",2,8,4)
hours_history = st.sidebar.slider("History hours",24,72,48)
battery_capacity = st.sidebar.slider("Battery capacity per village (kWh)",20,500,100)

# --- Dynamic village names ---
st.sidebar.subheader("Village Names")
village_names = []
for i in range(num_villages):
    name = st.sidebar.text_input(f"Village {i+1} Name", value=f"Village_{i+1}", key=f"village_name_{i}")
    village_names.append(name)
villages = village_names

# ---------------------------
# Session state
# ---------------------------
if 'refresh_trigger' not in st.session_state: st.session_state.refresh_trigger = 0
if 'transactions' not in st.session_state: st.session_state.transactions = []

# Rebuild appliance state
old_appliance_state = st.session_state.get('appliance_state', {})
st.session_state.appliance_state = {
    f"{v}_HH{h:02d}": old_appliance_state.get(f"{v}_HH{h:02d}", {'fan': 0, 'light': 0, 'appl': 0})
    for v in villages for h in range(1, households + 1)
}

if st.sidebar.button("🔄 Manual refresh data"):
    st.session_state.refresh_trigger +=1
refresh_counter = st.session_state.refresh_trigger

# ---------------------------
# Data generation
# ---------------------------
df_house = simulate_household_history(villages, households_per_village=households, hours=hours_history)
for hh, app in st.session_state.appliance_state.items():
    if hh in df_house['household'].values:
        df_house.loc[df_house['household']==hh,'consumption_kwh'] += app['fan']*0.1 + app['light']*0.05 + app['appl']*0.2
village_hourly = aggregate_village_hourly(df_house, battery_capacity=battery_capacity)
credits_table = compute_credits(df_house)
models = train_models(village_hourly)

# ---------------------------
# Layout and Tabs
# ---------------------------
st.title("⚡ AI-Driven Community Microgrid — Hackathon Prototype")
tabs = st.tabs(["Dashboard","Households","Predictions","UPI Trading","Green Credits & Leaderboard"])

# ---- Dashboard Tab ----
with tabs[0]:
    st.header("Community Dashboard")
    col1,col2,col3,col4 = st.columns(4)
    vsel = st.selectbox("Select Village", villages, key='vsel')
    latest = village_hourly[village_hourly['village']==vsel].sort_values('datetime').tail(24)
    total_solar = latest['solar_kwh'].sum()
    total_consumption = latest['consumption_kwh'].sum()
    total_grid = latest['grid_import_kwh'].sum()
    battery_now = latest['battery_level_kwh'].iloc[-1] if not latest.empty else 0.0
    col1.metric("24h Solar (kWh)",f"{total_solar:.2f}")
    col2.metric("24h Consumption (kWh)",f"{total_consumption:.2f}")
    col3.metric("Grid Import (kWh)",f"{total_grid:.2f}")
    col4.metric("Battery (kWh)",f"{battery_now:.1f} / {battery_capacity}")

    st.subheader("Last 48 hours: Generation vs Demand")
    fig = go.Figure()
    vh = village_hourly[village_hourly['village']==vsel].sort_values('datetime')
    fig.add_trace(go.Scatter(x=vh['datetime'],y=vh['solar_kwh'],name='Solar'))
    fig.add_trace(go.Scatter(x=vh['datetime'],y=vh['consumption_kwh'],name='Consumption'))
    fig.add_trace(go.Bar(x=vh['datetime'],y=vh['grid_import_kwh'],name='Grid Import',opacity=0.5))
    fig.update_layout(height=420,xaxis_title='Datetime',yaxis_title='kWh')
    st.plotly_chart(fig,use_container_width=True)

# ---- Households Tab ----
with tabs[1]:
    st.header("Household Appliance Control")
    for v in villages:
        st.subheader(v)
        for h in range(1,households+1):
            hh = f"{v}_HH{h:02d}"
            state = st.session_state.appliance_state[hh]
            col1,col2,col3 = st.columns(3)
            state['fan'] = col1.slider(f"{hh} Fan (kW)",0,5,state['fan'])
            state['light'] = col2.slider(f"{hh} Light (kW)",0,5,state['light'])
            state['appl'] = col3.slider(f"{hh} Other Appliances (kW)",0,5,state['appl'])

# ---- Predictions Tab ----
with tabs[2]:
    st.header("Next 24h Consumption Forecast")
    if models[vsel]:
        hours_next = np.arange(24)
        solar_pred = np.random.uniform(0.3,2.5,24)
        X_next = pd.DataFrame({'hour':hours_next,'solar_kwh':solar_pred})
        y_pred = models[vsel].predict(X_next)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=hours_next,y=y_pred,name='Predicted Consumption'))
        fig3.update_layout(height=400,xaxis_title='Hour',yaxis_title='kWh')
        st.plotly_chart(fig3,use_container_width=True)
    else:
        st.info("Not enough data to train prediction model for this village.")

# ---- UPI Trading Tab ----
with tabs[3]:
    st.header("Peer-to-Peer Energy Trading (Mock)")

    hh_list = df_house['household'].unique().tolist()
    sender = st.selectbox("Sender", hh_list, key="sender")
    receiver = st.selectbox("Receiver", hh_list, key="receiver")
    amount = st.number_input("Amount of kWh to send", 0.0, 100.0, 1.0, key="amount")

    if st.button("Send Energy Credits"):
        if sender != receiver:
            df_house.loc[df_house['household'] == sender, 'consumption_kwh'].iloc[-1] -= amount
            df_house.loc[df_house['household'] == receiver, 'consumption_kwh'].iloc[-1] += amount
            st.session_state.transactions.append({
                'sender': sender,
                'receiver': receiver,
                'amount_kwh': amount
            })
            st.success(f"{amount} kWh sent from {sender} to {receiver}!")

            # Show table *only after sending*
            st.subheader("Transactions History")
            st.table(pd.DataFrame(st.session_state.transactions))

# ---- Green Credits Tab ----
with tabs[4]:
    st.header("Green Credits Leaderboard")
    credits_table = compute_credits(df_house)
    st.table(credits_table[['household','village','credits','total_solar']].reset_index(drop=True))