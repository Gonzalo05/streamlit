import streamlit as st
from my_map_component import my_map
from geopy.geocoders import Nominatim
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pytz
import plotly.express as px
import plotly.graph_objects as go
import requests, pandas as pd, datetime as dt
from datetime import datetime, timedelta, time, date
from meteostat import Point, Hourly
from zoneinfo import ZoneInfo
import holidays
import joblib
import math
import os
import json
import urllib.parse


#_____Global Variables_______________________________________________________________________________________
API_KEY = "39dc9e88-98fb-449f-ae0f-f44d99a4fc5b"
LATITUDE = 35.7968864
LONGITUDE = -78.8080444
DISTANCE = 50
MAX_RESULTS = 300
EM_TOKEN = "WL8ZxbXPsYcbQsQAoFq9"          
EM_URL   = "https://api.electricitymap.org/v3/carbon-intensity/forecast"
EM_URL_HIST = "https://api.electricitymap.org/v3/carbon-intensity/history"
DATA_PATH = 'data/ev-data.json'
PERSIST_PATH = 'data/last_selection.json'
ORS_API_KEY = "5b3ce3597851110001cf62484dc6927bf4c24f84b8d06a72563555cc" 
GEOCODE_SEARCH_URL = "https://api.openrouteservice.org/geocode/search"
GEOCODE_REVERSE_URL = "https://api.openrouteservice.org/geocode/reverse"


#____Load EV Data_______________________________________________________________________________________
# Load EV data once
@st.cache_data
def load_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

# Load last selection (brand, model, specs) if exists
def load_selection():
    if os.path.exists(PERSIST_PATH):
        with open(PERSIST_PATH, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

last_sel = load_selection()


#____Load Models_________________________________________________________________________________________
@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model("predictions/ev_energy_prediction.joblib")
pipe = load_model("predictions/ev_duration_prediction.joblib")

@st.cache_resource
def inject_css():
    st.markdown("""
    <style>
    :root {
        --primary-bg: #FFFFFF!important;
        --secondary-bg: #F0F2F6!important;
        --primary-text: #262730!important;
        --secondary-text: #6E7191!important;
        --tertiary-bg: #DFE0EB!important;
        --accent-color: #blue!important;
    }
    
    .topCharger{
        display: grid;
        grid-template-columns: 55px 1fr;
        font-size: 22px;
        grid-gap: 15px;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .topMetric{
        width: 100%;
        padding-left: 85px;
        pading-right: 30px;
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr;
        grid-gap: 20px;
        text-align: center;
        font-size: 20px;
        margin-bottom: 10px;
        text-align: center;
        border-bottom: 1px gray solid;
    }
    
    .topTitle{
        font-weight: 800;
        font-size: 35px;
        text-align: center;
        margin-bottom: 10px;
        margin-top: 20px;
    }
    
    .topContent{
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr;
        grid-gap: 20px;
        background: rgb(240, 242, 246);
        border-radius: 15px;
        padding: 10px 30px;
    }
    
    .topRank, .TopTotal, .topLoc{
        font-weight: 800;
    }
    
    .topLoc > a{
        text-align: left;
        color: black;
        text-decoration: none;
    }
    
    .topContainer > .topCharger:nth-of-type(2) > .topContent{
        background: black;
        color: white;
    }
    
    .topContainer > .topCharger:nth-of-type(2) > .topContent > .topLoc > a{
        color: white;
    }
    
    .topRank{
        text-align: center;
        font-weight: 800;
        background: black;
        color: white;
        border-radius: 10px;
        font-size: 30px;
        line-height: 53px;
    }
    
    .stSidebar, div[data-testid="stSidebarCollapsedControl"]{
        display:none!important;
    }
    
    button[data-testid="stNumberInputStepUp"], button[data-testid="stNumberInputStepDown"]{
        display: none;
    }
    html, body, [data-testid="stAppViewContainer"], .block-container {
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        overflow: hidden; /* no scrollbars */
    }
    #MainMenu, header, footer {
        visibility: hidden;
    }
    .st-emotion-cache-b92z60{
        color: black!important;
    }
    iframe{
        position:fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
    }
    .info-box, .stVerticalBlock:has(#info){
        padding: 10px;
        position: fixed;
        bottom: 20px;
        border-radius: 25px;
        padding-left: 20px;
        padding-right: 20px;
        background: white;
        border: black 3px solid;
        box-shadow: black 5px 5px 0 0;
        width: 346px;
        right: 20px;
        z-index: 10;
        padding-top: 0;
    }

    #info{
        display: none;
    }

    @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
    }

    .st-emotion-cache-1104ytp b, .st-emotion-cache-1104ytp strong{
        font-weight: 800!important;
    }

    .stVerticalBlock:has(#search-now), .stVerticalBlock:has(#carbonIntensity), .stVerticalBlock:has(#vehicle){
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 10;
        width: 74vw;
        background-color: white;
        padding: 20px;
        padding-top: 0;
        border: black 3px solid;
        box-shadow: black 3px 4px 0 0;
        border-radius: 25px;
    }
    
    #search-now{
        display:none;
    }
    
    .stVerticalBlock:has(#vehicle){
        left: 20px;
        top: 20px;
        width: 22vw;
    }
    
    #vehicle{
        padding-top: 15px;
    }
    
    .titlecar{
        color: rgb(79 91 119);
        font-size: 20px;
        height: 30px;
    }
        
    .carmodel{
        font-size: 35px;
        font-weight: 800;
        line-height: 35px;
        margin-bottom: 30px;
    }
    
    button[kind="secondary"], button[kind="secondaryFormSubmit"]{
        background: black!important;
        color: white!important;
    }

    .modal-overlay { 
        position: fixed;
        width: 100vw;
        height: 100vh;
        background: #000000e8;
        top: 0;
        z-index: 15;
        left: 0;
    }

    .stVerticalBlock:has(#find-chargers), .stVerticalBlock:has(#prediction-calculator), .stVerticalBlock:has(#carbon-intensity-forecast), .stVerticalBlock:has(#choose-your-car){
        position: fixed;
        width: 90vw;
        height: 90vh;

        top: 50%;
        left: 50%;

        margin-left: -45vw!important;
        margin-top: -45vh!important;
        margin: auto;
        overflow: scroll;
        z-index: 20;
        padding: 70px;
        border-radius: 25px;
        background: white;
        border: black 3px solid;
        box-shadow: black 5px 5px 0 0;
    }

    .stVerticalBlock:has(#find-chargers) > div:nth-of-type(2) > div > button, .stVerticalBlock:has(#prediction-calculator) > div:nth-of-type(2) > div > button, .stVerticalBlock:has(#carbon-intensity-forecast) > div:nth-of-type(2) > div > button, .stVerticalBlock:has(#choose-your-car) > div:nth-of-type(2) > div > button{
        position: absolute;
        right: 0;
        top: -40px;
    }
    .stVerticalBlock:has(#search-now) > div:nth-of-type(3) > div > button{
        width: 100%;
    }

    .stVerticalBlock:has(#search-now) > div:nth-of-type(3) > div > button > div > p{
        font-size: 22px;
        font-weight: 800;
        text-transform: uppercase;
    }
    
    .stVerticalBlock:has(#prediction-calculator) > div:nth-of-type(9) > div {
        width: 100%;
        display: flex;
        justify-content: center;
        padding: 50px;
    }
    
    .stVerticalBlock:has(#prediction-calculator) > div:nth-of-type(9) > div > button > div > p {
        font-size: 36px;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 700;
    }

    #prediction-calculator, #carbon-intensity-forecast, #choose-your-car, #find-chargers{
        position: absolute;
        top: -70px;
    }
    
    label > div > p{
        font-size: 17px!important;
        font-weight: 700;
    }

    .stRadio > div{
        padding: 20px;
        background: rgb(240, 242, 246);
        border-radius: 15px;
    }

    .output{
        font-size: 30px;
        font-weight: 700;
        border: black 3px solid;
        box-shadow: black 5px 5px 0 0;
        margin-bottom: 250px;
        text-align: center;
        width: 300px;
        padding: 5px;
        border-radius: 20px;
        background-color:white;
        height: 160px;
    }
    .title{
        color: rgb(79 91 119)
    }
    .content{
        font-size: 80px;
        color: black;
        font-weight: 700;
        height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .contentinfo{
        color: black;
        font-size: 25px;
    }

    .outputcontainer{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 40px;
    }

    #time-cost-prediction, #budget-driven-prediction, #charge-cost-prediction{
        font-weight: 700;
        text-align: center;
        font-size: 50px;
        margin-bottom: 40px;
    }

    div[data-testid="stTextInput"] label, div[data-testid="stMultiSelect"] label {display: none;}    
    #carbonIntensity{
        margin: 0;
        border: none;
        box-shadow: none;
    }
    .stVerticalBlock:has(#carbonIntensity){
        position: fixed; 
        bottom: 20px;
        left: 20px;
        top: inherit;
        right: inherit;
        width: auto;
        z-index: 10;
        display: flex;
        align-items: center;
    }
    .timecointainer{
        border: 1px solid rgba(0, 0, 0, 0.2);
        border-radius: 0.5rem;    
        padding: 25px;
        text-align: center;
    }
    .timetitle{
        font-size: 50px!important;
        font-weight: 700;
        margin: 0!important;
    }
    .timecontent{
        font-weight: 800;
        font-size: 55px;
        color: white;
        background: black;
        border-radius: 10px;
        padding: 5px;
        text-transform:uppercase;
        line-height: 0;
        padding-left: 15px;
        padding-right: 15px;
    }
    .untilltiltle{
        font-size: 30px!important;
        color: gray;
        margin: 0;
        text-align: center;
    }
    .untilcontent{
        font-weight: 800;
        color: black;
        font-size: 35px;
    }
    .weather{
        border-width: 1px;
        border-color: black;
        text-align: center;
        border: black 3px solid;
        box-shadow: black 5px 5px 0 0;
        border-radius: 15px;
        padding: 25px;
        padding-top: 5px;
        padding-bottom: 5px;
        background: white;
    }
    .emoj{
        font-size: 35px;
        height: 50px;
    }
    .metric{
        height: 15px;
        color: gray;
        font-size: 18px;
    }
    .weathernum{
        font-size: 34px;
        font-weight: 800;
    }
    .weathercontainer{
        display: flex;
        position: fixed;
        column-gap: 10px;
        left: 0;
        bottom: 20px;
        width: 100vw;
        justify-content: center;
        z-index: 1;
    }
    .specscontainer {
        display: flex;
        flex-wrap: wrap;
    }
    .spec {
        display: inline-block;
        background: #f0f2f6;
        border-radius: 8px;
        padding: 12px;
        margin: 8px;
        text-align: center;
        width: 120px;
    }
    .emoji {
        font-size: 24px;
    }
    .specmetric {
        font-size: 14px;
        margin-top: 4px;
        color: #555;
    }
    .specnum {
        font-size: 18px;
        font-weight: bold;
        margin-top: 2px;
    }
    </style>
    """, unsafe_allow_html=True)

inject_css()

#_____Carbon Intensity Data__________________________________________________________________________________
def carbon_forecast(lat: float, lon: float) -> pd.DataFrame:
    """
    Fetch up-to-72 h forecast for the given lat/lon.
    Returns a DataFrame with UTC 'time' index and 'gco2_kwh' column.
    """
    hdrs = {"auth-token": EM_TOKEN}
    params = {"lat": lat, "lon": lon}
    r = requests.get(EM_URL, headers=hdrs, params=params, timeout=15)
    r.raise_for_status()
    
    raw = r.json().get("forecast", [])

    df = pd.DataFrame(raw)             # columns: carbonIntensity, datetime, zone
    df['timestamp'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return (
        df.rename(columns={"carbonIntensity": "gco2_kwh"})
        [["timestamp", "gco2_kwh"]]
    )

def carbon_hisotry(lat: float, lon: float):
    """
    Fetch up-to-72 h forecast for the given lat/lon.
    Returns a DataFrame with UTC 'time' index and 'gco2_kwh' column.
    """
    hdrs = {"auth-token": EM_TOKEN}
    params = {"lat": lat, "lon": lon}
    r = requests.get(EM_URL_HIST, headers=hdrs, params=params, timeout=15)
    r.raise_for_status()
    
    raw = r.json().get("history", [])

    df = pd.DataFrame(raw)             # columns: carbonIntensity, datetime, zone
    df['timestamp'] = pd.to_datetime(df['datetime'], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    return (
        df
        .rename(columns={"carbonIntensity": "gco2_kwh"})
        [["timestamp", "gco2_kwh"]]
    )

def next_lowest_emission(
    df: pd.DataFrame,
    now: pd.Timestamp,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp
):
    window_df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
    if window_df.empty:
        raise ValueError(f"No timestamps between {start_dt} and {end_dt}.")

    # find minimum gco2_kwh
    idx = window_df['gco2_kwh'].idxmin()
    best_ts = window_df.at[idx, 'timestamp']
    delta = best_ts - now
    return best_ts, delta


#____Carbon Intensity Dynamics_____________________________________________________________________________________
@st.cache_data
def make_figure(hist, forecast):
    # Split historical vs forecast
    hist = hist
    fcst = forecast.iloc[1:]

    fig = go.Figure()
    df_combined = pd.concat([hist, fcst], axis=0, ignore_index=True)
    # 1) Full continuous blue line for everything
    fig.add_trace(go.Scatter(
        x=df_combined["timestamp"], y=df_combined["gco2_kwh"],
        mode="lines",
        name="gco2_kwh",
        line=dict(color="blue"),
        showlegend=False
    ))

    # 2) Black line JUST over historical points
    fig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["gco2_kwh"],
        mode="lines",
        name="Historical Segment",
        line=dict(color="black", width=2),
        showlegend=False  # legend already shows gco2_kwh / Forecast markers
    ))

    # 3) Markers: black for historical, blue for forecast
    fig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["gco2_kwh"],
        mode="markers",
        name="Historical",
        marker=dict(color="black", size=8)
    ))
    fig.add_trace(go.Scatter(
        x=fcst["timestamp"], y=fcst["gco2_kwh"],
        mode="markers",
        name="Forecast",
        marker=dict(color="blue", size=8)
    ))

    fig.update_layout(
        title=f"Carbon Intensity for {df_combined['timestamp'].dt.date.iloc[0]}",
        xaxis_title="Time (America/New York)",
        yaxis_title="gco2_kwh (CO2/kWh)",
        hovermode="x unified",
        xaxis=dict(rangeslider_visible=True),
        margin=dict(t=50, b=40, l=40, r=10)
    )
    return fig

forecast_ci = carbon_forecast(LATITUDE, LONGITUDE)
hist_ci = carbon_hisotry(LATITUDE, LONGITUDE)
fig = make_figure(hist_ci, forecast_ci)

with st.container():
    st.markdown(f'''
    <div class="output" id="carbonIntensity">
        <div class="title">Carbon Intensity</div>
        <div class="content">{hist_ci.iloc[-1]["gco2_kwh"]}</div>
        <div class="contentinfo">g of CO₂/Kwh</div>
    </div>
    ''', unsafe_allow_html=True)
    if st.button("Lower my Emissions"):
        st.session_state["carboninstensiy"] = True
        
if st.session_state.get("carboninstensiy"):
    # overlay
    st.markdown('<div class="modal-overlay"></div>', unsafe_allow_html=True)
    # modal box container
    container = st.container()
    container.markdown('<div class="modal-box">', unsafe_allow_html=True)
    if container.button("✕", key="modal_close"):
        st.session_state["carboninstensiy"] = False
        st.rerun()
        
    container.markdown("# Carbon Intensity Forecast")
    container.plotly_chart(fig, use_container_width=True)

    # Read data
    df = forecast_ci
    tz = df['timestamp'].dt.tz
    now = pd.Timestamp.now(ZoneInfo("America/New_York")).replace(minute=0, second=0, microsecond=0, tzinfo=None)

    # Initialize session state for slider default
    if 'time_range' not in st.session_state:
        start_default = now.to_pydatetime()
        end_default = (now + pd.Timedelta(hours=24)).to_pydatetime()
        st.session_state.time_range = (start_default, end_default)
    # Form for slider to delay execution until submission
    fc = container.form(key='time_form')
    start_dt, end_dt = fc.slider(
        "Select time window:",
        min_value=now.to_pydatetime(),
        max_value=(now + pd.Timedelta(hours=24)).to_pydatetime(),
        value=st.session_state.time_range,
        format="HH:MM",
        step=dt.timedelta(minutes=1),
        key='time_range'
    )
    submitted = fc.form_submit_button("Update")

    # Run calculation only when form is submitted
    if submitted:
        start_ts = pd.Timestamp(start_dt)
        start_ts = start_ts.tz_localize(tz) if start_ts.tzinfo is None else start_ts.tz_convert(tz)
        end_ts = pd.Timestamp(end_dt)
        end_ts = end_ts.tz_localize(tz) if end_ts.tzinfo is None else end_ts.tz_convert(tz)

        try:
            minnow = pd.Timestamp.now(ZoneInfo("America/New_York")).replace(tzinfo=None)
            best_ts, delta = next_lowest_emission(df, minnow, start_ts, end_ts)
            # format time as e.g. '6am'
            time_str = best_ts.strftime('%I%p').lstrip('0').lower()
            # compute hours and minutes
            total_secs = int(delta.total_seconds())
            hrs = total_secs // 3600
            mins = (total_secs % 3600) // 60
            delta_str = f"{hrs} hours and {mins} minutes"

            st.session_state.best_time = time_str
            st.session_state.time_until = delta_str
            container.markdown(f'''
            <div class="timecointainer">
                <div class="bestTime">
                    <p class="timetitle">Lowest Emmsions at  <span class="timecontent">{st.session_state.best_time}</span></p>
                </div>
                <div class="timeUntill">
                    <p class="untilltiltle">Time untill then:  <span class="untilcontent">{st.session_state.time_until}</span></p>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        except ValueError as e:
            st.error(str(e))

    container.markdown('</div>', unsafe_allow_html=True)

#_____Weather Data___________________________________________________________________________________________
location = Point(LATITUDE, LONGITUDE)

# Time range: now → next 24 hours
start = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None) - timedelta(hours=1)
end = start + timedelta(hours=25)

# Get forecast
data = Hourly(location, start, end)
forecast = data.fetch()
first = forecast.iloc[0]
weather = {
    "temp": first['temp'],
    "prcp": first['prcp'],
    "wspd": first['wspd'],
    "rhum": first['rhum'],
}
st.markdown(f'''
    <div class="weathercontainer">
        <div class="weather">
            <div class="emoj">🌡️</div>
            <div class="metric">Temperature</div>
            <div class="weathernum">{first['temp']:.1f} °C</div>
        </div>
        <div class="weather">
            <div class="emoj">💧</div>
            <div class="metric">Humidity</div>
            <div class="weathernum">{first['rhum']}</div>
        </div>
        <div class="weather">
            <div class="emoj">☔</div>
            <div class="metric">Precipitation</div>
            <div class="weathernum">{first['prcp']} mm</div>
        </div>
        <div class="weather">
            <div class="emoj">💨</div>
            <div class="metric">WIND</div>
            <div class="weathernum">{first['wspd']} km/h</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


#_____Station Data___________________________________________________________________________________________
@st.cache_data(ttl=3600)
def get_charging_stations():
    url = (
        "https://api.openchargemap.io/v3/poi/?output=json"
        f"&latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&distance={DISTANCE}&maxresults={MAX_RESULTS}&key={API_KEY}"
    )
    response = requests.get(url)
    return response.json()

def extract_relevant_data(item):
    """
    Extract more technical and accessibility-related details.
    Feel free to expand or customize as needed.
    """

    # Basic info
    station_id = item.get("ID", "N/A")
    is_verified = item.get("IsRecentlyVerified", False)
    last_verified = item.get("DateLastVerified", "N/A")
    last_status_update = item.get("DateLastStatusUpdate", "N/A")
    date_created = item.get("DateCreated", "N/A")
    number_of_points = item.get("NumberOfPoints", 0)
    usage_cost = item.get("UsageCost", "N/A")
    general_comments = item.get("GeneralComments", "N/A")

    # Operator info
    operator_info = item.get("OperatorInfo") or {}
    operator_name = operator_info.get("Title", "Unknown")
    operator_website = operator_info.get("WebsiteURL", "N/A")
    operator_phone = operator_info.get("PhonePrimaryContact", "N/A")
    operator_transform = {
            "(Business Owner at Location)": "Business Owner at Location",
            "(Unknown Operator)": "Unknown",
            "EnBW (D)": "EnBW",
            "Tesla (Tesla-only charging)": "Tesla",
            "Innogy SE (RWE eMobility)": "Innogy SE",
            # Optionally, unify "Unknown" to "Unknown" explicitly:
            "Unknown": "Unknown",
            "Be Energised (has-to-be)": "Be Energised",
            "E.ON (DE)": "E.ON",
            "Elli (Volkswagen Group Charging GmbH)": "Elli Volkswagen",
            "Allego BV": "Allego"
        }
    operator_name = operator_transform.get(operator_name, operator_name)


    usage_type = item.get("UsageType") or {}
    usage_type_ID = item.get("UsageTypeID", None)
    usage_title = usage_type.get("Title", "Unknown")
    is_pay_at_location = usage_type.get("IsPayAtLocation", False)
    is_membership_required = usage_type.get("IsMembershipRequired", False)
    is_access_key_required = usage_type.get("IsAccessKeyRequired", False)
    
    # Status
    status_type = item.get("StatusType") or {}
    status_title = status_type.get("Title", "Unknown")

    # Address Info
    address_info = item.get("AddressInfo", {})
    address_line1 = address_info.get("AddressLine1", "N/A")
    town = address_info.get("Town", "N/A")
    postcode = address_info.get("Postcode", "N/A")
    country = address_info.get("Country", {}).get("Title", "N/A")
    latitude = address_info.get("Latitude", 0.0)
    longitude = address_info.get("Longitude", 0.0)
    access_comments = address_info.get("AccessComments", "N/A")

    # Connections (technical aspects)
    conns = item.get("Connections", [])
    if conns:
        conn = conns[0]
    else:
        conn = {}
    ctype = conn.get("ConnectionType", {}).get("Title", "Unknown")
    conn_type = conn.get("ConnectionTypeID", None)
    currentType = conn.get("CurrentTypeID", None)
    power_kw = conn.get("PowerKW", None)
    voltage = conn.get("Voltage", "Unknown")
    amps = conn.get("Amps", "Unknown")
    ctp = conn.get("CurrentType") or {}
    current_type = ctp.get("Title", "Unknown")
    lt = conn.get("Level") or {}
    level_title = lt.get("Title", "Unknown")
    qty = conn.get("Quantity", 1)

    detail = f"{ctype} | {power_kw} kW | {voltage}V | {amps}A | {current_type} | {level_title} | x{qty}"

    # Combine connection details into a single string (or keep separate if you prefer)
    connections_info = detail

    return {
        "cx": LATITUDE,
        "cy": LONGITUDE,
        "ID": station_id,
        "power_kw": power_kw,
        "voltage": voltage,
        "amps": amps,
        "CurrentTypeID": currentType,
        "ConnectionTypeID": conn_type,
        "UsageTypeID": usage_type_ID,
        "NumConnectors": qty,
        "brand": operator_name,
        "Operator Website": operator_website,
        "Operator Phone": operator_phone,
        "Usage Type": usage_title,
        "Is Pay At Location?": is_pay_at_location,
        "Is Membership Required?": is_membership_required,
        "Is Access Key Required?": is_access_key_required,
        "Status": status_title,
        "Verified?": is_verified,
        "Last Verified": last_verified,
        "Last Status Update": last_status_update,
        "Date Created": date_created,
        "Number of Points": number_of_points,
        "Usage Cost": usage_cost,
        "General Comments": general_comments,
        "name": address_line1,
        "Town": town,
        "Postcode": postcode,
        "Country": country,
        "lat": latitude,
        "lon": longitude,
        "Access Comments": access_comments,
        "Connections": conns,
        "Connections (Detailed)": connections_info,
    }


#____EV Vehicle Dynamics_________________________________________________________________________________________
    # Function to render specs cards
def render_specs(specs, container, graph=True, side=False):
    if side==False: container.subheader(f"Specs for {specs.get('brand')} {specs.get('model')}")
    # Prepare entries: emoji, label, value
    entries = [
        ('🚗', 'Variant', specs.get('variant') or 'Standard'),
        ('📅', 'Year', specs.get('release_year')),
        ('🔋', 'Battery', f"{specs.get('usable_battery_size', specs.get('battery_size'))} kWh"),
        ('⚡', 'Consumption', f"{specs.get('energy_consumption', {}).get('average_consumption')} kWh/100km"),
        ('📏', 'Range', f"{specs.get('range')} km"),
        ('🔌', 'Voltage', f"{specs.get('charging_voltage')} V"),
        ('⚙️', 'AC Max', f"{specs.get('ac_charger', {}).get('max_power')} kW"),
        ('🚀', 'DC Max', f"{specs.get('dc_charger', {}).get('max_power')} kW"),
    ]
    # Build HTML
    
    if side==False: html = '<div class="specscontainer">'
    else: html = '<div class="specscontainer" id="sidespecs">'
    for emoji, label, value in entries:
        html += f'''
            <div class="spec">
            <div class="emoji">{emoji}</div>
            <div class="specmetric">{label}</div>
            <div class="specnum">{value}</div>
            </div>'''
    html += '</div>'
    container.markdown(html, unsafe_allow_html=True)
    if specs.get('dc_charger') and specs['dc_charger'].get('charging_curve') and graph:
        container.write("**DC Charging Curve**")
        curve = specs['dc_charger']['charging_curve']
        df_curve = {p['percentage']: p['power'] for p in curve}
        container.line_chart(df_curve)

has_selection = 'specs' in last_sel and last_sel['specs']
battery = None
curve = None
voltage = None
if has_selection:
    battery = last_sel['specs'].get('usable_battery_size', last_sel['specs'].get('battery_size'))
    curve = last_sel['specs']['dc_charger']['charging_curve']
    voltage = last_sel['specs'].get('charging_voltage')
else: 
    st.session_state["vehicle"] = True 
    
with st.container():
    default_brand = last_sel.get('brand', "No Car Selected")
    default_model = last_sel.get('model', '')
    st.markdown(f'''
    <div id="vehicle">
        <div class="titlecar">Your Car Model</div>
        <div class="carmodel">{default_brand} {default_model}</div>
    </div>
    ''', unsafe_allow_html=True)
    if st.button("Find My Car"):
        st.session_state["vehicle"] = True 
    
if st.session_state.get("vehicle"):
    # overlay
    st.markdown('<div class="modal-overlay"></div>', unsafe_allow_html=True)
    # modal box container
    container = st.container()
    container.markdown('<div class="modal-box">', unsafe_allow_html=True)
    if container.button("✕", key="modal_close"):
        st.session_state["vehicle"] = False
        st.rerun()
    
    ev_data = load_data()
    
    container.markdown("# Choose Your Car")
    brands = sorted({item['brand'] for item in ev_data['data']})
    default_brand = last_sel.get('brand', '-- Select Brand --')
    brand_options = ['-- Select Brand --'] + brands
    brand_index = brand_options.index(default_brand) if default_brand in brand_options else 0
    selected_brand = container.selectbox('Choose a Brand', options=brand_options, index=brand_index)

    # Model selection, default to last
    selected_model = None
    if selected_brand != '-- Select Brand --':
        models = sorted({item['model'] for item in ev_data['data'] if item['brand'] == selected_brand})
        default_model = last_sel.get('model', '-- Select Model --')
        model_options = ['-- Select Model --'] + models
        model_index = model_options.index(default_model) if default_model in model_options else 0
        selected_model = container.selectbox('Choose a Model', options=model_options, index=model_index)
    # If user made a selection, update persistence and show specs
    if selected_brand != '-- Select Brand --' and selected_model and selected_model != '-- Select Model --':
        # Retrieve specs
        specs = next((item for item in ev_data['data']
                    if item['brand'] == selected_brand and item['model'] == selected_model), None)
        if specs:
            # Save selection and specs
            with open(PERSIST_PATH, 'w', encoding='utf-8') as f:
                json.dump({
                    'brand': selected_brand,
                    'model': selected_model,
                    'specs': specs
                }, f, indent=2)
            render_specs(specs, container)
        else:
            st.warning('No specs found for the selected model.')
    # If no fresh selection, but persisted specs exist, display them
    elif last_sel.get('specs'):
        render_specs(last_sel['specs'], container)
    if container.button("Go to Map"):
        st.session_state["vehicle"] = False
        st.rerun()

    container.markdown('</div>', unsafe_allow_html=True)


#_____Stations Display and Interactions_____________________________________________________________________

def estimate_charge_time(
    soc_start: float,
    soc_end:   float,
    capacity_kwh: float,
    station_power: float,
    curve: list[dict],          # [{"percentage": %, "power": kW}, …]
    temp: float = None,         # ambient °C
) -> float:
    # 1) Build interpolation arrays
    soc_pts   = np.array([pt["percentage"] for pt in curve])
    p_car_pts = np.array([pt["power"]      for pt in curve])
    # 2) Temperature derate 
    if temp is not None:
        fT = 1.5 if 15 <= temp <= 35 else 1.2
    else:
        fT = 1.4
    # 3) Discretize SOC
    steps = 100
    socs = np.linspace(soc_start, soc_end, steps+1)
    times = []
    energies = []
    cum_time  = 0.0
    cum_energy = 0.0

    # Since energy dE = C(u) × d(SOC) and power P = dE/dt,
    # then dt = dE / P = (C(u) / 100) * d(SOC) / P(SOC)
    #
    # So the total charging time T is:
    # T = ∫ from SOC1 to SOC2 of [ (C(u) / 100) * d(SOC) / P_delivered(SOC) ]
    # So we approximate it using the sums over small steps
    for i in range(steps):
        soc_mid = 0.5*(socs[i] + socs[i+1])
        # interpolate car’s power at soc_mid
        p_car = np.interp(soc_mid, soc_pts, p_car_pts)
        # delivered power
        p_del = fT * min(station_power, p_car)
        # energy in this slice (kWh)
        delta_e = capacity_kwh * (socs[i+1] - socs[i]) / 100.0
        # time = energy / power (hours)
        cum_time  += (delta_e / p_del) * 60
        cum_energy += delta_e
        times.append(cum_time)
        energies.append(cum_energy)

    return cum_time, cum_energy


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ, dλ = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def filter_within_radius(chargers, lat, lon, radius_km=10):
    nearby = []
    for c in chargers:
        d = haversine_distance(lat, lon, c['lat'], c['lon'])
        if d <= radius_km:
            c['distance_km'] = d
            nearby.append(c)
    return nearby

def get_driving_times_ors(origin, destinations, api_key):
    url = "https://api.openrouteservice.org/v2/matrix/driving-car"
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    coords = [[origin[1], origin[0]]] + [[c['lon'], c['lat']] for c in destinations]
    body = {
        "locations": coords,
        "sources": [0],
        "destinations": list(range(1, len(coords))),
        "metrics": ["duration"],
        "units": "km"
    }
    resp = requests.post(url, json=body, headers=headers)
    resp.raise_for_status()
    return resp.json()['durations'][0]

def get_top5_by_total_time(chargers, origin_lat, origin_lon, api_key, capacity_kwh, soc_start, soc_end, curve, temp, radius_km=10):
    nearby = filter_within_radius(chargers, origin_lat, origin_lon, radius_km)
    if not nearby:
        return []

    drive_secs = get_driving_times_ors((origin_lat, origin_lon), nearby, api_key)

    results = []
    for charger, dt in zip(nearby, drive_secs):
        if dt is None:
            continue
        c = charger.copy()
        c['drive'] = dt / 60

        # Use charger['power_kw'] as station_power
        station_power = c.get('power_kw') or 3.7

        c['charge'], _ = estimate_charge_time(
            soc_start, soc_end,
            capacity_kwh,
            station_power,
            curve,
            temp
        ) 
        c['total'] = c['drive'] + c['charge']
        results.append(c)

    results.sort(key=lambda x: x['total'])
    return results[:5]



if "stations" not in st.session_state:
    st.session_state.stations = get_charging_stations()
stations = st.session_state.stations
results_list = [extract_relevant_data(item) for item in stations]
unique_brands = sorted({marker["brand"] for marker in results_list})

with st.container():
    st.title("Search Now!")
    selected_brands = st.multiselect(
        "",
        options=unique_brands,
        default=unique_brands,
        placeholder="Choose brands"
    )
    if st.button("Find Chargers", disabled=not has_selection):
        st.session_state["find_chargers"] = True
    if not has_selection:
        st.info("ℹ️ Please first select your car in the Find Car section before you can make charge forecasts.")


# 4) Filter marker data based on selected brands
filtered_markers = [
    m for m in results_list
    if m["brand"] in selected_brands
]

# 5) Render the map with only the filtered markers
click_result = my_map(data=filtered_markers)

def hour_display(total_minutes):
    # total minutes, rounded to nearest int
    hours = total_minutes // 60
    minutes = total_minutes % 60
    text = (
        (f"{hours:.0f} hour{'s' if hours != 1 else ''} and " if hours > 0 else "")
        + f"{minutes:.0f} min{'s' if minutes != 1 else ''}"
    )    
    return text

def make_gmaps_directions_link(
    start_lat: float,
    start_lon: float,
    dest_lat: float,
    dest_lon: float,
    start_label: str = None,
    dest_label: str = None,
) -> str:
    """
    Build a Google Maps URL showing driving directions from the start point to the destination.
    
    :param start_lat: latitude of the origin
    :param start_lon: longitude of the origin
    :param dest_lat: latitude of the destination
    :param dest_lon: longitude of the destination
    :param start_label: optional human-readable label for the origin (e.g. "Home")
    :param dest_label: optional human-readable label for the destination (e.g. "Charger")
    :return: URL string
    """
    base_url = "https://www.google.com/maps/dir/?api=1"
    params = {
        "origin": f"{start_lat},{start_lon}",
        "destination": f"{dest_lat},{dest_lon}",
        "travelmode": "driving",
    }
    if start_label:
        params["origin_place_id"] = start_label
    if dest_label:
        params["destination_place_id"] = dest_label

    return base_url + "&" + urllib.parse.urlencode(params)



if st.session_state.get("find_chargers"):
    # overlay
    st.markdown('<div class="modal-overlay"></div>', unsafe_allow_html=True)
    
    # modal box container
    container = st.container()
    container.markdown('<div class="modal-box">', unsafe_allow_html=True)
    if container.button("✕", key="modal_close"):
        st.session_state["find_chargers"] = False
        st.rerun()
    # --- Form contents ---
    container.markdown("# Find Chargers")

    if "user_lat" not in st.session_state:
        st.session_state.user_lat = None
    if "user_lon" not in st.session_state:
        st.session_state.user_lon = None

    with container.container():
        tab1, tab2 = st.tabs(["📍 Address", "📌 Coordinates"])

        with tab1:
            q = st.text_input("Enter address", placeholder="e.g. 23 Robadors, Barcelona")
            selected = None
            if len(q) >= 3:
                params = {"api_key": ORS_API_KEY, "text": q, "size": 5}
                resp = requests.get(GEOCODE_SEARCH_URL, params=params)
                resp.raise_for_status()
                features = resp.json().get("features", [])
                if features:
                    st.write("**Select from suggestions:**")
                    for i, feat in enumerate(features):
                        label = feat["properties"]["label"]
                        if st.button(f"📍 {label}", key=f"fwd_{i}"):
                            selected = feat
                if selected:
                    lat, lon = selected["geometry"]["coordinates"]
                    st.session_state.user_lat = lon
                    st.session_state.user_lon = lat
                    st.success(f"**Selected address:** {selected['properties']['label']}")

        with tab2:
            lat = float(st.number_input("Latitude", format="%.6f"))
            lon = float(st.number_input("Longitude", format="%.6f"))
            if st.button("Use these coords"):
                st.session_state.user_lat = lat
                st.session_state.user_lon = lon
    
    user_lat = st.session_state.user_lat
    user_lon = st.session_state.user_lon
    
    # Collect inputs dynamically
    with container.form(key="inputs"):
        cur_level = st.number_input("Current battery level (%)", min_value=0, max_value=100, key="cur_level")
        
        target_level = st.number_input("Target battery level (%)", min_value=0, max_value=100, key="target_level")
        calculate = st.form_submit_button("Calculate Predictions",disabled=(user_lon is None))
     
    if calculate:   
        best5 = get_top5_by_total_time(filtered_markers, user_lat, user_lon, ORS_API_KEY, battery, cur_level, target_level, curve, weather['temp'])
        html = '''<div class="topTitle">Top Time-Saving Chargers</div><div class="topContainer"><div class="topMetric">
                        <div class="topLoc">Adress</div>
                        <div class="topDrive">Driving Time</div>
                        <div class="TopCharge">Charing Time</div>
                        <div class="TopTotal">Total Time</div>
                    </div>'''
        for i, ch in enumerate(best5, 1):
            link = make_gmaps_directions_link(
                user_lat, user_lon,
                ch['lat'], ch['lon'],
                start_label="My+Location",
                dest_label="Charger"
            )
            html = html + f'''<div class="topCharger">
                    <div class="topRank">{i}</div>
                    <div class="topContent">
                        <div class="topLoc"><a href="{link}">{ch['name']}</a></div>
                        <div class="topDrive">{hour_display(ch['drive'])}</div>
                        <div class="TopCharge">{hour_display(ch['charge'])}</div>
                        <div class="TopTotal">{hour_display(ch['total'])}</div>
                    </div>
                </div>'''
        html = html + "</div>"
        container.markdown(html, unsafe_allow_html=True)     
    elif user_lon is None:
        container.info("ℹ️  Please enter your address first; the button will unlock once we have your location.")      
    container.markdown('</div>', unsafe_allow_html=True)


chargerdata = {}
# 6) If a marker is clicked, show a stylized info box
if click_result:
    geolocator = Nominatim(user_agent="my_map_app")
    location = geolocator.reverse(f"{click_result['lat']}, {click_result['lon']}")
    #address = location.address if location else "Unknown address"

    # Build a Google Maps link for the lat/lon
    #google_maps_link = f"https://www.google.com/maps/search/?api=1&query={click_result['Latitude']},{click_result['Longitude']}"
    
    marker_data = next(
        (m for m in filtered_markers if m["name"] == click_result["name"]),
        None
    )
 
    chargerdata = {
        "Latitude": click_result['lat'],
        "Longitude": click_result['lon'],
        "dist_km": 0,
        "Power_kW": click_result['power_kw'],
        "Amps": click_result['amps'],
        "Volate": click_result['voltage'],
        "NumConnectors": click_result['NumConnectors'],
        "Fee": 0,
        "StationID": click_result['ID'],
        "Model Number": click_result['brand'],
        "Model Number_api": "",
        "CurrentTypeID": click_result['ConnectionTypeID'],
        "ConnectionTypeID": click_result['ConnectionTypeID'],
        "UsageTypeID": click_result['UsageTypeID'],
    }
    
    # Show some basic info right away
    has_selection = 'specs' in last_sel and last_sel['specs']
    with st.container():
        st.title("info")
        st.subheader(f"**{click_result['name']}**")
        st.markdown(
            f"""
            <p><strong>Operator: </strong><a href="{click_result['Operator Website']}" target="_blank">{click_result['brand']}</a></p>
            """,
            unsafe_allow_html=True
        )
        st.write(f"**Status:** {click_result['Status']}")
        st.write(f"**Usage Cost:** {click_result['Usage Cost']}")
        st.write(f"**Address:** {click_result['name']}, {click_result['Town']}, {click_result['Postcode']}, {click_result['Country']}")
        if not has_selection:
            st.info("ℹ️ Please first select your car in the Find Car section before you can make charge forecasts.")
        if st.button("Predict", disabled=not has_selection):
            st.session_state["show_predict"] = True

        with st.expander("More User Info"):
            st.write(f"**Is Pay At Location?:** {click_result['Is Pay At Location?']}")
            st.write(f"**Is Membership Required?r:** {click_result['Is Membership Required?']}")
            st.write(f"**Is Access Key Required?:** {click_result['Is Access Key Required?']}")
            st.write(f"**Connections (Detailed):** {click_result['Connections (Detailed)']}")
    
#____Predictions__________________________________________________________________________________________________
def extract_time_features(timestamp, charging_time):
    ts_utc = pd.to_datetime(timestamp, utc=True)
    # now ts_utc is always tz=UTC
    ts_local = ts_utc.tz_convert("America/New_York")
    
    # 3) Extract components
    hour      = ts_local.hour
    dow       = ts_local.dayofweek      # 0=Monday, …, 6=Sunday
    month     = ts_local.month
    is_weekend = int(dow >= 5)
    
    # 4) Cyclical encodings
    hour_sin  = np.sin(2 * np.pi * hour  / 24)
    hour_cos  = np.cos(2 * np.pi * hour  / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    
    # 5) Parse charging_time into minutes
    if isinstance(charging_time, str):
        td = pd.to_timedelta(charging_time)
    elif isinstance(charging_time, timedelta):
        td = pd.Timedelta(charging_time)
    else:
        raise TypeError("charging_time must be str or timedelta")
    duration_min = td.total_seconds() / 60
    
    today = date.today()
    nc_holidays = holidays.US(state='NC')
    is_holiday_bool = today in nc_holidays
    is_holiday = int(is_holiday_bool)
    holiday_name_today = nc_holidays.get(today, "None")
    
    return {
        "hour": hour,
        "dow": dow,
        "month": month,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "Duration_min": duration_min,
        "is_holiday": is_holiday,
        "holiday_name": holiday_name_today
    }

def predict_energy(feature_row):
    """feature_row must supply all numeric+categorical columns used above"""
    X_new = pd.DataFrame([feature_row])
    return model.predict(X_new)[0]*4

def predict_duration(feature_dict: dict, desired_kwh: float) -> float:
    """
    feature_dict: all numeric & categorical features except DesiredEnergy
    desired_kwh : user’s requested energy to add
    returns minutes required at this charger
    """
    row = {**feature_dict, "DesiredEnergy": desired_kwh}
    X_new = pd.DataFrame([row])
    return pipe.predict(X_new)[0]

if st.session_state.get("show_predict"):
    # overlay
    st.markdown('<div class="modal-overlay"></div>', unsafe_allow_html=True)
    
    # modal box container
    container = st.container()
    container.markdown('<div class="modal-box">', unsafe_allow_html=True)
    if container.button("✕", key="modal_close"):
        st.session_state["show_predict"] = False
        st.rerun()
    # --- Form contents ---
    container.markdown("# Prediction Calculator")
    mode = container.radio("Prediction Mode", ["Time", "Charge"])

    # Collect inputs dynamically
    with container.form(key="inputs"):
        required = []
        # shared inputs
        cur_level = st.number_input("Current battery level (%)", min_value=0, max_value=100, key="cur_level")
        required.append(cur_level is not None)

        now = pd.Timestamp.now(ZoneInfo("America/New_York")).floor("H")
        tomorrow  = now + timedelta(hours=24)
        today_date    = now.date()
        tomorrow_date = (now + timedelta(days=1)).date()
        
        hourly = [ (now + timedelta(hours=i)).to_pydatetime() for i in range(25) ]

        # Prepare display labels (just hour:minute, but you still get full datetime back)
        labels = [dt.strftime("%Y-%m-%d %H:%M") for dt in hourly]


        start_time = st.selectbox(
            "Start Charge",
            options=hourly,
            format_func=lambda dt: (
                ("Today at " if dt.date()==today_date else
                "Tomorrow at " if dt.date()==tomorrow_date else
                dt.strftime("%Y-%m-%d"))
                + " " + dt.strftime("%H:%M")
            ),
            index=0,
            key="selected_hour"
        )
        
        dur = 0
        
        if mode == "Time":
            target_level = st.number_input("Target battery level (%)", min_value=0, max_value=100, key="target_level")
            required.append(target_level is not None)
        elif mode == "Charge":
            dur = st.slider(
                "Select charging duration (minutes):",
                min_value=0,
                max_value=480,
                value=60,
                step=15,
                format="%d min"
            )
            required.append(dur > 0)
        calculate = st.form_submit_button("Calculate Predictions")
    
    # 4) Render chart INSIDE the modal
    #container.plotly_chart(fig, use_container_width=True)
    
    if calculate:     
        price_kwh = 0.30  # constant for simplicity
        charging_duration = timedelta(minutes=dur)
        start_time = start_time.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        times = extract_time_features(start_time, charging_duration)
        selected_row = forecast.loc[start_time]
        weatherForecast = {
            "temp": selected_row['temp'],
            "prcp": selected_row['prcp'],
            "wspd": selected_row['wspd'],
            "rhum": selected_row['rhum'],
        }
        combined = {**chargerdata, **weatherForecast, **times}
        
        if mode == "Time":
            #Energy_needed (kWh) = Usable_capacity (kWh)  × (Target_% – Current_%)  ÷ 100
            desired_kwh = battery * (target_level - cur_level) / 100 
            timeopt2, energy = estimate_charge_time(cur_level, target_level,battery,chargerdata["Power_kW"], curve, weatherForecast["temp"])
            combined.pop("Duration_min", None)  # won't raise error if not present
            
            ci =  forecast_ci.loc[forecast_ci["timestamp"] == start_time, "gco2_kwh"].squeeze()
            emissions = ci * energy / 1000

            container.markdown("### Time & Cost Prediction")
            container.markdown(f'''
            <div class="outputcontainer">
                <div class="output">
                    <div class="title">Chargin Time</div>
                    <div class="content">{timeopt2:.2f}</div>
                    <div class="contentinfo">Minutes</div>
                </div>
                <div class="output">
                    <div class="title">Energy Delivered</div>
                    <div class="content">{energy:.2f}</div>
                    <div class="contentinfo">Kwh</div>
                </div>
                <div class="output">
                    <div class="title">Emissions</div>
                    <div class="content">{emissions:.2f}</div>
                    <div class="contentinfo">Kg CO2</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        elif mode == "Charge":
            est_kwh = predict_energy(combined)
            cost = np.round(est_kwh * price_kwh, 2)
            delta_pct = (est_kwh / battery) * 100            
            final_soc = cur_level + delta_pct
            final_soc = max(0.0, min(100.0, final_soc))
            ci =  forecast_ci.loc[forecast_ci["timestamp"] == start_time, "gco2_kwh"].squeeze()
            emissions = ci * est_kwh / 1000
            
            container.markdown("### Charge & Cost Prediction")
            container.markdown(f'''
            <div class="outputcontainer">
                <div class="output">
                    <div class="title">Energy delivered</div>
                    <div class="content">{est_kwh:.2f}</div>
                    <div class="contentinfo">kWh</div>
                </div>
                <div class="output">
                    <div class="title">Final SoC</div>
                    <div class="content">{final_soc:.2f}</div>
                    <div class="contentinfo">%</div>
                </div>
                <div class="output">
                    <div class="title">Emissions</div>
                    <div class="content">{emissions:.2f}</div>
                    <div class="contentinfo">Kg CO2</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

    
            
    container.markdown('</div>', unsafe_allow_html=True)
