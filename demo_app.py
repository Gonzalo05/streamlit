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
from datetime import datetime, time

st.title("Leaflet Clickable Map Demo")

@st.cache_resource
def inject_css():
    st.markdown("""
    <style>
    :root {
        --primary-bg: #FFFFFF;
        --secondary-bg: #F0F2F6;
        --primary-text: #262730;
        --secondary-text: #6E7191;
        --tertiary-bg: #DFE0EB;
        --accent-color: #FF4B4B;
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
        bottom: 1rem;
        border-radius: 25px;
        padding-left: 20px;
        padding-right: 20px;
        background: white;
        border: black 3px solid;
        box-shadow: black 5px 5px 0 0;
        width: 25vw;
        right: 1rem;
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

    .stVerticalBlock:has(#search-now){
        position: fixed;
        top: 1rem;
        left: 25vw;
        z-index: 10;
        width: 50vw;
        background-color: white;
        padding: 20px;
        padding-top: 0;
        border: black 3px solid;
        box-shadow: black 3px 4px 0 0;
        border-radius: 25px;
    }
    button{
        background: black!important;
        color: white!important;
    }
    .st-bn{
        background-color: black!important;
    }

    .modal-overlay { 
        position: fixed;
        width: 100vw;
        height: 100vh;
        background: #000000c7;
        top: 0;
        z-index: 10;
        left: 0;
    }

    .stVerticalBlock:has(#prediction-calculator){
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

    .stVerticalBlock:has(#prediction-calculator) > div:nth-of-type(2) > div > button {
        position: absolute;
        right: 0;
        top: -40px;
    }

    .stVerticalBlock:has(#prediction-calculator) > div:nth-of-type(10) > div {
        width: 100%;
        display: flex;
        justify-content: center;
        padding: 50px;
    }

    .stVerticalBlock:has(#prediction-calculator) > div:nth-of-type(10) > div > button > div > p {
        font-size: 36px;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 700;
    }

    #prediction-calculator{
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
        padding: 20px;
        border-radius: 20px;
        background-color:white;
        height: 200px;
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
    }

    #outputcontainer{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 40px;
    }

    #time-cost-prediction{
        font-weight: 700;
        text-align: center;
        font-size: 50px;
        margin-bottom: 40px;
    }

    div[data-testid="stTextInput"] label, div[data-testid="stMultiSelect"] label {display: none;}
    </style>
    """, unsafe_allow_html=True)

inject_css()

API_KEY = "39dc9e88-98fb-449f-ae0f-f44d99a4fc5b"
LATITUDE = 48.1351
LONGITUDE = 11.5820
DISTANCE = 50
MAX_RESULTS = 300

@st.cache_data(ttl=3600)
def get_charging_stations():
    url = (
        "https://api.openchargemap.io/v3/poi/?output=json"
        f"&latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&distance={DISTANCE}&maxresults={MAX_RESULTS}&key={API_KEY}"
    )
    response = requests.get(url)
    return response.json()
# --------------------------------------------------------------------
# 2) EXTRACT RELEVANT FIELDS
# --------------------------------------------------------------------
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
    usage_title = usage_type.get("Title", "Unknown")
    is_pay_at_location = usage_type.get("IsPayAtLocation", False)
    is_membership_required = usage_type.get("IsMembershipRequired", False)
    is_access_key_required = usage_type.get("IsAccessKeyRequired", False)

    # Status
    status_type = item.get("StatusType", {})
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
    connections = item.get("Connections", [])
    connection_str_list = []
    for conn in connections:
        ctype = conn.get("ConnectionType", {}).get("Title", "Unknown")
        power_kw = conn.get("PowerKW", "Unknown")
        voltage = conn.get("Voltage", "Unknown")
        amps = conn.get("Amps", "Unknown")
        ctp = conn.get("CurrentType") or {}
        current_type = ctp.get("Title", "Unknown")
        lt = conn.get("Level") or {}
        level_title = lt.get("Title", "Unknown")
        qty = conn.get("Quantity", 1)

        detail = (
            f"{ctype} | {power_kw} kW | {voltage}V | {amps}A | {current_type} | {level_title} | x{qty}"
        )
        connection_str_list.append(detail)

    # Combine connection details into a single string (or keep separate if you prefer)
    connections_info = "\n".join(connection_str_list) if connection_str_list else "N/A"

    return {
        "ID": station_id,
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
        "Connections (Detailed)": connections_info,
    }


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

# 4) Filter marker data based on selected brands
filtered_markers = [
    m for m in results_list
    if m["brand"] in selected_brands
]

# 5) Render the map with only the filtered markers
click_result = my_map(data=filtered_markers)

info_placeholder = st.empty()

# 6) If a marker is clicked, show a stylized info box
if click_result:
    geolocator = Nominatim(user_agent="my_map_app")
    location = geolocator.reverse(f"{click_result['lat']}, {click_result['lon']}")
    #address = location.address if location else "Unknown address"

    # Build a Google Maps link for the lat/lon
    #google_maps_link = f"https://www.google.com/maps/search/?api=1&query={click_result['Latitude']},{click_result['Longitude']}"

    info_placeholder.write("**Station Data:**")
    
    marker_data = next(
        (m for m in filtered_markers if m["name"] == click_result["name"]),
        None
    )
    
    # Show some basic info right away
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
        if st.button("Predict"):
            st.session_state["show_predict"] = True

        with st.expander("More User Info"):
            st.write(f"**Is Pay At Location?:** {click_result['Is Pay At Location?']}")
            st.write(f"**Is Membership Required?r:** {click_result['Is Membership Required?']}")
            st.write(f"**Is Access Key Required?:** {click_result['Is Access Key Required?']}")
            st.write(f"**Connections (Detailed):** {click_result['Connections (Detailed)']}")
    

@st.cache_data
def make_dummy_df():
    tz = pytz.timezone("Europe/Berlin")
    today = datetime.now(tz).date()
    idx = pd.date_range(start=pd.Timestamp(today, tz=tz), periods=24, freq="H")
    np.random.seed(42)
    vals = 0.30 + np.random.normal(0, 0.02, size=len(idx))
    return pd.DataFrame({"timestamp": idx, "price": vals})

@st.cache_data
def make_figure(df):
    # Split historical vs forecast
    hist = df[df["timestamp"].dt.hour < 12]
    fcst = df[df["timestamp"].dt.hour >= 12]

    fig = go.Figure()

    # 1) Full continuous blue line for everything
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["price"],
        mode="lines",
        name="Price",
        line=dict(color="blue"),
        showlegend=False
    ))

    # 2) Black line JUST over historical points
    fig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["price"],
        mode="lines",
        name="Historical Segment",
        line=dict(color="black", width=2),
        showlegend=False  # legend already shows Price / Forecast markers
    ))

    # 3) Markers: black for historical, blue for forecast
    fig.add_trace(go.Scatter(
        x=hist["timestamp"], y=hist["price"],
        mode="markers",
        name="Historical",
        marker=dict(color="black", size=8)
    ))
    fig.add_trace(go.Scatter(
        x=fcst["timestamp"], y=fcst["price"],
        mode="markers",
        name="Forecast",
        marker=dict(color="blue", size=8)
    ))

    fig.update_layout(
        title=f"Dummy Electricity Prices for {df['timestamp'].dt.date.iloc[0]}",
        xaxis_title="Time (Europe/Berlin)",
        yaxis_title="Price (€/kWh)",
        hovermode="x unified",
        xaxis=dict(rangeslider_visible=True),
        margin=dict(t=50, b=40, l=40, r=10)
    )
    return fig

df = make_dummy_df()
fig = make_figure(df)

# (3) Render the “modal” if requested
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
    mode = container.radio("Prediction Mode", ["Time & Cost", "Charge & Cost", "Budget‑Driven"])

    # Collect inputs dynamically
    required = []
    # shared inputs
    cur_level = container.number_input("Current battery level (%)", min_value=0, max_value=100, key="cur_level")
    required.append(cur_level is not None)

    ev_type = container.selectbox("EV battery type", ["Li-ion", "Lead‑acid", "NMC"], key="ev_type")
    required.append(bool(ev_type))

    start_time = container.time_input(
        "Charging start time",
        value=time(12, 0),
        key="start_time"
    )    
    required.append(start_time is not None)

    if mode == "Time & Cost":
        target_level = container.number_input("Target battery level (%)", min_value=0, max_value=100, key="target_level")
        required.append(target_level is not None)
    elif mode == "Charge & Cost":
        dur = container.slider("Charge duration", 0.0, 10.0, 1.0, step=0.25, key="dur")
        required.append(dur > 0)
    else:  # Budget‑Driven
        budget = container.number_input("Amount to spend (€)", min_value=0.0, key="budget")
        required.append(budget > 0)

    # 4) Render chart INSIDE the modal
    container.plotly_chart(fig, use_container_width=True)
    
    if container.button("Calculate Predictions"):
        # Generate random “price per kWh” to base costs on
        # (or pull from your dummy df)
        price_kwh = 0.30  # constant for simplicity

        if mode == "Time & Cost":
            # Random duration between 0.5 and 4 hours
            hours = np.round(np.random.uniform(0.5, 4.0), 2)
            # Random energy drawn (kWh)
            energy = np.round(np.random.uniform(10, 50), 1)
            cost = np.round(energy * price_kwh, 2)

            container.markdown("### Time & Cost Prediction")
            container.markdown(f'''
            <div id="outputcontainer">
                <div class="output">
                    <div class="title">Chargin Time</div>
                    <div class="content">{hours}</div>
                    <div class="contentinfo">Hours</div>
                </div>
                <div class="output">
                    <div class="title">Estimated Cost</div>
                    <div class="content">€{cost}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

        elif mode == "Duration & Cost":
            # User gives duration; we compute energy & cost
            # Here we randomize for demo
            dur = np.round(np.random.uniform(1, 3), 2)          # hours
            energy = np.round(dur * np.random.uniform(7, 11), 1)  # kW charging rate
            cost = np.round(energy * price_kwh, 2)
            final_soc = np.round(np.random.uniform(50, 100), 1)   # %
            
            container.markdown("### Duration & Cost Prediction")
            container.write(f"⏱️ **Charge duration:** {dur} hours")
            container.write(f"🔋 **Energy delivered:** {energy} kWh")
            container.write(f"⚡ **Final SoC:** {final_soc}%")
            container.write(f"💶 **Estimated cost:** €{cost}")

        else:  # Budget-Driven
            # User gives budget; we compute energy & duration
            budget = np.round(np.random.uniform(5, 20), 2)       # €
            energy = np.round(budget / price_kwh, 1)             # kWh
            rate = np.random.uniform(7, 11)                      # kW
            hours = np.round(energy / rate, 2)
            final_soc = np.round(np.random.uniform(40, 90), 1)    # %
            
            container.markdown("### Budget-Driven Prediction")
            container.write(f"💶 **Budget:** €{budget}")
            container.write(f"🔋 **Energy purchased:** {energy} kWh")
            container.write(f"⏱️ **Charging time:** {hours} hours")
            container.write(f"⚡ **Final SoC:** {final_soc}%")
            
    container.markdown('</div>', unsafe_allow_html=True)
