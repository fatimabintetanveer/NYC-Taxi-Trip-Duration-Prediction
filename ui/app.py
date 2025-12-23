import streamlit as st
import requests
import numpy as np
from math import radians, cos, sin, sqrt, atan2
import folium
from streamlit_folium import st_folium

# ─── Page Config & Sidebar ─────────────────────────────────────────────────
st.set_page_config(page_title="NYC Taxi Duration Predictor", page_icon="🚕", layout="wide")
st.sidebar.header("⚙️ Configuration")
API_URL = st.sidebar.text_input("FastAPI Endpoint", "http://localhost:8000/predict")

st.sidebar.markdown("---")
st.sidebar.header("📋 Trip Details")

vendor_id = st.sidebar.selectbox(
    "Vendor ID", 
    [1, 2], 
    help="Taxi company identifier: 1 or 2 (represents the two companies providing the data)"
)

passenger_count = st.sidebar.slider(
    "Passengers", 
    1, 6, 1, 
    help="Number of passengers in the trip (1 to 6)"
)

pickup_hour = st.sidebar.slider(
    "Hour of Day", 
    0, 23, 12, 
    help="Hour when the trip started (0 = midnight, 23 = 11 PM)"
)

pickup_weekday = st.sidebar.slider(
    "Weekday (0=Mon)", 
    0, 6, 2, 
    help="Day of the week (0 = Monday, 6 = Sunday)"
)

pickup_month = st.sidebar.slider(
    "Month", 
    1, 12, 6, 
    help="Month of the trip (1 = January, 12 = December)"
)

is_holiday = st.sidebar.selectbox(
    "Holiday?", 
    [0, 1], 
    help="1 = The trip occurred on a US public holiday, 0 = Not a holiday"
)

store_and_fwd = st.sidebar.selectbox(
    "Store & Fwd?", 
    [0, 1], 
    help=(
        "Whether the trip record was stored and forwarded due to connectivity issues:\n"
        "0 = No, sent immediately; 1 = Yes, stored then sent later"
    )
)

# ─── Helpers ────────────────────────────────────────────────────────────────
def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ─── Main ──────────────────────────────────────────────────────────────────
st.title("🚖 NYC Taxi Trip Duration Predictor")
st.markdown("Select pickup/dropoff on the maps below, then click **Predict**.")

# ── Two side-by-side maps ───────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Pickup Location")
    pickup_map = folium.Map(location=[40.7580, -73.9855], zoom_start=12)
    pickup_data = st_folium(pickup_map, height=400, width=400, key="pickup")
    if pickup_data and pickup_data.get("last_clicked"):
        pickup_lat = pickup_data["last_clicked"]["lat"]
        pickup_lon = pickup_data["last_clicked"]["lng"]
    else:
        pickup_lat = pickup_lon = None

with col2:
    st.subheader("📍 Dropoff Location")
    dropoff_map = folium.Map(location=[40.7580, -73.9855], zoom_start=12)
    dropoff_data = st_folium(dropoff_map, height=400, width=400, key="dropoff")
    if dropoff_data and dropoff_data.get("last_clicked"):
        dropoff_lat = dropoff_data["last_clicked"]["lat"]
        dropoff_lon = dropoff_data["last_clicked"]["lng"]
    else:
        dropoff_lat = dropoff_lon = None

# ── Coordinate display row ───────────────────────
if pickup_lat and dropoff_lat:
    coord_col1, coord_col2 = st.columns(2)
    with coord_col1:
        st.markdown(f"**Pickup Coordinates:** {pickup_lat:.5f}, {pickup_lon:.5f}")
    with coord_col2:
        st.markdown(f"**Dropoff Coordinates:** {dropoff_lat:.5f}, {dropoff_lon:.5f}")


# compute distance and show predict button
if pickup_lat and dropoff_lat:
    dist_km  = haversine_km(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    log_dist = np.log1p(dist_km)
    st.markdown(f"**Distance:** {dist_km:.2f} km  |  **Log Distance:** {log_dist:.3f}")

    # center the button using a 3-column trick
    b1, b2, b3 = st.columns([1,2,1])
    with b2:
        if st.button("🧠 Predict Duration", use_container_width=True):
            payload = {
                "vendor_id": vendor_id,
                "passenger_count": passenger_count,
                "pickup_hour": pickup_hour,
                "pickup_weekday": pickup_weekday,
                "pickup_month": pickup_month,
                "pickup_longitude": pickup_lon,
                "pickup_latitude": pickup_lat,
                "dropoff_longitude": dropoff_lon,
                "dropoff_latitude": dropoff_lat,
                "store_and_fwd_flag": store_and_fwd,
                "log_haversine_distance": log_dist,
                "is_holiday": is_holiday
            }
            try:
                res = requests.post(API_URL, json=payload, timeout=5)
                res.raise_for_status()
                mins = res.json().get("predicted_duration_min")
                st.success(f"⏱ **Estimated Duration:** {mins:.2f} min")
            except Exception as e:
                st.error(f"❌ API Error: {e}")
else:
    st.info("⏳ Please click on both maps to choose pickup & dropoff locations.")
