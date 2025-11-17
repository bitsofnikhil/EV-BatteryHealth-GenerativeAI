import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EV Battery AI",
    page_icon="🔋",
    layout="wide"
)

# --- CUSTOM CSS FOR MODERN UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("EV Battery Health AI Assistant")
st.markdown("**Powered by Generative AI & Random Forest Regression**")
st.markdown("---")

# --- MODEL TRAINING (Cached so it only runs once) ---
@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv("ev_battery_charging_data.csv")
        
        # Encoding
        encoders = {}
        categorical_cols = ['Charging Mode', 'Battery Type', 'EV Model']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
            
        X = df.drop(columns=['Degradation Rate (%)', 'Optimal Charging Duration Class'], errors='ignore')
        y = df['Degradation Rate (%)']
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        return model, encoders, X.columns
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

model, encoders, feature_names = load_and_train_model()

if model:
    # --- SIDEBAR (USER INPUTS) ---
    st.sidebar.header("⚙️ Vehicle Parameters")
    
    def user_input_features():
        soc = st.sidebar.slider("State of Charge (SOC %)", 0, 100, 50)
        voltage = st.sidebar.number_input("Voltage (V)", 3.0, 4.5, 3.7)
        current = st.sidebar.number_input("Current (A)", 0.0, 100.0, 25.0)
        temp = st.sidebar.slider("Battery Temp (°C)", -10, 60, 30)
        amb_temp = st.sidebar.slider("Ambient Temp (°C)", -10, 50, 25)
        duration = st.sidebar.number_input("Charging Duration (min)", 0, 300, 60)
        efficiency = st.sidebar.slider("Efficiency (%)", 80, 100, 95)
        cycles = st.sidebar.number_input("Charging Cycles", 0, 5000, 200)
        
        c_mode = st.sidebar.selectbox("Charging Mode", encoders['Charging Mode'].classes_)
        b_type = st.sidebar.selectbox("Battery Type", encoders['Battery Type'].classes_)
        ev_model = st.sidebar.selectbox("EV Model", encoders['EV Model'].classes_)
        
        data = {
            'SOC (%)': soc,
            'Voltage (V)': voltage,
            'Current (A)': current,
            'Battery Temp (°C)': temp,
            'Ambient Temp (°C)': amb_temp,
            'Charging Duration (min)': duration,
            'Efficiency (%)': efficiency,
            'Charging Cycles': cycles,
            'Charging Mode': encoders['Charging Mode'].transform([c_mode])[0],
            'Battery Type': encoders['Battery Type'].transform([b_type])[0],
            'EV Model': encoders['EV Model'].transform([ev_model])[0]
        }
        
        # Ensure correct column order
        features = pd.DataFrame([data])
        features = features[feature_names]
        return features

    input_df = user_input_features()

    # --- MAIN PANEL ---
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("AI Analysis")
        if st.button("Predict Battery Health"):
            prediction = model.predict(input_df)[0]
            
            # Chatbot-style interaction
            st.chat_message("assistant").write("Processing vehicle telemetry data...")
            
            st.markdown(f"""
            <div class="prediction-box">
                <h3>Predicted Degradation Rate</h3>
                <h1 style="color: {'#ff4b4b' if prediction > 15 else '#4CAF50'};">{prediction:.2f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            if prediction < 5:
                st.success("**Status: Excellent.** Your battery is in great condition. The Generative AI model detects no anomalies.")
            elif prediction < 15:
                st.warning("**Status: Moderate.** Standard degradation detected. Consider optimizing charging habits.")
            else:
                st.error("**Status: Critical.** High degradation detected. Maintenance is recommended immediately.")

    with col2:
        st.subheader("📊 Visuals")
        st.info("Adjust the sliders in the sidebar to see how different factors affect the battery health in real-time.")
        st.image("Feature Importance.png", caption="AI Feature Importance Analysis", use_column_width=True)

else:
    st.warning("Please ensure 'ev_battery_charging_data.csv' is in the same folder.")