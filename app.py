import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Config ---
st.set_page_config(page_title="Dry Bean Classifier", layout="wide")

# --- Load the Model ---
@st.cache_resource
def load_model():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Bean Classes Mapping
bean_types = ['BARBUNYA', 'BOMBAY', 'CALI', 'DERMASON', 'HOROZ', 'SEKER', 'SIRA']

# --- UI Header ---
st.title("🌱 Dry Bean Type Classification")
st.markdown("Devloped by PRINCE RAJPUT")

# --- Sidebar Input with Sliders ---
st.sidebar.header("Adjust Bean Measurements")

def user_input_features():
    # Dataset ki ranges ke hisaab se sliders set kiye gaye hain
    area = st.sidebar.slider("Area", 20000.0, 255000.0, 50000.0)
    perimeter = st.sidebar.slider("Perimeter", 500.0, 2000.0, 850.0)
    major_axis = st.sidebar.slider("Major Axis Length", 180.0, 750.0, 320.0)
    minor_axis = st.sidebar.slider("Minor Axis Length", 120.0, 450.0, 200.0)
    aspect_ratio = st.sidebar.slider("Aspect Ratio", 1.0, 2.5, 1.5)
    eccentricity = st.sidebar.slider("Eccentricity", 0.2, 1.0, 0.7)
    convex_area = st.sidebar.slider("Convex Area", 20000.0, 265000.0, 52000.0)
    equiv_diameter = st.sidebar.slider("Equiv Diameter", 160.0, 600.0, 250.0)
    extent = st.sidebar.slider("Extent", 0.4, 0.9, 0.7)
    solidity = st.sidebar.slider("Solidity", 0.9, 1.0, 0.99)
    roundness = st.sidebar.slider("Roundness", 0.4, 1.0, 0.8)
    compactness = st.sidebar.slider("Compactness", 0.6, 1.0, 0.8)
    sf1 = st.sidebar.slider("Shape Factor 1", 0.002, 0.01, 0.006, format="%.5f")
    sf2 = st.sidebar.slider("Shape Factor 2", 0.0005, 0.004, 0.001, format="%.5f")
    sf3 = st.sidebar.slider("Shape Factor 3", 0.4, 1.0, 0.6)
    sf4 = st.sidebar.slider("Shape Factor 4", 0.9, 1.0, 0.99, format="%.4f")
    
    data = {
        'Area': area, 'Perimeter': perimeter, 'MajorAxisLength': major_axis,
        'MinorAxisLength': minor_axis, 'AspectRation': aspect_ratio,
        'Eccentricity': eccentricity, 'ConvexArea': convex_area,
        'EquivDiameter': equiv_diameter, 'Extent': extent, 'Solidity': solidity,
        'roundness': roundness, 'Compactness': compactness, 'ShapeFactor1': sf1,
        'ShapeFactor2': sf2, 'ShapeFactor3': sf3, 'ShapeFactor4': sf4
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Main Display ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Selected Parameters")
    st.write(input_df.T) # Transpose karke dikhane mein zyada saaf lagta hai

with col2:
    st.subheader("Prediction Result")
    if st.button("Predict Bean Type", use_container_width=True):
        prediction = model.predict(input_df)
        
        # Checking if prediction is numeric index or string
        if isinstance(prediction[0], (int, np.integer)):
            result = bean_types[prediction[0]]
        else:
            result = prediction[0]
            
        st.success(f"Detechted Bean Type: ### **{result}**")
        
        # Visual indicator (optional)
        st.info(f"Thank u for visiting here :- this is a {result} bean.")

# --- Deployment Note ---
st.divider()
st.caption("Developed for Dry Bean Classification Project | Model: SVC / Random Forest")
