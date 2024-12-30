import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the Streamlit app
st.title('House Price Prediction')

# Create input fields
squareMeters = st.number_input('Square Meters')
numberOfRooms = st.number_input('Number of Rooms')
hasYard = st.selectbox('Has Yard', [0, 1])
hasPool = st.selectbox('Has Pool', [0, 1])
floors = st.number_input('Floors')
cityCode = st.number_input('City Code')
cityPartRange = st.number_input('City Part Range')
numPrevOwners = st.number_input('Number of Previous Owners')
made = st.number_input('Year Made')
isNewBuilt = st.selectbox('Is New Built', [0, 1])
hasStormProtector = st.selectbox('Has Storm Protector', [0, 1])
basement = st.selectbox('Has Basement', [0, 1])
attic = st.selectbox('Has Attic', [0, 1])
garage = st.selectbox('Has Garage', [0, 1])
hasStorageRoom = st.selectbox('Has Storage Room', [0, 1])
hasGuestRoom = st.selectbox('Has Guest Room', [0, 1])

# Prediction
if st.button('Predict'):
    features = pd.DataFrame({
        'squareMeters': [squareMeters],
        'numberOfRooms': [numberOfRooms],
        'hasYard': [hasYard],
        'hasPool': [hasPool],
        'floors': [floors],
        'cityCode': [cityCode],
        'cityPartRange': [cityPartRange],
        'numPrevOwners': [numPrevOwners],
        'made': [made],
        'isNewBuilt': [isNewBuilt],
        'hasStormProtector': [hasStormProtector],
        'basement': [basement],
        'attic': [attic],
        'garage': [garage],
        'hasStorageRoom': [hasStorageRoom],
        'hasGuestRoom': [hasGuestRoom],
        'LivingAreaPerRoom': [squareMeters / numberOfRooms]
    })

    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features_scaled)
    st.write(f'Predicted Price: {prediction[0]}')

