import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from streamlit_extras.stylable_container import stylable_container
import plotly.express as px

# Load car dataset
car_df = pd.read_csv("final_df.csv")

# # Load the model and scaler
# model = joblib.load("best_xgb_model.pkl")
# scaler = joblib.load('scaler.pkl')

# Initialize LabelEncoders for categorical features
categorical_columns = [
    'bodytype', 'fueltype', 'transmission', 'DriveType', 'Insurance', 'oem', 'city'
]
# #label_encoders = {col: LabelEncoder().fit(car_df[col]) for col in categorical_columns}
# label_encoder_bodytype = joblib.load("labelencoded.pkl")
# onehotencoder = joblib.load("onehotencoded.pkl")

# Function to predict the resale price
def predict_resale_price(m_bodytype, m_seats, m_km, m_modelYear, m_ownerNo, 
                        m_Engine, m_gear, m_mileage, m_fuel_type,
                        m_transmission, m_Insurance, m_oem, m_drivetype, m_city):
    # Prepare numerical features
    num_features = np.array([
        int(m_seats),
        int(m_km),
        int(m_modelYear),
        int(m_ownerNo),
        int(m_Engine),
        int(m_gear),
        float(m_mileage)
    ]).reshape(1, -1)
    
    # Scale numerical features
    scaled_num_features = scaler.transform(num_features)
    
    # Prepare categorical features for one-hot encoding
    categorical_data = pd.DataFrame({
        'bodytype': [m_bodytype],
        'fueltype': [m_fuel_type],
        'transmission': [m_transmission],
        'Drive_Type': [m_drivetype],
        'Insurance_Validity': [m_Insurance],
        'oem': [m_oem],
        'City': [m_city]
    })
    
    # One-hot encode categorical features
    encoded_cats = pd.get_dummies(categorical_data, columns=categorical_data.columns)
    
    # Ensure all columns from training are present
    for col in model.feature_names_in_:
        if col not in encoded_cats.columns:
            encoded_cats[col] = 0
            
    # Reorder columns to match training data
    encoded_cats = encoded_cats[model.feature_names_in_[7:]]  # Skip numerical features
    
    # Combine numerical and categorical features
    final_features = np.hstack((scaled_num_features, encoded_cats))
    
    # Make prediction
    prediction = model.predict(final_features)
    return prediction[0]


# Streamlit Page Configuration

# Title

import streamlit as st
# Move all other imports here at the top

# Place set_page_config as the very first Streamlit command
st.set_page_config(
    layout="wide",
    page_icon=":material/directions_bus:",
    page_title="CarPrediction Project",
    initial_sidebar_state="expanded"
)

# Rest of your Streamlit app code goes here

st.markdown(
    """
    <style>
    .logo-container {
        position: relative;
        float: right;
        margin-top: 80px;
        margin-right: 20px;
        z-index: 999;
    }
    .logo-image {
        width: 150px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    <div class="logo-container">
        <img class="logo-image" src="https://stimg2.cardekho.com/images/carNewsimages/userimages/650X420/30183/1672738680556/GeneralNew.jpg">
    </div>
    """,
    unsafe_allow_html=True
)



st.markdown(
    f"""
    <style>
    .stApp {{
        background-size: cover; /* Ensures the image covers the entire container */
        background-position: center; /* Centers the image */
        background-repeat: no-repeat; /* Prevents the image from repeating */
        background-attachment: fixed; /* Fixes the image in place when scrolling */
        height: 100vh; /* Sets the height to 100% of the viewport height */
        width: 100vw; /* Sets the width to 100% of the viewport width */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] {{
        background-color: #60191900; /* Replace with your desired color */
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    
    """,
    unsafe_allow_html=True
)






# Sidebar for user inputs
with st.sidebar:
    st.image(
        "https://stimg2.cardekho.com/images/carNewsimages/userimages/650X420/30183/1672738680556/GeneralNew.jpg",
        width=400
    )
    st.title(":red[Used Car Price Prediction]")
    st.title(":red[Features]")

    # Main title and subtitle
    st.markdown("## Car Selection Form")  # Main title in a larger font

    # City selection
    st.markdown("**City:**")  # Bold heading
    m_city = st.selectbox(label="", options=car_df['City'].unique())

    # Transmission selection
    st.markdown("**Transmission:**")  # Bold heading
    m_transmission = st.selectbox(label="", options=car_df['transmission'].unique(), key="transmission")


    # Car Brand selection
    st.markdown("**Car Brand:**")  # Bold heading
    m_oem = st.selectbox(label="", options=car_df['oem'].unique())

    # Filter models based on selected brand
    filtered_models = car_df[car_df['oem'] == m_oem]['model'].unique()
    st.markdown("**Car Model:**")  # Bold heading
    m_model = st.selectbox(label="", options=filtered_models)

    # Filter body types based on selected model
    filtered_bodytypes = car_df[car_df['oem'] == m_oem][car_df['model'] == m_model]['bodytype'].unique()

    # Body Type selection based on selected model
    st.markdown("**Body Type:**")  # Bold heading
    m_bodytype = st.selectbox(
        label="",
        options=filtered_bodytypes,
        key="bodytype_sidebar"
    )

    # Filter model years based on selected car brand and model
    filtered_model_years = sorted(car_df[(car_df['oem'] == m_oem) & (car_df['model'] == m_model)]['modelYear'].unique().astype(int))

    # Model Year selection based on selected car model
    st.markdown("**Model Year:**")  # Bold heading
    m_modelYear = st.selectbox(label="", options=filtered_model_years)

    # Filter the number of seats based on the selected Model Year
    filtered_seats = sorted(car_df[(car_df['oem'] == m_oem) & 
                                (car_df['model'] == m_model) & 
                                (car_df['modelYear'] == m_modelYear)]['seats'].unique().astype(int))

    # Number of Seats selection based on the selected Model Year
    st.markdown("**Number of Seats:**")  # Bold heading
    m_seats = st.selectbox(label="", options=filtered_seats)

    # Number of Owners selection
    st.markdown("**Number of Owners:**")  # Bold heading
    m_ownerNo = st.selectbox(label="", options=sorted(car_df['owner'].unique().astype(int)))


    # KMs Driven input
    st.markdown("**Enter KMs Driven:**")  # Bold heading
    m_km = st.number_input(
        label="",
        min_value=0,
        max_value=1000000,
        value=10000,
        step=1000,
        help="Enter the total kilometers driven by the car"
    )

    # Filter the number of gears based on the selected car model
    filtered_gears = sorted(car_df[(car_df['oem'] == m_oem) & (car_df['model'] == m_model)]['Gear_Box'].unique().astype(int))

    # Number of Gears selection based on the selected car model
    st.markdown("**Number of Gears:**")  # Bold heading
    m_gear = st.selectbox(label="", options=filtered_gears, key=f"gear_select_{m_oem}_{m_model}")


    # Filter fuel types based on the selected car brand and model
    filtered_fuel_types = car_df[(car_df['oem'] == m_oem) & (car_df['model'] == m_model)]['fueltype'].unique()

    # Fuel Type selection based on the selected car model
    st.markdown("**Fuel Type:**")  # Bold heading
    m_fuel_type = st.selectbox(label="", options=filtered_fuel_types)


    # Mileage inputs with unique keys
    if m_fuel_type == 'Electric':
        st.markdown("**Range (km/charge):**")  # Bold heading
        m_mileage = st.number_input(
            label="",
            min_value=0.0,
            max_value=800.0,
            value=300.0,
            step=1.0,
            help="Enter the range in kilometers per full charge",
            key="mileage_electric"
        )
    elif m_fuel_type in ['Petrol', 'Diesel']:
        st.markdown("**Mileage (km/l):**")  # Bold heading
        m_mileage = st.number_input(
            label="",
            min_value=0.0,
            max_value=50.0,
            value=15.0,
            step=0.1,
            help="Enter the mileage in kilometers per liter",
            key="mileage_petrol_diesel"
        )
    elif m_fuel_type == 'CNG':
        st.markdown("**Mileage (km/kg):**")  # Bold heading
        m_mileage = st.number_input(
            label="",
            min_value=0.0,
            max_value=40.0,
            value=20.0,
            step=0.1,
            help="Enter the mileage in kilometers per kg",
            key="mileage_cng"
        )

    # Filter drive types based on the selected car brand and model
    filtered_drive_types = car_df[(car_df['oem'] == m_oem) & (car_df['model'] == m_model)]['Drive_Type'].unique()

    # Drive Type selection based on the selected car model
    st.markdown("**Drive Type:**")  # Bold headings
    m_drivetype = st.selectbox(label="", options=filtered_drive_types)


    

    # Filter engine displacements based on the selected car brand and model
    filtered_engine_displacements = sorted(car_df[(car_df['oem'] == m_oem) & (car_df['model'] == m_model)]['Engine_CC'].unique())

    # Engine Displacement selection based on the selected car model
    st.markdown("**Engine Displacement:**")  # Bold heading
    m_Engine = st.selectbox(label="Engine Displacement (CC):", options=filtered_engine_displacements)


    # Insurance selection
    st.markdown("**Insurance:**")  # Bold heading
    m_Insurance = st.selectbox(label="", options=car_df['Insurance_Validity'].unique())

    with stylable_container(
        key="red_button",
        css_styles="""
            button {
                background-color: green;
                color: white;
                border-radius: 20px;
                background-image: linear-gradient(90deg, #0575e6 0%, #021b79 100%);
            }
        """
    ):
        pred_price_button = st.button("Estimate Used Car Price")
        
if pred_price_button:
    prediction_value = predict_resale_price(
        m_bodytype=m_bodytype,
        m_seats=m_seats,
        m_km=m_km,
        m_modelYear=m_modelYear,
        m_ownerNo=m_ownerNo,
        m_Engine=m_Engine,
        m_gear=m_gear,
        m_mileage=m_mileage,
        m_fuel_type=m_fuel_type,
        m_transmission=m_transmission,
        m_Insurance=m_Insurance,
        m_oem=m_oem,
        m_drivetype=m_drivetype,
        m_city=m_city
    )

    st.subheader(f"The estimated used car price is :blue[₹ {float(prediction_value) / 100000:,.2f} Lakhs]")
st.title('Car Price Distribution by Body Type')

# Bar plot using plotly express
fig = px.scatter(car_df, x='km', y='price', color='fueltype')
st.plotly_chart(fig)


# Scatter plot using plotly
st.title("Kms vs Price Scatter Plot")
# Create the scatter plot
fig = px.scatter(car_df, x='km', y='price', color='fueltype')

# Display the plot with a unique key
st.plotly_chart(fig, key="km_price_scatter")


st.title('Mileage Distribution')



# Histogram using plotly express
fig = px.histogram(car_df, x='Mileage', nbins=30, title='Distribution of Car Mileage')
st.plotly_chart(fig)


st.title('Correlation Heatmap')

# Compute correlation matrix
corr_matrix = car_df[['km','Engine_CC','price','seats','owner']].corr()

# Heatmap using plotly express
fig = px.imshow(corr_matrix, text_auto=True, title='Correlation Heatmap')
st.plotly_chart(fig)
