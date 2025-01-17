#importing libraries
import pickle
import pandas as pd
import streamlit as slt
import numpy as np

# page setting
slt.set_page_config(layout="wide")
slt.markdown(
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

# Sidebar for user inputs
with slt.sidebar:
    slt.image(
        "https://stimg2.cardekho.com/images/carNewsimages/userimages/650X420/30183/1672738680556/GeneralNew.jpg",
        width=400
    )
slt.header(':blue[Used Car-Price Prediction 🚗]')

# Load data
df = pd.read_csv("final_df.csv")
print(df.columns)

# Streamlit interface
col1, col2 = slt.columns(2)
with col1:
    Ft = slt.selectbox("Fuel type", ['Petrol', 'Diesel', 'Lpg', 'Cng', 'Electric'])

    Bt = slt.selectbox("Body type", ['Hatchback', 'SUV', 'Sedan', 'MUV', 'Coupe', 'Minivans',
                                     'Convertibles', 'Hybrids', 'Wagon', 'Pickup Trucks'])
    Tr = slt.selectbox("Transmission", ['Manual', 'Automatic'])

    Owner = slt.selectbox("Owner", [0, 1, 2, 3, 4, 5])

    Brand = slt.selectbox("Brand", options=df['Brand'].unique())

    filtered_models = df[(df['Brand'] == Brand) & (df['body type'] == Bt) & (df['Fuel type'] == Ft)]['model'].unique()


    Model=slt.selectbox("Model",options=filtered_models)

    Model_year = slt.selectbox("Model Year", options=sorted(df['modelYear'].unique()))
    
    
    IV = slt.selectbox("Insurance Validity", ['Third Party insurance', 'Comprehensive', 'Third Party',
                                              'Zero Dep', '2', '1', 'Not Available'])
    
    Km = slt.slider("Kilometers Driven", min_value=100, max_value=100000, step=1000)

    ML = slt.number_input("Mileage", min_value=5, max_value=50, step=1)  

    seats = slt.selectbox("Seats", options=sorted(df['Seats'].unique()))
    
    color = slt.selectbox("Color", df['Color'].unique())

    city = slt.selectbox("City", options=df['City'].unique())

with col2:
    Submit = slt.button("Estimate Used Car Price")

    if Submit:
    
       # load the model,scaler and encoder
       with open('pipeline.pkl','rb') as files:
        pipeline=pickle.load(files)

        # input data
        new_df=pd.DataFrame({
        'Fuel type': Ft,
        'body type':Bt,
        'transmission':Tr,
        'ownerNo':Owner,
        'Brand':Brand,
        "model":Model,
        'modelYear':Model_year,
        'Insurance Validity':IV,
        'Kms Driven':Km,
        'Mileage':ML,
        'Seats':seats,
        'Color':color,
        'City': city},index=[0])
        
        # Display the selected details
        data = [Ft, Bt, Tr, Owner,Brand, Model,Model_year, IV, Km, ML, seats, color, city]

        slt.write(data)

        # Make prediction using the trained pipeline
        prediction = pipeline.predict(new_df)

        # Get the brand and price prediction for the car
        car_brand = new_df['Brand'].iloc[0]
        predicted_price = round(prediction[0], 2)

        # Display the result with better formatting
        slt.write(f"🚗 **Car Price Prediction** 🚗\n")
        slt.write(f"**Brand**: {car_brand}")
        slt.write(f"**Predicted Price**: ₹{predicted_price} Lakhs")


