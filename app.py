'''
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Define expected feature names
features = ['Age', 'ALT', 'BMI', 'DM.IFG', 'FBG', 'GGT', 'TG', 'AST.PLT']

# Load model and scaler
with open("fibrosis.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("A Machine learning based Combined Approach for Liver Fibrosis Diagnosis in NAFLD Using Biomarkers and Demographics")

# Collect user input for all 8 features
input_data = []
for feature in features:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    input_data.append(value)

# Predict button
if st.button("Predict"):
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data], columns=features)
        
        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction = model.predict(scaled_input)
        
        # Display result
        st.success(f"Predicted Fibrosis Stage: {prediction[0]}")
    except Exception as e:
        st.error(f"Error: {e}")

'''
import streamlit as st
import pickle
import pandas as pd

# Define features
features = ['Age', 'ALT', 'BMI', 'DM.IFG', 'FBG', 'GGT', 'TG', 'AST.PLT']

# Load model and scaler
with open("fibrosis.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Title
st.title("A Machine learning based Combined Approach for Liver Fibrosis Diagnosis in NAFLD Using Biomarkers and Demographics")
st.markdown("Enter all values below and click **Predict** to know the fibrosis stage.")

# Input fields (shown all together)
user_inputs = {}
for feature in features:
    user_inputs[feature] = st.text_input(f"{feature}:")

# Predict button
if st.button("Predict"):
    # Check if all fields are filled
    if all(user_inputs[feature].strip() != '' for feature in features):
        try:
            # Convert inputs to float
            input_values = [float(user_inputs[feature]) for feature in features]
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_values], columns=features)

            # Scale input
            scaled_input = scaler.transform(input_df)

            # Predict
            prediction = model.predict(scaled_input)

            # Interpret result
            fibrosis_stage = "Early Fibrosis" if str(prediction[0]).strip() in ['0'] else "Advanced Fibrosis"

            # Show result
            st.success(f"Predicted Fibrosis Stage: {fibrosis_stage}")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please fill in **all fields** before predicting.")


'''
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64

# Function to set the background image
def set_background(image_file):
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{image_file}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Convert image to base64 format
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Call the function to set background
image_path = "background.png"  # Ensure this image is in your project directory
set_background(get_base64_image(image_path))


# Define expected feature names
features = ['Age', 'ALT', 'BMI', 'DM.IFG', 'FBG', 'GGT', 'TG', 'AST.PLT']

# Load model and scaler
with open("fibrosis.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("A Machine Learning Based Combined Approach for Liver Fibrosis Diagnosis in NAFLD")

st.markdown("### Enter patient details below:")

# Collect user input for all 8 features without +/- buttons
input_data = []

all_empty = True  # Track if all inputs are empty
for feature in features:
    value = st.text_input(f"Enter {feature}:", value="")  # Text input instead of number input
    try:
        value = float(value) if value.strip() else 0.0  # Convert input to float, default to 0.0 if empty
    except ValueError:
        st.error(f"Invalid input for {feature}. Please enter a numeric value.")
        st.stop()

    if value is not None:
        all_empty = False  # If at least one value is entered, set flag to False

    input_data.append(value if value is not None else 0.0)  # Default empty values to 0.0


# Predict button
if st.button("Predict"):
    if all_empty:
        st.warning("Please enter values before making a prediction.")
    else:
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data], columns=features)
            
            # Scale input
            scaled_input = scaler.transform(input_df)

            # Predict
            prediction = model.predict(scaled_input)
            
            # Mapping 0 -> Early Fibrosis, 1 -> Advanced Fibrosis
            fibrosis_stage = "Early Fibrosis" if prediction == 0 else "Advanced Fibrosis"
            
            # Display result
            st.success(f"Predicted Fibrosis Stage: {fibrosis_stage}")
        except Exception as e:
            st.error(f"Error: {e}")
'''

'''
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Define expected feature names
features = ['Age', 'ALT', 'BMI', 'DM.IFG', 'FBG', 'GGT', 'TG', 'AST.PLT']

# Load model and scaler
with open("fibrosis.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# App title
st.title("Liver Fibrosis Diagnosis in NAFLD")

st.markdown("### Enter patient details below:")

# Input collection
input_data = []
all_entered = True

for feature in features:
    value = st.text_input(f"{feature}:", value="")
    
    if value.strip() == "":
        all_entered = False
        input_data.append(None)
    else:
        try:
            float_val = float(value)
            input_data.append(float_val)
        except ValueError:
            st.error(f"Please enter a valid number for {feature}.")
            st.stop()

# Prediction
if st.button("Predict"):
    if not all_entered or None in input_data:
        st.warning("Please enter all 8 values before prediction.")
    else:
        try:
            input_df = pd.DataFrame([input_data], columns=features)
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)
            result = "Early Fibrosis" if prediction == 0 else "Advanced Fibrosis"
            st.success(f"Predicted Fibrosis Stage: {result}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

 '''           

'''import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Define expected feature names
features = ['Age', 'ALT', 'BMI', 'DM.IFG', 'FBG', 'GGT', 'TG', 'AST.PLT']

# Load model and scaler
with open("fibrosis.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit UI
st.title("A Machine Learning Based Combined Approach for Liver Fibrosis Diagnosis in NAFLD")

st.markdown("### Enter patient details below:")

# Collect user input for all 8 features without +/- buttons
input_data = []

all_empty = True  # Track if all inputs are empty
for feature in features:
    value = st.text_input(f"Enter {feature}:", value="")  # Text input instead of number input
    try:
        value = float(value) if value.strip() else 0.0  # Convert input to float, default to 0.0 if empty
    except ValueError:
        st.error(f"Invalid input for {feature}. Please enter a numeric value.")
        st.stop()

    if value is not None:
        all_empty = False  # If at least one value is entered, set flag to False

    input_data.append(value if value is not None else 0.0)  # Default empty values to 0.0

# Predict button
if st.button("Predict"):
    if all_empty:
        st.warning("Please enter values before making a prediction.")
    else:
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data], columns=features)
            
            # Scale input
            scaled_input = scaler.transform(input_df)

            # Predict
            prediction = model.predict(scaled_input)
            
            # Mapping 0 -> Early Fibrosis, 1 -> Advanced Fibrosis
            fibrosis_stage = "Early Fibrosis" if prediction == 0 else "Advanced Fibrosis"
            
            # Display result
            st.success(f"Predicted Fibrosis Stage: {fibrosis_stage}")
        except Exception as e:
            st.error(f"Error: {e}")'''
