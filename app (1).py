import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re

# Set page configuration
st.set_page_config(
    page_title="Pizza Category Classifier",
    page_icon="🍕",
    layout="centered"
)

@st.cache_resource
def load_model_artifacts():
    """Loads the trained model and label encoders securely."""
    try:
        model = joblib.load('pizza_category_classifier.pkl')
        le_size = joblib.load('size_encoder.pkl')
        le_target = joblib.load('target_encoder.pkl')
        return model, le_size, le_target
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

artifacts = load_model_artifacts()
model, le_size, le_target = artifacts

def sanitize_input(user_input):
    """Sanitizes user numeric input vectors to maintain pipeline security profile."""
    sanitized = re.sub(r'[^0-9.\s,]', '', str(user_input))
    return sanitized.strip()

# App Header
st.title("🍕 Pizza Category Predictive Classifier")
st.markdown("""
This production-grade AI system predicts the category of a pizza order vector by parsing transaction properties.
""")

if model is not None:
    st.subheader("🛠️ Transaction Feature Vector Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        quantity_input = st.text_input("Order Quantity", value="1")
        unit_price_input = st.text_input("Unit Price ($)", value="18.50")
        
    with col2:
        total_price_input = st.text_input("Calculated Total Price ($)", value="18.50")
        # Extract dynamic classes safely from target encoder mapping arrays
        size_options = [str(x) for x in le_size.classes_]
        selected_size = st.selectbox("Select Pizza Size Matrix", options=size_options, index=size_options.index('L') if 'L' in size_options else 0)

    if st.button("🚀 Execute Classifier Inference"):
        # Validate and clean inputs
        clean_qty = sanitize_input(quantity_input)
        clean_unit = sanitize_input(unit_price_input)
        clean_tot = sanitize_input(total_price_input)
        
        try:
            qty_val = float(clean_qty) if clean_qty else 1.0
            unit_val = float(clean_unit) if clean_unit else 18.50
            tot_val = float(clean_tot) if clean_tot else qty_val * unit_val
            size_encoded = le_size.transform([selected_size])
            
            # Form structural dataframe matching original training configurations
            feature_names = ['quantity', 'unit_price', 'total_price', 'pizza_size_encoded']
            input_df = pd.DataFrame([[qty_val, unit_val, tot_val, size_encoded]], columns=feature_names)
            
            # Predict
            prediction_encoded = model.predict(input_df)
            predicted_category = le_target.inverse_transform([prediction_encoded])
            
            # Display results
            st.success("### 🎉 Prediction Vector Resolved Matrix")
            st.metric(label="Predicted Pizza Classification", value=str(predicted_category))
            
            # Feature diagnostics
            st.info(f"**Diagnostic Payload Context:** Quantized input values parsed successfully as Size: `{selected_size}`, Quantity: `{qty_val}`, Unit Price: `${unit_val}`, Total: `${tot_val}`.")
            
        except Exception as e:
            st.error(f"Execution Error: Input structural matrix parsing mapping failed. Context: {e}")
else:
    st.warning("⚠️ Application State Halted: Please ensure 'pizza_category_classifier.pkl', 'size_encoder.pkl', and 'target_encoder.pkl' files are placed inside the repository root namespace directory.")
