"""
Streamlit frontend for Gold Price Prediction
"""

import streamlit as st
import requests
import os

# Page configuration
st.set_page_config(
    page_title="Gold Price Prediction",
    page_icon="📊",
    layout="wide"
)

# API URL - Points to the deployed backend Space
API_URL = os.getenv("API_URL", "https://octaviodiego78-gold-predictions-backend.hf.space")

# Show connection info
st.sidebar.info(f"Backend API: {API_URL}")

# Title and description
st.title("Gold Price Prediction")

st.markdown("""
This application predicts tomorrow's gold price using a machine learning model trained on historical gold and S&P 500 data.
The prediction uses live data fetched from Yahoo Finance.
""")

st.markdown("---")

# Main content
st.subheader("Make a Prediction")

st.markdown("""
Click the button below to predict tomorrow's gold price. The system will:
1. Fetch the latest gold and S&P 500 prices from Yahoo Finance
2. Apply feature engineering (lags, moving averages, volatility)
3. Use the trained ML model to predict tomorrow's price
""")

# Predict button
predict_button = st.button("Predict Tomorrow's Gold Price", use_container_width=True, type="primary")

st.markdown("---")

# Make prediction when button is clicked
if predict_button:
    with st.spinner("Fetching live data and making prediction..."):
        try:
            # Check API health first
            health_response = requests.get(f"{API_URL}/health", timeout=10)
            
            if health_response.status_code != 200:
                st.error("API is not healthy. Please check the backend service.")
                st.json(health_response.json())
            else:
                health_data = health_response.json()
                
                if health_data['status'] != 'healthy':
                    st.warning("API components not fully loaded:")
                    st.json(health_data)
                else:
                    # Make prediction request
                    response = requests.post(
                        f"{API_URL}/predict",
                        json={"predict_tomorrow": True},
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display success message
                        st.success("Prediction Complete!")
                        
                        # Create metrics display
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                label="Predicted Gold Price",
                                value=f"${result['prediction']:.2f}",
                            )
                        
                        with col2:
                            st.metric(
                                label="Prediction Date",
                                value=result['predicted_date']
                            )
                        
                        with col3:
                            st.metric(
                                label="Model Type",
                                value=result['model_type']
                            )
                        
                        # Display additional details
                        st.markdown("---")
                        st.subheader("Details")
                        
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            st.write(f"**Today's Date:** {result['today_date']}")
                            st.write(f"**Tomorrow's Date:** {result['predicted_date']}")
                            st.write(f"**Predicted Price:** ${result['prediction']:.2f}")
                        
                        with col_right:
                            st.write(f"**Model:** {result['model_name'].split('.')[-1]}")
                            st.write(f"**Model Type:** {result['model_type']}")
                            st.write(f"**Status:** Active")
                        
                        # Show full API response in expander
                        with st.expander("View Full API Response"):
                            st.json(result)
                    
                    else:
                        st.error(f"API Error: {response.status_code}")
                        try:
                            st.json(response.json())
                        except:
                            st.text(response.text)
                    
        except requests.exceptions.Timeout:
            st.error("Request timed out. The API might be slow or processing large amounts of data.")
            st.info("Please try again in a few moments.")
            
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot connect to API at {API_URL}")
            st.info("Make sure the backend API is running and the URL is correct.")
            st.code(f"Expected API URL: {API_URL}", language="text")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check the logs for more details.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit, FastAPI, TensorFlow, and MLflow</p>
    <p>Data source: Yahoo Finance | Model: Databricks MLflow</p>
</div>
""", unsafe_allow_html=True)

