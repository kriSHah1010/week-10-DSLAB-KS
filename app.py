import streamlit as st
import pandas as pd
import numpy as np
from apputil import predict_rating

# --- Configuration & Shared Constants ---
st.set_page_config(
    layout="centered", 
    page_title="Wk 10: ML Deployment Assignment",
    initial_sidebar_state="collapsed" # Changed to collapsed since we moved content
)

# Constants for UI
KNOWN_ROASTS = ['Light', 'Medium-Light', 'Medium', 'Medium-Dark', 'Dark', 'Very Light', 'Very Dark']
UNKNOWN_ROAST = ' (Unknown to Model 2)'
ROAST_OPTIONS = KNOWN_ROASTS + [UNKNOWN_ROAST]

# --- Title and Project Overview ---
st.markdown(
    """
    # ☕ Week 10 Assignment: Coffee Rating Predictor
    This application deploys the machine learning models trained for the assignment, demonstrating conditional logic and text processing.
    """
)

# --- Consolidated Project Background and Logic (New Section) ---
st.markdown(
    """
    ## Project Background and Logic Breakdown

    The complete solution involves three distinct models, all trained in the `train.py` script:

    ### **Model Training Summary**
    * **Ex 1**: Trained `LinearRegression` on `100g_USD` → saved as `model_1.pickle`.
    * **Ex 2**: Trained `DecisionTreeRegressor` on `100g_USD` & `roast` → saved as `model_2.pickle` and `roast_map.pickle`.
    * **Bonus**: Trained `LinearRegression` on TF-IDF of `desc_3` → saved as `model_3.pickle` and `tfidf_vectorizer.pickle`.

    ### **Application Logic (Exercise 3)**
    The **Price/Roast Prediction** tab implements the core logic required by **Exercise 3**:
    1.  **If Roast is Known:** The app uses the complex **Model 2** (Decision Tree Regressor), which utilizes both the **Price** and **Roast** features (based on Ex 2 training).
    2.  **If Roast is Unknown:** The app falls back to the simple **Model 1** (Linear Regression), using only the **Price** feature (based on Ex 1 training).

    ### **Bonus Logic**
    The **Text Description Prediction** tab implements the **Bonus Exercise**:
    * **Model Used:** **Model 3** (Linear Regression) is used, which was trained on the TF-IDF vectorization of the coffee descriptions. The user's input text is vectorized using the saved `tfidf_vectorizer.pickle` before prediction.

    ---
    """
)


# --- Main Content: Tabbed Interface ---
tab1, tab2 = st.tabs(["💰 Price/Roast Prediction (Ex 3 Logic)", "✍️ Text Description Prediction (Bonus)"])

# ==============================================================================
# --- Tab 1: Price and Roast Prediction (Exercises 1, 2, 3 Logic) ---
# ==============================================================================
with tab1:
    st.subheader("Predict Rating from Price and Roast")
    
    with st.form("price_roast_form"):
        # Input 1: Price
        usd_price = st.number_input(
            "1️⃣ Price per 100g (USD):", 
            min_value=1.00, 
            max_value=100.00, 
            value=18.00, 
            step=0.10, 
            format="%.2f"
        )

        # Input 2: Roast Type
        selected_roast = st.selectbox(
            "2️⃣ Select Roast Category:",
            options=ROAST_OPTIONS,
            index=2 # Default to Medium
        )
        
        submitted_price_roast = st.form_submit_button("Predict Rating (Model 1 or 2)")

    if submitted_price_roast:
        # Prepare Input Data
        roast_for_prediction = selected_roast
        if selected_roast == UNKNOWN_ROAST:
            roast_for_prediction = '---UNKNOWN_MARKER---' 
            
        df_X = pd.DataFrame([{"100g_USD": usd_price, "roast": roast_for_prediction}])
        y_pred_array = predict_rating(df_X, text=False)
        predicted_rating = y_pred_array[0]

        st.markdown("---")
        if pd.isna(predicted_rating):
            st.error("❌ Prediction Failed. Please ensure you have run **python train.py**.")
        else:
            is_known = selected_roast in KNOWN_ROASTS
            model_name = "Decision Tree Regressor (Model 2)" if is_known else "Linear Regression (Model 1)"
            
            # Display result with styling
            st.markdown("#### ✅ Predicted Rating")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label="Score", value=f"{predicted_rating:.2f} / 100")
            with col2:
                st.info(f"**Model Used (Exercise 3 Logic):** {model_name}")
                st.caption("Model 2 is used when Roast is specified; Model 1 is used as a fallback based only on Price.")


# ==============================================================================
# --- Tab 2: Text Description Prediction (Bonus Exercise Logic) ---
# ==============================================================================
with tab2:
    st.subheader("Predict Rating from Coffee Description")

    with st.form("text_form"):
        description = st.text_area(
            "💬 Enter Coffee Description:",
            value="A complex flavor profile with strong notes of tropical fruit, cocoa, and a bright acidity.",
            height=150
        )
        submitted_text = st.form_submit_button("Predict Rating (Model 3)")

    if submitted_text:
        # Input must be a DataFrame with a 'text' column
        df_X_text = pd.DataFrame({"text": [description]})
        
        # Get Prediction (text=True to use Model 3)
        y_pred_array = predict_rating(df_X_text, text=True)
        predicted_rating = y_pred_array[0]

        st.markdown("---")
        if pd.isna(predicted_rating):
            st.error("❌ Prediction Failed. Please ensure you have run **python train.py**.")
        else:
            # Display result with styling
            st.markdown("#### ✅ Predicted Rating")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(label="Score", value=f"{predicted_rating:.2f} / 100")
            with col2:
                st.success(f"**Model Used (Bonus Exercise):** Linear Regression (Model 3) on TF-IDF Vectors")
                st.caption("The prediction is based on the weights assigned to the words in description")
            
            
