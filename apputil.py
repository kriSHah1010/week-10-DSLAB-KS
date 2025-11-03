import pandas as pd
import pickle
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.base import BaseEstimator 

# Define file paths for the saved assets
MODEL_1_PATH = 'model_1.pickle'
MODEL_2_PATH = 'model_2.pickle'
ROAST_MAP_PATH = 'roast_map.pickle'
MODEL_3_PATH = 'model_3.pickle'
VECTORIZER_PATH = 'tfidf_vectorizer.pickle'

# Global variables to store loaded assets
model_1: BaseEstimator = None
model_2: BaseEstimator = None
roast_map: dict = None
model_3: BaseEstimator = None
tfidf_vectorizer: TfidfVectorizer = None 
is_loaded = False

def load_models():
    """Loads all necessary models, the roast mapping, and the TF-IDF vectorizer."""
    global model_1, model_2, roast_map, model_3, tfidf_vectorizer, is_loaded
    if is_loaded:
        return

    try:
        # Check if all files exist before attempting to load
        required_paths = [MODEL_1_PATH, MODEL_2_PATH, ROAST_MAP_PATH, MODEL_3_PATH, VECTORIZER_PATH]
        if not all(os.path.exists(p) for p in required_paths):
            print("--- WARNING ---")
            print("Model files are missing. Did you run 'python train.py'?")
            print("--- WARNING ---")
            return

        # Load all assets
        with open(MODEL_1_PATH, 'rb') as f: model_1 = pickle.load(f)
        with open(MODEL_2_PATH, 'rb') as f: model_2 = pickle.load(f)
        with open(ROAST_MAP_PATH, 'rb') as f: roast_map = pickle.load(f)
        with open(MODEL_3_PATH, 'rb') as f: model_3 = pickle.load(f)
        with open(VECTORIZER_PATH, 'rb') as f: tfidf_vectorizer = pickle.load(f)
            
        is_loaded = True
        print("All models, map, and vectorizer loaded successfully.")
    except Exception as e:
        print(f"An unexpected error occurred during asset loading: {e}")

def predict_rating(df_X: pd.DataFrame, text: bool = False) -> np.ndarray:
    """
    Predicts the rating using Model 1, Model 2, or Model 3 (if text=True).
    
    Args:
        df_X: DataFrame containing input features.
        text: If True, uses Model 3 (text-based prediction). If False, uses Model 1/2.
              
    Returns:
        An array of predicted rating values. Returns NaNs if models failed to load.
    """
    load_models()
    
    if not is_loaded:
        return np.full(len(df_X), np.nan)

    if text:
        # --- BONUS EXERCISE LOGIC (Model 3) ---
        if 'text' not in df_X.columns:
            return np.full(len(df_X), np.nan)
        
        # Transform the input text using the fitted vectorizer
        X_input_text = tfidf_vectorizer.transform(df_X['text'].fillna(''))

        # Predict using Model 3 (Linear Regression on TF-IDF features)
        y_pred = model_3.predict(X_input_text)
        return y_pred.flatten()
        
    else:
        # --- EXERCISE 3 LOGIC (Model 1 / Model 2) ---
        if '100g_USD' not in df_X.columns or 'roast' not in df_X.columns:
            return np.full(len(df_X), np.nan)
            
        y_pred = np.zeros(len(df_X))
        for i, row in df_X.iterrows():
            usd_price = row['100g_USD']
            roast_type = row['roast']
            
            if roast_type in roast_map:
                # Case 1: Use Model 2 (Decision Tree) with both features
                roast_label = roast_map[roast_type]
                X_input_2 = pd.DataFrame([[usd_price, roast_label]], columns=['100g_USD', 'roast_label'])
                y_pred[i] = model_2.predict(X_input_2)[0]
            else:
                # Case 2: Use Model 1 (Linear Regression) with only 100g_USD
                X_input_1 = pd.DataFrame([[usd_price]], columns=['100g_USD'])
                y_pred[i] = model_1.predict(X_input_1)[0]
                
        return y_pred
