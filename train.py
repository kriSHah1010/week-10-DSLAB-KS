import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Load Data
DATA_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
try:
    df = pd.read_csv(DATA_URL)
    # Fill NaN values in 'desc_3' with an empty string for text processing
    df['desc_3'] = df['desc_3'].fillna('')
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

y = df['rating']

# --- Exercise 1: Linear Regression (Predict rating from 100g_USD) ---
model_1 = LinearRegression()
X1 = df[['100g_USD']]
model_1.fit(X1, y)
with open('model_1.pickle', 'wb') as f:
    pickle.dump(model_1, f)
print("Saved model_1.pickle (Ex 1)")

# --- Exercise 2: Decision Tree Regressor (Predict rating from 100g_USD and roast) ---
roast_map = {
    'Medium-Light': 1, 'Light': 2, 'Medium': 3, 'Medium-Dark': 4, 
    'Dark': 5, 'Very Light': 6, 'Very Dark': 7
}
df['roast_label'] = df['roast'].map(roast_map)
with open('roast_map.pickle', 'wb') as f:
    pickle.dump(roast_map, f)

model_2 = DecisionTreeRegressor(random_state=42)
X2 = df[['100g_USD', 'roast_label']]
model_2.fit(X2, y)
with open('model_2.pickle', 'wb') as f:
    pickle.dump(model_2, f)
print("Saved model_2.pickle (Ex 2)")
print("Saved roast_map.pickle (Ex 2)")


# --- Bonus Exercise: TF-IDF Vectorization and Model 3 ---
# 1. Initialize and Fit the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_text = tfidf_vectorizer.fit_transform(df['desc_3'])

# 2. Save the fitted Vectorizer
with open('tfidf_vectorizer.pickle', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
print("Saved tfidf_vectorizer.pickle (Bonus)")

# 3. Train Model 3 (Linear Regression on vectorized text data)
model_3 = LinearRegression()
model_3.fit(X_text, y)

# 4. Save Model 3
with open('model_3.pickle', 'wb') as f:
    pickle.dump(model_3, f)
print("Saved model_3.pickle (Bonus)")

print("\nTraining complete. All 5 files saved.")