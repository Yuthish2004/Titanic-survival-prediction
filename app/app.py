"""
Titanic Survival Predictor — Streamlit App
---------------------------------------------
A simple interactive demo: enter passenger details, get a
survival prediction from the trained model.

Run locally:   streamlit run app/app.py
Deploy:        see README.md for Streamlit Community Cloud steps
"""

import streamlit as st
import pandas as pd
import joblib
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "models", "titanic_model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    with open(os.path.join(BASE_DIR, "models", "feature_columns.json")) as f:
        feature_columns = json.load(f)
    with open(os.path.join(BASE_DIR, "models", "model_meta.json")) as f:
        meta = json.load(f)
    return model, scaler, feature_columns, meta


model, scaler, feature_columns, meta = load_artifacts()

st.title("🚢 Titanic Survival Predictor")
st.caption(f"Model in use: **{meta['model_name']}** · trained on the classic Titanic passenger dataset")

st.markdown(
    "Enter a passenger's details below and the model will estimate their "
    "probability of survival, based on patterns learned from 891 real Titanic passengers."
)

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2,
                           help="1 = Upper, 2 = Middle, 3 = Lower")
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 0, 80, 28)
    embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"],
                             help="S = Southampton, C = Cherbourg, Q = Queenstown")

with col2:
    sibsp = st.number_input("Siblings / Spouses Aboard", 0, 8, 0)
    parch = st.number_input("Parents / Children Aboard", 0, 6, 0)
    fare = st.slider("Fare Paid (£)", 0.0, 512.0, 32.0)

# ---- Feature engineering (must mirror src/eda_and_preprocessing.py) ----
family_size = sibsp + parch + 1
is_alone = int(family_size == 1)
fare_per_person = fare / family_size if family_size > 0 else fare

if age <= 12:
    age_band = "child"
elif age <= 18:
    age_band = "teen"
elif age <= 35:
    age_band = "young_adult"
elif age <= 60:
    age_band = "adult"
else:
    age_band = "senior"

if sex == "male" and age > 16:
    who = "man"
elif sex == "female" and age > 16:
    who = "woman"
else:
    who = "child"

row = {col: 0 for col in feature_columns}
row["pclass"] = pclass
row["age"] = age
row["sibsp"] = sibsp
row["parch"] = parch
row["fare"] = fare
row["family_size"] = family_size
row["is_alone"] = is_alone
row["fare_per_person"] = fare_per_person

if sex == "male" and "sex_male" in row:
    row["sex_male"] = 1
if embarked == "Q" and "embarked_Q" in row:
    row["embarked_Q"] = 1
if embarked == "S" and "embarked_S" in row:
    row["embarked_S"] = 1
if who == "man" and "who_man" in row:
    row["who_man"] = 1
if who == "woman" and "who_woman" in row:
    row["who_woman"] = 1
for band in ["teen", "young_adult", "adult", "senior"]:
    key = f"age_band_{band}"
    if age_band == band and key in row:
        row[key] = 1

X_input = pd.DataFrame([row])[feature_columns]

if meta["needs_scaling"]:
    X_input_final = scaler.transform(X_input)
else:
    X_input_final = X_input

if st.button("Predict Survival", type="primary"):
    pred = model.predict(X_input_final)[0]
    proba = model.predict_proba(X_input_final)[0][1]

    if pred == 1:
        st.success(f"✅ Likely to **survive** — estimated probability: {proba:.1%}")
    else:
        st.error(f"❌ Likely **not** to survive — estimated probability of survival: {proba:.1%}")

    st.progress(float(proba))

st.markdown("---")
with st.expander("ℹ️ About this model"):
    st.write(
        "Trained on the classic Titanic dataset (891 passengers) using "
        "scikit-learn. Compared Logistic Regression and Random Forest "
        "with 5-fold cross-validation; selected by F1-score. "
        "See the project README for full methodology, metrics, and plots."
    )
