import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model_Random_Forest.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

# App Title
st.title("📰 Fake News Detection App")
st.write("Paste a news headline or article below, and the AI will predict whether it is **Real** or **Fake**.")

# Text input
user_input = st.text_area("Enter News Text:", height=200)

# Prediction button
if st.button("Check News"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some news text to analyze.")
    else:
        # Convert input using vectorizer
        features = vectorizer.transform([user_input])
        
        # Predict class
        prediction = model.predict(features)[0]
        
        # Predict probability (works only for models that support predict_proba)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = round(max(proba) * 100, 2)
        else:
            confidence = None
        
        # Show result
        if prediction == 1:
            st.success(f"✅ This looks like **Real News**")
        else:
            st.error(f"❌ This looks like **Fake News**")
        
        if confidence:
            st.info(f"Confidence: {confidence}%")
