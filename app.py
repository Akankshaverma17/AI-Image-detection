import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd

st.set_page_config(page_title="AI vs Real Image Detection", layout="wide")

st.title("🧠 AI vs Real Image Detection")
st.write("Upload an image to check whether it is AI-generated or Real.")

# Load HuggingFace model (cached)
@st.cache_resource
def load_model():
    detector = pipeline("image-classification", 
                        model="umm-maybe/AI-image-detector")
    return detector

detector = load_model()

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Prediction
    result = detector(image)
    
    label = result[0]["label"]
    confidence = result[0]["score"]

    st.subheader("Prediction Result")
    st.write(f"**Result:** {label}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")

    # Save history
    st.session_state.history.append({
        "Filename": uploaded_file.name,
        "Prediction": label,
        "Confidence (%)": round(confidence*100, 2)
    })

# Filter section
if st.session_state.history:
    st.subheader("📊 Prediction History")
    df = pd.DataFrame(st.session_state.history)

    filter_option = st.selectbox(
        "Filter Results",
        ["All"] + list(df["Prediction"].unique())
    )

    if filter_option != "All":
        df = df[df["Prediction"] == filter_option]

    st.dataframe(df, use_container_width=True)