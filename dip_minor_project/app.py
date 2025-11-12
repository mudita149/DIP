# app.py
import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
import io
from watermark_models import ImagePreprocessor, VisibleWatermarker, InvisibleWatermarker

# ------------------------------------------------------------
# Load models
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    with open("image_preprocessing.pkl", "rb") as f:
        preprocess_model = pickle.load(f)
    with open("visible_watermarker.pkl", "rb") as f:
        visible_model = pickle.load(f)
    with open("invisible_watermarker.pkl", "rb") as f:
        invisible_model = pickle.load(f)
    return preprocess_model, visible_model, invisible_model

preprocess_model, visible_model, invisible_model = load_models()

# ------------------------------------------------------------
# Streamlit Page Configuration
# ------------------------------------------------------------
st.set_page_config(page_title=" DWT Image Watermarking", layout="centered")
st.title(" DWT Image Watermarking")

st.markdown("""
Upload your **Main Image** and **Watermark Image**,  
then select whether you want **Visible** or **Invisible** watermarking.  
Preprocessing happens automatically in the background.
""")

# ------------------------------------------------------------
# Upload Section
# ------------------------------------------------------------
main_file = st.file_uploader("📸 Upload Main Image", type=["jpg", "jpeg", "png"])
wm_file = st.file_uploader("💧 Upload Watermark Image", type=["jpg", "jpeg", "png"])

operation = st.selectbox("Select Watermark Type", ["-- Select --", "Visible Watermarking", "Invisible Watermarking"])

# ------------------------------------------------------------
# Processing
# ------------------------------------------------------------
if main_file and wm_file and operation != "-- Select --":

    # Convert uploaded images to numpy arrays (BGR for OpenCV)
    main_img_rgb = np.array(Image.open(main_file).convert("RGB"))
    wm_img_rgb = np.array(Image.open(wm_file).convert("RGB"))

    # Convert to BGR for OpenCV operations
    main_img = cv2.cvtColor(main_img_rgb, cv2.COLOR_RGB2BGR)
    wm_img = cv2.cvtColor(wm_img_rgb, cv2.COLOR_RGB2BGR)

    # Show input images (as uploaded)
    st.image(
        [main_img_rgb, wm_img_rgb],
        caption=["Main Image (Input)", "Watermark Image (Input)"],
        width=300,
    )

    # Intensity control
    alpha_default = 0.4 if operation == "Visible Watermarking" else 0.05
    alpha = st.slider("Watermark Intensity", 0.01, 1.0, alpha_default, step=0.01)

    if st.button("🚀 Apply Watermark"):
        with st.spinner("Processing and embedding watermark..."):
            # Step 1: Preprocess (auto, hidden)
            preprocessed_main = preprocess_model.apply(main_img)

            # Step 2: Watermark embedding
            if operation == "Visible Watermarking":
                visible_model.alpha = alpha
                result_bgr = visible_model.apply(preprocessed_main, wm_img)
            else:
                invisible_model.alpha = alpha
                result_bgr = invisible_model.apply(preprocessed_main, wm_img)

        # Convert BGR → RGB for display
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        # Show final output only
        st.image(result_rgb, caption="✅ Final Watermarked Image", use_container_width=True)

        # Download output
        buf = io.BytesIO()
        Image.fromarray(result_rgb).save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Watermarked Image",
            data=buf.getvalue(),
            file_name=f"{operation.lower().replace(' ', '_')}.png",
            mime="image/png",
        )

elif operation == "-- Select --":
    st.info("👆 Please select a watermark type after uploading both images.")
