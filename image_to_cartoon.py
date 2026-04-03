#1. importing libraries
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

#2. Page setup
st.set_page_config(page_title="🎨 AI Image Art Generator", layout="wide")

st.title("🎨 AI Image Art Generator")
st.write("Convert your photos into sketch, cartoon and artistic styles 🚀")

#3. Upload MULTIPLE images
uploaded_files = st.file_uploader(
    "Upload Images", 
    type=["jpg", "png", "jpeg"], 
    accept_multiple_files=True
)

#4. Select mode
mode = st.selectbox(
    "Choose Style",
    ["Pencil Sketch", "Cartoon", "Black & White", "Edge Detection"]
)

#5. Process images
if uploaded_files:

    for uploaded_file in uploaded_files:

        # Read image
        image = Image.open(uploaded_file)
        img = np.array(image)

        # 🔥 Resize for consistency (fix mobile issue)
        height, width = img.shape[:2]
        if width > 800:
            scale = 800 / width
            img = cv2.resize(img, (int(width * scale), int(height * scale)))

        # Convert RGB → BGR (ONLY ONCE)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(image, use_column_width=True)

        # =========================
        # IMAGE PROCESSING LOGIC
        # =========================

        if mode == "Pencil Sketch":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Normalize brightness (fix white issue)
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

            blur = cv2.GaussianBlur(gray, (15, 15), 0)
            sketch = cv2.divide(gray, blur, scale=256)

            # Reduce over-brightness
            sketch = cv2.convertScaleAbs(sketch, alpha=0.9, beta=-20)

            result = sketch

        elif mode == "Cartoon":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 7)

            edges = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                9, 5
            )

            # Strong smoothing (better cartoon effect)
            color = cv2.bilateralFilter(img, 15, 250, 250)

            cartoon = cv2.bitwise_and(color, color, mask=edges)
            result = cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB)

        elif mode == "Black & White":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            result = gray

        else:  # Edge Detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            result = edges

        # =========================
        # DISPLAY RESULT
        # =========================

        with col2:
            st.subheader("Result")

            # Handle grayscale properly
            if len(result.shape) == 2:
                st.image(result, use_column_width=True, clamp=True)
                final_img = Image.fromarray(result.astype(np.uint8))
            else:
                st.image(result, use_column_width=True)
                final_img = Image.fromarray(result.astype(np.uint8))

        # =========================
        # DOWNLOAD BUTTON
        # =========================

        buf = io.BytesIO()
        final_img.save(buf, format="PNG")

        st.download_button(
            label="📥 Download",
            data=buf.getvalue(),
            file_name=f"{mode}_{uploaded_file.name}",
            mime="image/png"
        )

        st.divider()

# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by **Rudra Joshi**")
