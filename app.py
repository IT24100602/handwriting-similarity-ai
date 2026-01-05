import streamlit as st
from PIL import Image
import cv2
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Handwriting Similarity Analyzer",
    page_icon="✍️",
    layout="wide"
)

# ---------- STYLE ----------
st.markdown("""
<style>
.title {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 30px;
}
.card {
    background: white;
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
}
.score {
    font-size: 3rem;
    font-weight: 800;
}
.high { color: #27ae60; }
.mid { color: #f39c12; }
.low { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

# ---------- FUNCTIONS ----------
def preprocess_image(image):
    img = np.array(image)

    # Convert to grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    img = cv2.resize(img, (300, 300))

    # Blur to remove noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    return img


def compare_handwriting(img1, img2):
    # Mean Squared Error
    error = np.mean((img1.astype("float") - img2.astype("float")) ** 2)

    # Convert error to similarity percentage
    similarity = max(0, 100 - (error / 40))
    return similarity


# ---------- UI ----------
st.markdown('<div class="title">✍️ Handwriting Similarity Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload two handwriting images to compare visual similarity</div>',
    unsafe_allow_html=True
)

st.info(
    "⚠️ Best results are obtained when both images contain **similar or same text**. "
    "Different text written by the same person may result in lower similarity scores."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Handwriting Sample 1")
    img1_file = st.file_uploader(
        "Upload first image",
        type=["jpg", "png", "jpeg"],
        key="img1"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Handwriting Sample 2")
    img2_file = st.file_uploader(
        "Upload second image",
        type=["jpg", "png", "jpeg"],
        key="img2"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- PROCESS ----------
if img1_file and img2_file:
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    p1 = preprocess_image(img1)
    p2 = preprocess_image(img2)

    e1 = cv2.Canny(p1, 50, 150)
    e2 = cv2.Canny(p2, 50, 150)

    st.subheader("Processed Images")
    c1, c2 = st.columns(2)

    with c1:
        st.image(p1, caption="Processed Sample 1", clamp=True)
        st.image(e1, caption="Edges Sample 1", clamp=True)

    with c2:
        st.image(p2, caption="Processed Sample 2", clamp=True)
        st.image(e2, caption="Edges Sample 2", clamp=True)

    similarity = compare_handwriting(p1, p2)

    st.subheader("Similarity Result")

    if similarity > 75:
        status = "Very High Similarity"
        cls = "high"
    elif similarity > 50:
        status = "Moderate Similarity"
        cls = "mid"
    else:
        status = "Low Similarity"
        cls = "low"

    st.markdown(
        f'<div class="card" style="text-align:center;">'
        f'<div class="score {cls}">{similarity:.2f}%</div>'
        f'<p class="{cls}">{status}</p>'
        f'</div>',
        unsafe_allow_html=True
    )

else:
    st.warning("Please upload **both** handwriting images to start comparison.")

st.markdown(
    "<hr><center>Built using Python • OpenCV • Streamlit</center>",
    unsafe_allow_html=True
)
