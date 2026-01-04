from skimage.metrics import structural_similarity as ssim
import streamlit as st
from PIL import Image
import cv2
import numpy as np

# Custom CSS for elegant styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #f8f9fa;
    }

    /* Title styling */
    .title-text {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #3498db, #2c3e50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 10px 0;
        font-family: 'Georgia', serif;
    }

    /* Subheader styling */
    .subheader-text {
        font-size: 1.6rem;
        font-weight: 600;
        color: #2c3e50;
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Info box styling */
    .stAlert {
        background-color: #e8f4fc;
        border-left: 5px solid #3498db;
        border-radius: 8px;
        padding: 15px;
    }

    /* Card styling for image containers */
    .image-card {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        margin-bottom: 20px;
    }

    /* Uploader styling */
    .uploadedFile {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        border: 1px dashed #3498db;
    }

    /* Result styling */
    .result-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 25px;
        text-align: center;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin: 25px 0;
        border: 1px solid #d0d7e2;
    }

    .similarity-score {
        font-size: 3.5rem;
        font-weight: 800;
        color: #2c3e50;
        margin: 15px 0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Button-like styling for uploaders */
    .stFileUploader > div > div {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s;
    }

    .stFileUploader > div > div:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(52, 152, 219, 0.3);
    }

    /* Caption styling */
    .image-caption {
        font-weight: 500;
        color: #555;
        text-align: center;
        margin-top: 8px;
        font-size: 0.95rem;
    }

    /* Status indicators */
    .status-high {
        color: #27ae60;
        font-weight: 600;
        font-size: 1.2rem;
    }

    .status-moderate {
        color: #f39c12;
        font-weight: 600;
        font-size: 1.2rem;
    }

    .status-low {
        color: #e74c3c;
        font-weight: 600;
        font-size: 1.2rem;
    }

    /* Footer note */
    .footer-note {
        text-align: center;
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 30px;
        padding-top: 15px;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


def preprocess_image(image):
    img = np.array(image)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    resized = cv2.resize(gray, (300, 300))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)

    return blurred


def compare_handwriting(img1, img2):
    score, _ = ssim(img1, img2, full=True)
    return score * 100


# Title with custom styling
st.markdown('<h1 class="title-text">✍️ Handwriting Similarity Analyzer</h1>', unsafe_allow_html=True)

# Info box
st.info(
    "⚠️ **Important Note**: This tool compares handwriting images using visual similarity. "
    "For best results, use samples containing the SAME or SIMILAR text. "
    "Different text written by the same person may produce lower scores."
)

# Introduction
st.write("### 📤 Upload two handwriting samples to compare their similarity")

# File uploaders with improved layout
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.markdown("**First Handwriting Sample**")
    img1_file = st.file_uploader(
        "Upload first handwriting image",
        type=["jpg", "png", "jpeg"],
        key="img1",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="image-card">', unsafe_allow_html=True)
    st.markdown("**Second Handwriting Sample**")
    img2_file = st.file_uploader(
        "Upload second handwriting image",
        type=["jpg", "png", "jpeg"],
        key="img2",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# 👇 EVERYTHING MUST BE INSIDE THIS BLOCK
if img1_file and img2_file:
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    processed_img1 = preprocess_image(img1)
    processed_img2 = preprocess_image(img2)
    edges1 = cv2.Canny(processed_img1, 50, 150)
    edges2 = cv2.Canny(processed_img2, 50, 150)

    # Processed Images
    st.markdown('<h3 class="subheader-text">📊 Processed Handwriting Images</h3>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.image(processed_img1, caption="Sample 1 (Processed)", clamp=True)
        st.markdown('<p class="image-caption">Grayscale, resized & blurred</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.image(processed_img2, caption="Sample 2 (Processed)", clamp=True)
        st.markdown('<p class="image-caption">Grayscale, resized & blurred</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Detected Stroke Edges
    st.markdown('<h3 class="subheader-text">🔍 Detected Stroke Edges</h3>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.image(edges1, caption="Edges – Sample 1", clamp=True)
        st.markdown('<p class="image-caption">Canny edge detection applied</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="image-card">', unsafe_allow_html=True)
        st.image(edges2, caption="Edges – Sample 2", clamp=True)
        st.markdown('<p class="image-caption">Canny edge detection applied</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    similarity = compare_handwriting(processed_img1, processed_img2)

    # Results Section
    st.markdown('<h3 class="subheader-text">📈 Analysis Results</h3>', unsafe_allow_html=True)

    # Result box with elegant styling
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown('<h3 style="margin-top: 0;">Handwriting Similarity Score</h3>', unsafe_allow_html=True)
    st.markdown(f'<div class="similarity-score">{similarity:.2f}%</div>', unsafe_allow_html=True)

    # Interpretation based on score
    if similarity > 75:
        st.markdown('<p class="status-high">✅ Very high similarity</p>', unsafe_allow_html=True)
        st.markdown('<p><i>Highly similar handwriting characteristics detected</i></p>', unsafe_allow_html=True)
    elif similarity > 50:
        st.markdown('<p class="status-moderate">🟡 Moderate similarity</p>', unsafe_allow_html=True)
        st.markdown('<p><i>Possible match - similar handwriting features identified</i></p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="status-low">🔴 Low similarity</p>', unsafe_allow_html=True)
        st.markdown('<p><i>Different text or distinct handwriting styles detected</i></p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Explanation note
    with st.expander("📝 How is similarity calculated?"):
        st.write("""
        The similarity score is calculated using **Structural Similarity Index (SSIM)** which compares:
        - **Luminance**: Brightness patterns
        - **Contrast**: Variation in stroke darkness
        - **Structure**: Overall shape and form of handwriting strokes

        The algorithm processes both images to:
        1. Convert to grayscale
        2. Resize to standard dimensions
        3. Apply Gaussian blur to reduce noise
        4. Compare structural patterns

        **Note**: This is a visual similarity comparison, not a forensic handwriting analysis tool.
        """)

# Footer note when no images uploaded
else:
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.markdown("### ⬆️ Upload two images to begin analysis")
    st.markdown(
        '<p style="color: #7f8c8d; font-size: 1rem;">Please upload two handwriting images using the file uploaders above.</p>',
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown(
    '<div class="footer-note">Handwriting Similarity Analyzer | Uses Structural Similarity Index (SSIM) for comparison</div>',
    unsafe_allow_html=True)
