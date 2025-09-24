import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from fpdf import FPDF
import os

# Configure Streamlit page
st.set_page_config(
    page_title="IVF Embryo Score Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2563eb;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e5f3ff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #2563eb;
        margin: 1rem 0;
    }
    .prediction-result {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 2px solid #10b981;
        text-align: center;
        margin: 2rem 0;
    }
    .error-message {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ef4444;
        color: #dc2626;
        margin: 1rem 0;
    }
    .info-box {
        background: #eff6ff;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Load IVF Model + Grad-CAM
# --------------------------
@st.cache_resource
def load_model():
    try:
        # Try multiple possible model paths
        possible_paths = [
            "../models/ivf_embryo_model.h5",
            "models/ivf_embryo_model.h5",
            "ivf_embryo_model.h5",
            "../models/embryo_classifier.h5",
            "models/embryo_classifier.h5",
            "embryo_classifier.h5"
        ]
        
        model = None
        model_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    model = tf.keras.models.load_model(path)
                    model_path = path
                    break
                except Exception as e:
                    continue
        
        if model is None:
            st.error(f"Model file not found. Please place your model file in one of these locations: {possible_paths}")
            return None, None
            
        return model, model_path
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

model, model_path = load_model()

if model is not None:
    # Detect model input shape
    if hasattr(model, "input_shape") and model.input_shape is not None:
        _, H, W, C = model.input_shape
        input_shape = (H or 224, W or 224, C or 3)
    else:
        input_shape = (224, 224, 3)

    # Warm up model (build input/output tensors)
    dummy = np.zeros((1, *input_shape), dtype=np.float32)
    _ = model.predict(dummy)

    # Auto-detect last Conv layer
    last_conv_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
            last_conv_name = layer.name
            break
    
    if not last_conv_name:
        st.warning("No Conv2D layer found in model. Grad-CAM visualization may not work.")

# Grad-CAM heatmap generator
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, eps=1e-8):
    if last_conv_layer_name is None:
        return None
        
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except Exception:
        try:
            x = tf.keras.Input(shape=input_shape)
            y = x
            conv_output = None
            for layer in model.layers:
                y = layer(y)
                if layer.name == last_conv_layer_name:
                    conv_output = y
            grad_model = tf.keras.models.Model(inputs=x, outputs=[conv_output, y])
        except Exception as e:
            st.error(f"Error creating Grad-CAM model: {str(e)}")
            return None

    try:
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if predictions.shape[-1] == 1:
                loss = predictions[:, 0]
            else:
                loss = predictions[:, tf.argmax(predictions[0])]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()

        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + eps)
        return heatmap
    except Exception as e:
        st.error(f"Error generating Grad-CAM: {str(e)}")
        return None

# --------------------------
# Preprocess helper
# --------------------------
def preprocess_image(uploaded_file):
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((input_shape[0], input_shape[1]))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        return img, img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None, None

# --------------------------
# PDF Export helper (with colored table)
# --------------------------
def export_pdf(results_df, filename="results.pdf"):
    try:
        pdf = FPDF("P", "mm", "A4")
        pdf.add_page()
        pdf.set_font("Arial", size=14, style="B")
        pdf.cell(200, 10, "Embryo Quality Assessment Results", ln=True, align="C")
        pdf.ln(10)

        # Page width (usable area)
        page_width = pdf.w - 2 * pdf.l_margin

        # Define column widths dynamically (equal share for all columns)
        col_width = page_width / len(results_df.columns)

        # Table header
        pdf.set_font("Arial", size=12, style="B")
        headers = ["Embryo", "AI Prediction", "Label", "Expansion", "ICM", "TE", "Morph Score"]

        for header in headers:
            pdf.cell(col_width, 10, header, border=1, align="C")
        pdf.ln()

        # Table rows
        pdf.set_font("Arial", size=11)
        for _, row in results_df.iterrows():
            pdf.cell(col_width, 10, str(row["Embryo"]), border=1, align="C")
            pdf.cell(col_width, 10, f"{row['AI Prediction']:.2f}", border=1, align="C")

            # Color label cell
            if "High" in row["Label"]:
                pdf.set_text_color(0, 128, 0)  # Green
                label_text = "High Quality"
            else:
                pdf.set_text_color(220, 20, 60)  # Red
                label_text = "Low Quality"

            pdf.cell(col_width, 10, label_text, border=1, align="C")

            # Reset text color
            pdf.set_text_color(0, 0, 0)

            # Remaining cells
            pdf.cell(col_width, 10, str(row["Expansion"]), border=1, align="C")
            pdf.cell(col_width, 10, str(row["ICM"]), border=1, align="C")
            pdf.cell(col_width, 10, str(row["TE"]), border=1, align="C")
            pdf.cell(col_width, 10, str(row["Morph Score"]), border=1, align="C")
            pdf.ln()

        pdf.output(filename)
        return True
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return False

# --------------------------
# Streamlit UI
# --------------------------

# Sidebar with information
with st.sidebar:
    st.header("📋 How to Use")
    st.markdown("""
    1. **Set Threshold**: Adjust the prediction threshold using the slider
    2. **Upload Images**: Select embryo images (Day 3-5)
    3. **AI Analysis**: Get quality predictions with confidence scores
    4. **Morphological Scoring**: Input expansion, ICM, and TE grades
    5. **View Results**: See comprehensive assessment results
    6. **Export Data**: Download results as CSV or PDF
    """)
    
    st.header("🔬 Model Information")
    if model is not None:
        st.success(f"✅ Model loaded successfully")
        st.info(f"""
        **Model Path**: {model_path}
        **Input Size**: {input_shape[0]}x{input_shape[1]} pixels
        **Architecture**: Sequential CNN
        **Grad-CAM**: {'✅ Available' if last_conv_name else '❌ Not Available'}
        """)
    else:
        st.error("❌ Model not loaded")
    
    st.header("⚠️ Important Notes")
    st.warning("""
    - This is a research prototype
    - For educational purposes only
    - Not for clinical diagnosis
    - Always consult medical professionals
    """)
    
    st.header("📊 Quality Grades")
    st.markdown("""
    **Expansion Stages:**
    - 1-2: Early blastocyst
    - 3-4: Blastocyst
    - 5-6: Expanded/Hatched
    
    **ICM/TE Grades:**
    - A: Excellent
    - B: Good
    - C: Poor
    """)

# Header banner
st.markdown("""
    <div style="background: linear-gradient(90deg, #1E90FF 0%, #6a11cb 50%, #ff7e5f 100%);
                padding: 25px; border-radius: 15px; text-align: center; margin-bottom:30px;">
        <h1 style="color: white; margin: 0; font-size: 36px;">🧬 Embryo Quality Assessment</h1>
        <p style="color: white; font-size:18px; margin-top:8px;">
            AI-Powered IVF Success Prediction
        </p>
    </div>
    """, unsafe_allow_html=True)

# Check if model is loaded
if model is None:
    st.markdown('<div class="error-message">❌ Model could not be loaded. Please check if the model file exists in the models directory.</div>', unsafe_allow_html=True)
    st.markdown("""
    ### 📁 Model File Requirements
    
    Please place your trained model file in one of these locations:
    - `../models/ivf_embryo_model.h5`
    - `../models/embryo_classifier.h5`
    - `models/ivf_embryo_model.h5`
    - `models/embryo_classifier.h5`
    
    The model should be a TensorFlow/Keras model saved in .h5 format.
    """)
    st.stop()

# Threshold section
st.markdown("""
    <h3 style="margin-bottom:0px;">⚙️ Set Prediction Threshold</h3>
""", unsafe_allow_html=True)

threshold = st.slider("", 0.0, 1.0, 0.5, 0.01, key="threshold_slider")

# Centered + larger font threshold value
st.markdown(
    f"<h3 style='text-align:center; font-size:22px; margin-top:0px;'>Current Threshold = {threshold:.2f}</h3>",
    unsafe_allow_html=True
)

# Add spacing after threshold
st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)

# File uploader
uploaded_files = st.file_uploader("📤 Upload Embryo Images (Day 3–5)",
                                  type=["jpg", "jpeg", "png"], accept_multiple_files=True)

results = []

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files[:3]):  # limit 3
        st.subheader(f"Embryo {idx+1}")
        
        # Preprocess image
        img, img_array = preprocess_image(uploaded_file)
        
        if img is None or img_array is None:
            st.error(f"Failed to process image {idx+1}")
            continue

        # Prediction
        try:
            with st.spinner(f"🔄 Analyzing Embryo {idx+1}..."):
                pred_result = model.predict(img_array)
                pred_prob = float(pred_result[0][0])
                pred_label = "High Quality" if pred_prob >= threshold else "Low Quality"
        except Exception as e:
            st.error(f"Error making prediction for Embryo {idx+1}: {str(e)}")
            continue

        # Grad-CAM
        heatmap = None
        overlay = np.array(img)
        
        if last_conv_name:
            try:
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_name)
                if heatmap is not None:
                    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
                    heatmap = np.uint8(255 * heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    overlay = cv2.addWeighted(np.array(img), 0.6, heatmap, 0.4, 0)
            except Exception as e:
                st.warning(f"Could not generate Grad-CAM for Embryo {idx+1}: {str(e)}")

        # Show side by side with reduced & centered images
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
            st.image(img, caption="Original", width=350)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
            if heatmap is not None:
                st.image(overlay, caption="Grad-CAM", width=350)
            else:
                st.image(img, caption="Original (Grad-CAM unavailable)", width=350)
            st.markdown("</div>", unsafe_allow_html=True)

        # Display prediction result
        st.markdown(f"""
        <div class="prediction-result">
            <h3 style="color: {'#10b981' if pred_label == 'High Quality' else '#ef4444'}; margin-bottom: 0.5rem;">
                {pred_label}
            </h3>
            <p style="color: #374151; margin: 0;">
                Confidence: {pred_prob:.2%}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Morphological scoring
        st.markdown("**Morphological Scoring**")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            exp = st.selectbox(f"Expansion (Embryo {idx+1})", [1, 2, 3, 4, 5, 6], key=f"exp_{idx}")
        with col4:
            icm = st.selectbox(f"ICM (Embryo {idx+1})", ["A", "B", "C"], key=f"icm_{idx}")
        with col5:
            te = st.selectbox(f"TE (Embryo {idx+1})", ["A", "B", "C"], key=f"te_{idx}")

        # Convert scores to numeric
        icm_map, te_map = {"A": 3, "B": 2, "C": 1}, {"A": 3, "B": 2, "C": 1}
        morph_score = exp + icm_map[icm] + te_map[te] + (pred_prob * 5)

        results.append({
            "Embryo": f"Embryo {idx+1}",
            "AI Prediction": pred_prob,
            "Label": f"<span style='color: {'green' if pred_label=='High Quality' else 'red'};'>{pred_label}</span>",
            "Expansion": exp,
            "ICM": icm,
            "TE": te,
            "Morph Score": round(morph_score, 2)
        })

        # Add separator between embryos
        if idx < len(uploaded_files[:3]) - 1:
            st.markdown("---")

# --------------------------
# Final summary
# --------------------------
if results:
    # Add spacing before summary
    st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)

    # Wrap the entire summary in a centered container
    st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; width: 80%;'>", unsafe_allow_html=True)

    # Heading
    st.markdown("""
        <h2 style="margin-bottom:20px;">
            📊 Final Results Summary
        </h2>
    """, unsafe_allow_html=True)

    # Results dataframe centered
    results_df = pd.DataFrame(results)

    st.markdown(
        results_df.to_html(index=False, escape=False, justify="center"),
        unsafe_allow_html=True
    )

    # Add extra spacing before buttons
    st.markdown("<div style='margin-top:50px;'>", unsafe_allow_html=True)

    # Export buttons centered
    col1, col2 = st.columns([0.5, 0.5], gap="large")
    with col1:
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "results.csv", "text/csv", use_container_width=True)
    with col2:
        pdf_path = "results.pdf"
        if export_pdf(results_df, pdf_path):
            try:
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download PDF", f, "results.pdf", "application/pdf", use_container_width=True)
            except Exception as e:
                st.error(f"Error reading PDF file: {str(e)}")

    st.markdown("</div></div>", unsafe_allow_html=True)

else:
    # Instructions when no files are uploaded
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("""
    ### 🚀 Get Started
    
    1. **Upload embryo images** using the file uploader above
    2. **Supported formats**: PNG, JPG, JPEG
    3. **Best results**: Clear, well-lit images with good contrast
    4. **Processing time**: Usually takes 2-5 seconds per image
    5. **Limit**: Up to 3 images per session
    
    ### 🎯 What You'll Get
    
    - **AI Quality Prediction**: Confidence score for embryo quality
    - **Morphological Assessment**: Manual scoring for expansion, ICM, and TE
    - **Combined Score**: Integrated assessment combining AI and morphological data
    - **Grad-CAM Visualization**: Heatmap showing AI attention areas
    - **Downloadable Results**: CSV and PDF reports for your records
    """)
    st.markdown('</div>', unsafe_allow_html=True)