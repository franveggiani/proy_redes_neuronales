import streamlit as st
from PIL import Image, ImageOps
import io
import base64
import numpy as np
from ultralytics import YOLO
from utils import segment
from style_transfer.utils import apply_style_transfer
import os

st.title("Segmentación con YOLOv8-seg (FastAPI)")

# Cargar el modelo YOLOv8-seg
model = YOLO("yolov8n-seg.pt")

# Inicializar estado si no existe
if "segment_result" not in st.session_state:
    st.session_state.segment_result = None
    st.session_state.original_img = None
    st.session_state.mask_imgs = []
    st.session_state.cropped_list = []
    st.session_state.combined = None

uploaded_file = st.file_uploader("Subí una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert("RGB")
    st.image(original_img, caption="Imagen original", use_column_width=True)

    # Crear carpeta para guardar imágenes
    os.makedirs("images", exist_ok=True)
    image_path = os.path.join("images", uploaded_file.name)

    if st.button("Segmentar"):
        with st.spinner("Segmentando..."):
            result = segment(original_img, model, image_path)

        if result is not None:
            st.session_state.segment_result = result
            st.session_state.original_img = original_img

            combined = original_img.copy().convert("RGBA")
            cropped_list = []
            mask_imgs = []

            for b64_mask in result["masks_base64"]:
                mask_bytes = base64.b64decode(b64_mask)
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L").resize(original_img.size)
                cropped = Image.composite(original_img, Image.new("RGB", original_img.size, (0, 0, 0)), mask_img)

                rgba_mask = Image.new("RGBA", combined.size, (255, 0, 0, 0))
                red = Image.new("RGBA", combined.size, (255, 0, 0, 100))
                rgba_mask = Image.composite(red, rgba_mask, mask_img)
                combined = Image.alpha_composite(combined, rgba_mask)

                cropped_list.append(cropped)
                mask_imgs.append(mask_img)

            st.session_state.cropped_list = cropped_list
            st.session_state.mask_imgs = mask_imgs
            st.session_state.combined = combined

# Si ya hay una segmentación previa, mostrar resultados y aplicar estilos
if st.session_state.segment_result is not None:
    st.subheader("Máscaras recortadas de la imagen original")
    col1, col2 = st.columns(2)
    for idx, img in enumerate(st.session_state.cropped_list):
        with col1 if idx % 2 == 0 else col2:
            st.image(img, caption=f"Máscara {idx + 1} recortada", use_column_width=True)

    style_dir = "prod/style_transfer/transforms"
    style_files = [f for f in os.listdir(style_dir) if f.endswith(".pth")]

    if style_files:
        selected_style = st.selectbox("Elegí el estilo a aplicar", style_files, key="selected_style_pth")

        if st.button("Aplicar estilo a las máscaras"):
            combined = st.session_state.combined.copy()

            for idx, (img, mask_img) in enumerate(zip(st.session_state.cropped_list, st.session_state.mask_imgs)):
                stylized = apply_style_transfer(img, os.path.join(style_dir, selected_style))
                st.image(stylized, caption=f"Máscara {idx + 1} con estilo", use_column_width=True)

                stylized = stylized.resize(st.session_state.original_img.size).convert("RGBA")
                mask_alpha = mask_img.point(lambda p: 255 if p > 0 else 0)
                stylized_masked = Image.composite(stylized, combined, mask_alpha)
                combined = Image.alpha_composite(combined, stylized_masked)

            st.subheader("Imagen con style transfer aplicado sobre las máscaras")
            st.image(combined, caption="Imagen final", use_column_width=True)
    else:
        st.warning("No se encontraron archivos .pth en la carpeta de estilos.")
