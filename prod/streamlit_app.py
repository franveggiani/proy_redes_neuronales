# streamlit_app.py
import streamlit as st
import requests
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

uploaded_file = st.file_uploader("Subí una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert("RGB")
    st.image(original_img, caption="Imagen original", use_column_width=True)
    
    # Creo carpeta para guardar imágenes
    os.makedirs("images", exist_ok=True)
    image_path = os.path.join("images", uploaded_file.name)

    if st.button("Segmentar"):
        with st.spinner("Segmentando..."):
            # Enviar la imagen a la API
            result = segment(original_img, model, image_path)

        if result is not None:

            num_masks = result["num_masks"]
            st.success(f"{num_masks} máscara(s) detectadas")

            # Crear una copia de la imagen original para superponer máscaras
            combined = original_img.copy().convert("RGBA")
            combined_list = []
            cropped_list = []

            for idx, b64_mask in enumerate(result["masks_base64"]):
                mask_bytes = base64.b64decode(b64_mask)
                mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")

                # Redimensionar máscara al tamaño de la imagen original
                mask_img = mask_img.resize(combined.size)
                
                # Recorto imagen original con la máscara
                cropped = Image.composite(original_img, Image.new("RGB", original_img.size, (0, 0, 0)), mask_img)
                cropped_list.append(cropped)

                # Crear una máscara RGBA: canal alpha es la máscara
                rgba_mask = Image.new("RGBA", combined.size, (255, 0, 0, 0))
                red = Image.new("RGBA", combined.size, (255, 0, 0, 100))  # rojo semi-transparente
                rgba_mask = Image.composite(red, rgba_mask, mask_img)

                # Superponer sobre la imagen original
                combined = Image.alpha_composite(combined, rgba_mask)
                combined_list.append(combined)
                
            # Mostramos mascaras superpuestas en formato de tabla con dos columnas    
            st.subheader("Máscaras superpuestas en la imagen original")
            col1, col2 = st.columns(2)
                
            for idx, img in enumerate(combined_list):
                with col1 if idx % 2 == 0 else col2:
                    st.image(img, caption=f"Máscara {idx + 1} (superpuesta)", use_column_width=True)
                    
            # Mostramos las imagenes recortadas    
            st.subheader("Máscaras recortadas de la imagen original")
            col1, col2 = st.columns(2)
                
            for idx, img in enumerate(cropped_list):
                with col1 if idx % 2 == 0 else col2:
                    st.image(img, caption=f"Máscara {idx + 1} recortada", use_column_width=True)
                    
                    stylized = apply_style_transfer(img, "prod/style_transfer/tokyo_ghoul_aggressive.pth")
                    st.image(stylized, caption=f"Máscara {idx + 1} con estilo", use_column_width=True)
                
                
        else:
            st.error(f"Error al segmentar: {result}")

