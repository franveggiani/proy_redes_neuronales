import streamlit as st
from PIL import Image
import io
import base64
import os
from ultralytics import YOLO
from utils import segment
from style_transfer.utils import apply_style_transfer

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
    st.session_state.style_choices = []

# Mostrar clases del dataset COCO
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

with st.expander("📚 Ver clases del dataset COCO (80)"):
    st.text("\n".join(f"{i+1}. {name}" for i, name in enumerate(COCO_CLASSES)))

uploaded_file = st.file_uploader("Subí una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert("RGB")
    st.image(original_img, caption="Imagen original", use_column_width=True)

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
            st.session_state.style_choices = [""] * len(cropped_list)  # Reset estilos por máscara

# Mostrar máscaras y permitir selección de estilos por separado
if st.session_state.segment_result is not None:
    st.subheader("Elegí un estilo para cada máscara")

    style_dir = "prod/style_transfer/transforms"
    style_files = [f for f in os.listdir(style_dir) if f.endswith(".pth")]

    if not style_files:
        st.warning("No se encontraron archivos .pth en la carpeta de estilos.")
    else:
        for idx, img in enumerate(st.session_state.cropped_list):
            clase_idx = st.session_state.segment_result["clases"][idx]
            clase_nombre = COCO_CLASSES[clase_idx] if clase_idx < len(COCO_CLASSES) else f"Clase {clase_idx}"
            st.image(img, caption=f"Máscara {idx + 1} recortada: {clase_nombre}", use_column_width=True)
            st.session_state.style_choices[idx] = st.selectbox(
                f"Estilo para máscara {idx + 1}",
                style_files,
                key=f"style_select_{idx}"
            )

        if st.button("Aplicar estilos a las máscaras"):
            combined = st.session_state.combined.copy()

            for idx, (img, mask_img, style_file) in enumerate(zip(
                st.session_state.cropped_list,
                st.session_state.mask_imgs,
                st.session_state.style_choices
            )):
                model_path = os.path.join(style_dir, style_file)
                stylized = apply_style_transfer(img, model_path)
                st.image(stylized, caption=f"Máscara {idx + 1} con estilo {style_file}", use_column_width=True)

                stylized = stylized.resize(st.session_state.original_img.size).convert("RGBA")
                mask_alpha = mask_img.point(lambda p: 255 if p > 0 else 0)
                stylized_masked = Image.composite(stylized, combined, mask_alpha)
                combined = Image.alpha_composite(combined, stylized_masked)

            st.subheader("Imagen final con estilos aplicados a las máscaras")
            st.image(combined, caption="Resultado", use_column_width=True)
