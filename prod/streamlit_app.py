import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO
from utils import segment
from style_transfer.utils import apply_style_transfer
import math

st.set_page_config(layout="wide", page_title="Segmentacion y Style Transfer")

st.title("Segmentación con YOLOv8-seg y Transferencia de Estilos")
st.markdown("""
En esta aplicacion podes cargar la imagen que quieras, luego realizar una segmentacion para que se puedan identificar distintos
objetos y formas en ella, y podras elegir el estilo a aplicar a cada una de esas mascaras para luego juntarlas en la imagen original.
No prometemos que quede lindo, pero si que va a quedar original.

Esta aplicación utiliza un modelo (YOLOv8-seg) entrenado con el dataset COCO para realizar segmentacion de objetos.
Dado que se entrenó a las redes con el dataset COCO, solo podrá trabajar con las 80 clases que te dejamos listadas debajo.
""")

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
    num_cols = 4 
    cols = st.columns(num_cols)
    clases_por_col = math.ceil(len(COCO_CLASSES) / num_cols)
    
    for i in range(num_cols):
        with cols[i]:
            for j in range(i * clases_por_col, min((i + 1) * clases_por_col, len(COCO_CLASSES))):
                st.write(f"{j+1}. {COCO_CLASSES[j]}")

uploaded_file = st.file_uploader("Subí una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert("RGB")
    st.image(original_img, caption="Imagen original", width=800)

    os.makedirs("images", exist_ok=True)
    image_path = os.path.join("images", uploaded_file.name)

    if st.button("Segmentar"):
        with st.spinner("Segmentando..."):
            result = segment(original_img, model, image_path)

        if result is not None:
            # Guardamos en el estado de streamlit para que no refresque toda la 
            # pagina cuando elijamos las mascaras u otras cosas
            st.session_state.segment_result = result
            st.session_state.original_img = original_img

            cropped_list = []
            mask_imgs = []

            for mask_img in result["masks_imgs"]:
                # Redimensionamos la mascara a la imagen original, ya que deben superponerse
                mask_img = mask_img.resize(original_img.size)
                # Y recortamos de la imagen original la mascara (pero con el contenido original, a color)
                # usando justamente la mascara segmentada (que venia en blanco y negro, 0s y 255s en modo L que es escala de grises)
                cropped = Image.composite(original_img, Image.new("RGB", original_img.size, (0, 0, 0)), mask_img)
                
                cropped_list.append(cropped)
                mask_imgs.append(mask_img)
            
            # Guardamos en el estado de nuevo para que no se reinicien cada vez que elijamos el estilo de las mascaras
            st.session_state.cropped_list = cropped_list
            st.session_state.mask_imgs = mask_imgs
            st.session_state.style_choices = [""] * len(cropped_list)  # Reset estilos por máscara

# Mostramos máscaras y permitimos selección de estilos por separado
if st.session_state.segment_result is not None:
    st.subheader("Elegí un estilo para cada máscara")

    style_dir = "prod/style_transfer/transforms"
    style_files = [f for f in os.listdir(style_dir) if f.endswith(".pth")]

    if not style_files:
        st.warning("No se encontraron archivos .pth en la carpeta de estilos.")
    else:
        num_masks = len(st.session_state.cropped_list)
        for i in range(0, num_masks, 4):
            cols = st.columns(4)

            for col_idx in range(4):
                idx = i + col_idx
                if idx < num_masks:
                    with cols[col_idx]:
                        clase_idx = st.session_state.segment_result["clases"][idx]
                        clase_nombre = (
                            COCO_CLASSES[clase_idx]
                            if clase_idx < len(COCO_CLASSES)
                            else f"Clase {clase_idx}"
                        )

                        st.image(
                            st.session_state.cropped_list[idx],
                            caption=f"Máscara {idx + 1}: {clase_nombre} (COCO {clase_idx})",
                            use_column_width=True,
                        )
                        st.session_state.style_choices[idx] = st.selectbox(
                            f"Estilo para máscara {idx + 1}",
                            style_files,
                            key=f"style_select_{idx}",
                        )

        if st.button("Aplicar estilos a las máscaras"):
            combined = original_img.copy()
            num_masks = len(st.session_state.cropped_list)

            for i in range(0, num_masks, 4):
                cols = st.columns(4)
                for col_idx in range(4):
                    idx = i + col_idx
                    if idx < num_masks:
                        img = st.session_state.cropped_list[idx]
                        mask_img = st.session_state.mask_imgs[idx]
                        style_file = st.session_state.style_choices[idx]
                        model_path = os.path.join(style_dir, style_file)

                        # Se estilizan las imagenes que tienen las mascaras a color recortadas
                        stylized = apply_style_transfer(img, model_path)
                        stylized = stylized.resize(combined.size)

                        # Mostrar imagen estilizada
                        with cols[col_idx]:
                            st.image(
                                stylized,
                                caption=f"Máscara {idx + 1} con estilo {style_file}",
                                use_column_width=True
                            )

                        # Y listo superponemos la mascara estilizada sobre la imagen original, usando la mascara
                        # blanco y negro como mascara del composite
                        combined = Image.composite(stylized, combined, mask_img)

            st.subheader("Imagen final con estilos aplicados a las máscaras")
            st.image(combined, caption="Resultado", width=800)
