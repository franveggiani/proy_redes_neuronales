import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# --- Configuración ---
SAM_MAX_SIZE = 768  # Tamaño máximo permitido por SAM

# --- Cargar y redimensionar la imagen ---
original_image = Image.open("ejemplo.jpg")

# Redimensionar manteniendo el aspecto
original_width, original_height = original_image.size
scale = SAM_MAX_SIZE / max(original_width, original_height)
new_width = int(original_width * scale)
new_height = int(original_height * scale)
resized_image = original_image.resize((new_width, new_height))

# Canvas interactivo
st.write("Hacé clic en dos puntos sobre la imagen")
canvas_result = st_canvas(
    fill_color="rgba(255, 0, 0, 0.3)",
    stroke_width=3,
    background_image=resized_image,
    update_streamlit=True,
    height=resized_image.height,
    width=resized_image.width,
    drawing_mode="point",
    point_display_radius=5,
    key="canvas",
)

# Mostrar coordenadas
if canvas_result.json_data is not None:
    puntos = []
    for obj in canvas_result.json_data["objects"]:
        x = int(obj["left"])
        y = int(obj["top"])
        puntos.append([x, y])
    st.write("Puntos clickeados:", puntos)