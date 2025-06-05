from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = FastAPI()

# Carga el modelo YOLOv8 nano para segmentación
model = YOLO("yolov8n-seg.pt")

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Guardar la imagen en memoria para pasársela al modelo
    results = model(image)

    masks = []
    for result in results:
        for mask in result.masks.data.cpu().numpy():
            # Convertir máscara booleana a base64 PNG para enviar como string
            mask_img = Image.fromarray((mask * 255).astype("uint8"))
            buffered = io.BytesIO()
            mask_img.save(buffered, format="PNG")
            mask_b64 = base64.b64encode(buffered.getvalue()).decode()
            masks.append(mask_b64)

    return JSONResponse(content={"num_masks": len(masks), "masks_base64": masks})
