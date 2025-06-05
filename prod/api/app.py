from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
import io
import base64

app = FastAPI()

# Carga el modelo YOLOv8 nano para segmentación
model = YOLO("yolov8n-seg.pt")

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Guardar la imagen en memoria para pasársela al modelo
    results = model(image)

    masks = []
    for i, result in enumerate(results):
        for mask in result.masks.data.cpu().numpy():
            # Convertir máscara booleana a base64 PNG para enviar como string
            mask_img = Image.fromarray((mask * 255).astype("uint8"))
            buffered = io.BytesIO()
            mask_img.save(buffered, format="PNG")
            mask_b64 = base64.b64encode(buffered.getvalue()).decode()
            masks.append(mask_b64)
            
            mask_img.save(f"mask_{i}.png", format="PNG")


    return JSONResponse(content={"num_masks": len(masks), "masks_base64": masks})
