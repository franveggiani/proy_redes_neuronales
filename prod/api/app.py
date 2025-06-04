from fastapi import FastAPI, File, UploadFile, Form
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import numpy as np
import json
import io
import base64

app = FastAPI()

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam.to("cuda")
predictor = SamPredictor(sam)

@app.get("/ur")
def read_root():
    return {"message": "SAM Backend funcionando"}

@app.post("/segment/")
async def segment(
    file: UploadFile = File(...),
    points: str = Form(...),  # JSON string: [[x1, y1], [x2, y2], ...]
    labels: str = Form(...)   # JSON string: [1, 0, 1, ...] (1=foreground, 0=background)
): 
    
    # Adapto la imagen para la red
    image = await file.read()
    image = Image.open(io.BytesIO(image)).convert("RGB") 
    image_np = np.array(image)
    
    predictor.set_image(image_np)
    
    # Adapto los puntos y etiquetas para la red
    input_point = np.array(json.loads(points))
    input_label = np.array(json.loads(labels))
    
    # Segmentar
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False
    )
    
    mask = masks[0].astype(np.uint) * 255
    mask_image = Image.fromarray(mask)
    buffered = io.BytesIO()
    mask_image.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Devuelvo los resultados
    return {
        "mask_b64": mask_base64,
        "score": scores[0].item()
    }
    