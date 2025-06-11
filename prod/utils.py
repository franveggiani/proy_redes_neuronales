import io
import base64
from PIL import Image
from fastapi.responses import JSONResponse

def segment_image_sync(image: Image.Image, model, image_path):
    # Asegurarse de que la imagen esté en RGB
    image = image.convert("RGB")
    
    # Ejecutar el modelo sobre la imagen
    results = model(image)

    masks = []
    for i, result in enumerate(results):
        for mask in result.masks.data.cpu().numpy():
            # Convertir la máscara a imagen en escala de grises
            mask_img = Image.fromarray((mask * 255).astype("uint8"))

            # Codificar en base64
            buffered = io.BytesIO()
            mask_img.save(buffered, format="PNG")
            mask_b64 = base64.b64encode(buffered.getvalue()).decode()
            masks.append(mask_b64)
        
            mask_path = f"{image_path}_mask_{i}.png"
            
            # Guardar en disco si se desea
            mask_img.save(mask_path, format="PNG")

    return {"num_masks": len(masks), "masks_base64": masks}
