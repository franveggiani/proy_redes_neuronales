import io
import base64
from PIL import Image

def segment(image: Image.Image, model, image_path):
    image = image.convert("RGB")
    results = model(image)
    
    # Guardar máscaras y clases
    masks = []
    clases = []

    for i, result in enumerate(results):
        # Asegurarse de que haya máscaras y clases
        if not result.masks or not result.boxes:
            continue

        mask_data = result.masks.data.cpu().numpy()
        class_indices = result.boxes.cls.cpu().numpy().astype(int)

        for j, mask in enumerate(mask_data):
            # Convertir la máscara a imagen en escala de grises
            mask_img = Image.fromarray((mask * 255).astype("uint8"))

            # Codificar la máscara como base64
            buffered = io.BytesIO()
            mask_img.save(buffered, format="PNG")
            mask_b64 = base64.b64encode(buffered.getvalue()).decode()
            masks.append(mask_b64)

            # Guardar clase correspondiente
            if j < len(class_indices):
                clases.append(class_indices[j])
            else:
                clases.append(-1)  # clase desconocida o fuera de rango

            # Guardar imagen de máscara si se desea
            mask_path = f"{image_path}_mask_{i}_{j}.png"
            mask_img.save(mask_path, format="PNG")

    return {
        "num_masks": len(masks),
        "masks_base64": masks,
        "clases": clases
    }
