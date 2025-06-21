import io
import base64
from PIL import Image

def segment(image: Image.Image, model, image_path):
    image = image.convert("RGB")
    results = model(image)
    
    # Para guardar las máscaras y clases
    masks = []
    clases = []

    for i, result in enumerate(results):
        # Nos aseguramos de que haya máscaras y clases
        if not result.masks or not result.boxes:
            continue

        mask_data = result.masks.data.cpu().numpy()
        class_indices = result.boxes.cls.cpu().numpy().astype(int)

        for j, mask in enumerate(mask_data):
            # Convertimos la máscara a imagen en escala de grises
            mask_img = Image.fromarray((mask * 255).astype("uint8"))
            masks.append(mask_img)
            clases.append(class_indices[j])

            # Guardar imagen de máscara si se desea
            mask_path = f"{image_path}_mask_{i}_{j}.png"
            mask_img.save(mask_path, format="PNG")

    return {
        "masks_imgs": masks,
        "clases": clases
    }
