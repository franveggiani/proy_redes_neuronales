# style_transfer/utils.py
import torch
from torchvision import transforms
from PIL import Image
from style_transfer.model import TransformerNet

def apply_style_transfer(content_image: Image.Image, model_path: str) -> Image.Image:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    # Preprocesamiento
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    img_tensor = transform(content_image).unsqueeze(0).to(device)

    # Cargar el modelo
    with torch.no_grad():
        style_model = TransformerNet().to(device)
        state_dict = torch.load(model_path)
        style_model.load_state_dict(state_dict, map_location='cpu')
        style_model.eval()

        # Inferencia
        output = style_model(img_tensor).clamp(0, 255).cpu()
    
    # Convertir a imagen PIL
    output = output.squeeze(0).permute(1, 2, 0).numpy().astype("uint8")
    return Image.fromarray(output)
