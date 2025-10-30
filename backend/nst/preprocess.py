import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def preprocess_image(image_path, img_size, device='cpu'):

    image = Image.open(image_path).convert('RGB').resize((img_size, img_size))
    transform = transforms.ToTensor()  
    image = transform(image).unsqueeze(0).to(device)  
    return image

def generate_image(content_image):

    generated_image = content_image.clone()
    noise = torch.rand_like(generated_image) * 0.1
    generated_image = generated_image + noise
    generated_image = clip_0_1(generated_image)
    generated_image.requires_grad_(True)
    return generated_image

def clip_0_1(image):
    
    return torch.clamp(image, 0.0, 1.0)

def tensor_to_image(tensor):
    tensor = tensor.detach().cpu().clamp(0,1)
    tensor = tensor.squeeze(0) 
    tensor = tensor.permute(1,2,0).numpy() 
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)
