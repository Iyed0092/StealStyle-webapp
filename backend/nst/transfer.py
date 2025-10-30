import torch
from nst.preprocess import preprocess_image, generate_image, tensor_to_image
from nst.train import train_step, get_optimizer
from nst.models import vgg_model_outputs
import os
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stylize_from_paths(content_path, style_path, output_dir=None,
                       content_size=400, style_size=400,
                       epochs=1000, alpha=100, beta=100, save_every=200):
    """
    Perform Neural Style Transfer and save intermediate images every `save_every` epochs.
    Returns a list of absolute file paths of saved images.
    """
  
    if output_dir is None:
        this_file = os.path.abspath(__file__)
        backend_dir = os.path.abspath(os.path.join(os.path.dirname(this_file), ".."))
        output_dir = os.path.join(backend_dir, "uploads", "generated_images")
    os.makedirs(output_dir, exist_ok=True)
    print("Writing generated images to:", output_dir)

    content_image = preprocess_image(content_path, content_size, device)
    style_image = preprocess_image(style_path, style_size, device)

    a_C = vgg_model_outputs(content_image)
    a_S = vgg_model_outputs(style_image)

    generated_image = generate_image(content_image).to(device)
    generated_image.requires_grad_(True)

    optimizer = get_optimizer(generated_image, lr=0.03)

    saved_paths = []

    for i in range(epochs + 1):
        loss = train_step(generated_image, a_S, a_C, optimizer, alpha=alpha, beta=beta)

        if i % save_every == 0 or i == epochs:
            try:
                img = tensor_to_image(generated_image)
                filename = f"generated_{i}.png"
                filepath = os.path.join(output_dir, filename)
                img.save(filepath)
                saved_paths.append(os.path.abspath(filepath))
                print(f"Saved image at epoch {i}: {filepath}")
            except Exception as e:
                print(f"Failed to save image at epoch {i}: {e}")

    return saved_paths
