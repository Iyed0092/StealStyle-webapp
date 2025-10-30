import torch
from nst.losses import compute_content_cost, compute_style_cost, total_cost
from nst.models import vgg_model_outputs
from nst.preprocess import clip_0_1

def get_optimizer(generated_image, lr=0.03):
    return torch.optim.Adam([generated_image], lr=lr)

def train_step(generated_image, a_S, a_C, optimizer, alpha=10, beta=40):
    optimizer.zero_grad()
    a_G = vgg_model_outputs(generated_image)
    J_style = compute_style_cost(a_S, a_G)
    J_content = compute_content_cost(a_C, a_G)
    J = total_cost(J_content, J_style, alpha=alpha, beta=beta)
    J.backward()
    optimizer.step()
    with torch.no_grad():
        generated_image.copy_(clip_0_1(generated_image))
    return J.item()

