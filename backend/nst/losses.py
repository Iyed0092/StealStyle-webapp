import torch

STYLE_LAYERS = [
    ('block1_conv1', 1.0),
    ('block2_conv1', 0.8),
    ('block3_conv1', 0.7),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.1)
]

def compute_content_cost(content_output, generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]

    m, n_C, n_H, n_W = a_G.shape
    a_C_unrolled = a_C.view(m, n_C, n_H * n_W)
    a_G_unrolled = a_G.view(m, n_C, n_H * n_W)
    J_content = torch.sum((a_C_unrolled - a_G_unrolled) ** 2) / (4.0 * n_H * n_W * n_C)
    return J_content

def gram_matrix(A):
    return torch.mm(A, A.t())

def compute_layer_style_cost(a_S, a_G):
    m, n_C, n_H, n_W = a_G.shape
    a_S = a_S.view(n_C, -1)
    a_G = a_G.view(n_C, -1)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    J_style_layer = torch.sum((GS - GG) ** 2) / (4.0 * (n_H * n_W * n_C) ** 2)
    return J_style_layer

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    a_S = style_image_output[1:]
    a_G = generated_image_output[1:]
    J_style = 0
    for i, (layer_name, weight) in enumerate(STYLE_LAYERS):
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        J_style += weight * J_style_layer
    return J_style

def total_cost(J_content, J_style, alpha=10, beta=40):
    return alpha * J_content + beta * J_style
