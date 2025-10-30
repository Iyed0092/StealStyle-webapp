import torch
import torch.nn as nn
from torchvision import models

vgg = models.vgg16(pretrained=True).features.eval()
for param in vgg.parameters():
    param.requires_grad = False

STYLE_LAYERS = [
    ('0', 1.0),   
    ('5', 0.8),   
    ('10', 0.7),  
    ('17', 0.2),  
    ('24', 0.1)   
]
CONTENT_LAYER = [('29', 1)]  

class VGGOutputs(nn.Module):
    def __init__(self, vgg, style_layers, content_layers):
        super().__init__()
        self.vgg = vgg
        self.style_layers = [int(name) for name, _ in style_layers]
        self.content_layers = [int(name) for name, _ in content_layers]
        self.selected_layers = sorted(self.style_layers + self.content_layers)

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.selected_layers:
                outputs.append(x)
        return outputs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg_model_outputs = VGGOutputs(vgg, STYLE_LAYERS, CONTENT_LAYER).to(device)
