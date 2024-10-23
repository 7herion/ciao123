import timm
import torch
import torchvision.transforms as transforms
from ForwardWrapper import forward_wrapper
import yaml
import numpy as np
from utils import getImageBBoxArea

class VitModel(torch.nn.Module):
    def __init__(self, model_name, num_classes):
        super(VitModel, self).__init__()
        self.modelVit = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        data_config = timm.data.resolve_model_data_config(self.modelVit)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        self.modelVit.blocks[-1].attn.forward = forward_wrapper(self.modelVit.blocks[-1].attn)

    def forward(self, x):
        # torch.cuda.empty_cache()
        return self.modelVit(x)

def load_model(model_path, model_name, num_classes):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    model = VitModel(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(DEVICE).eval()

    return model

def preprocess_image(image_path, img_size, bb_params):
    image = getImageBBoxArea(image_path, bb_params)
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    x = transform(image)
    x = x.unsqueeze(0)
    return x

def main(image_path: str, bb_params: list):
    
    # Define the path to the YAML config file
    config_path = 'config/config.yaml'

    # Load the config file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Extract parameters from the config
    num_classes = config["num_classes"] # 7
    img_size    = config["img_size"]    # 224
    model_path  = config["model_path"]  # modelTRANSFORMER_VIT
    model_name  = config["model_name"]  # vit_base_patch16_224.augreg_in21k

    model = load_model(model_path, model_name, num_classes)
    x = preprocess_image(image_path, img_size, bb_params)

    out = model(x)
    _, preds = torch.max(out, 1)

    classes = ['Domestic Cattle', 'Eurasian Badger', 'European Hare', 'Grey Wolf', 'Red Deer', 'Red Fox', 'Wild Boar']
    predicted_class = classes[preds.cpu().numpy()[0]]
    confidence = round(torch.nn.functional.softmax(out, dim=1).max().item(), 2)
    # da vedere nel caso in cui fallisce la classificazione

    return {'classification': predicted_class, 'confidence': confidence}