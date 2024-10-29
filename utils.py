import cv2
from PIL import Image
import yaml


def getImageBBoxArea(image_path: str, bounding_box_coordinates: list) -> Image:
    '''Ritorna la regione dell'immagine descritta dalle coordinate della bounding box'''

    I = cv2.imread(image_path)      # eg. demo_images/img__136.jpg
    b = bounding_box_coordinates    # eg. [0.5697, 0.3929, 0.1525, 0.2355]

    height_start    = int(b[1]*I.shape[0])
    height_end      = int((b[1]+b[3])*I.shape[0])
    width_start     = int(b[0]*I.shape[1])
    width_end       = int((b[0]+b[2])*I.shape[1])

    region_of_interest = I[height_start:height_end, width_start:width_end]

    return Image.fromarray(region_of_interest)
    # return Image.fromarray(I) # Se leggo l'immagine completa diminuisce la percentuale di affidabilità


def getClassificationId(common_name: str) -> int:
    '''Legge l'ID della classificazione dal file YAML di mapping'''

    with open('config/classification_mapping.yaml', 'r') as class_mappings:
        available_classes = yaml.safe_load(class_mappings)
    item_info = available_classes[common_name] # TODO: Potrebbe non esistere -> da gestire

    return item_info['classification_id']