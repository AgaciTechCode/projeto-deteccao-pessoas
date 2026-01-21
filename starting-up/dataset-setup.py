import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os, cv2, random
from google.colab.patches import cv2_imshow
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# Registrando os datasets (usando os nomes das pastas que o zip criou)
try:
    register_coco_instances("person_train", {}, "/content/train/_annotations.coco.json", "/content/train")
    register_coco_instances("person_valid", {}, "/content/valid/_annotations.coco.json", "/content/valid")
    register_coco_instances("person_test", {}, "/content/test/_annotations.coco.json", "/content/test")
except:
    print("Datasets já registrados ou erro nos caminhos.")

person_metadata = MetadataCatalog.get("person_train")
