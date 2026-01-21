import cv2
from google.colab.patches import cv2_imshow
from detectron2.utils.visualizer import Visualizer, ColorMode

# 1. Caminho da sua nova imagem
caminho_imagem_nova = "/content/WhatsApp Image 2026-01-18 at 11.06.07 PM.jpeg" # Substitua pelo nome do seu arquivo

# 2. Carregar a imagem com o OpenCV
im = cv2.imread(caminho_imagem_nova)

# 3. Fazer a predição (o modelo vai analisar a imagem)
outputs = predictor(im)

# 4. Filtrar para mostrar apenas a classe 1 (person)
# Isso evita que o modelo mostre a classe 0 (vazia)
instances = outputs["instances"].to("cpu")
person_only = instances[instances.pred_classes == 1]

# 5. Visualizar o resultado
v = Visualizer(im[:, :, ::-1],
               metadata=person_metadata,
               scale=0.8,
               instance_mode=ColorMode.IMAGE_BW)

out = v.draw_instance_predictions(person_only)
cv2_imshow(out.get_image()[:, :, ::-1])
