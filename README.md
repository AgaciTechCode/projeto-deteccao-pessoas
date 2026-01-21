# Projeto de Detecção de Pessoas com Detectron2

## 1. Descrição do Projeto

Este projeto tem como objetivo o desenvolvimento e a avaliação de um sistema de **detecção automática de pessoas** utilizando técnicas de **Visão Computacional** baseadas em **Redes Neurais Convolucionais (CNNs)**.  
Foi empregado o framework **Detectron2**, com o modelo **Faster R-CNN com backbone ResNet-50 e FPN**, treinado a partir de um conjunto de dados anotado no formato **COCO**.

O sistema é capaz de identificar a presença de pessoas em imagens, gerando **caixas delimitadoras (bounding boxes)** e probabilidades associadas às detecções. O foco do projeto está na aplicação prática do modelo, desde o treinamento até a inferência, com ênfase em cenários reais.

---

## 2. Ambiente Utilizado

O desenvolvimento e os testes foram realizados em ambiente computacional com suporte a aceleração por GPU, visando reduzir o tempo de treinamento e inferência.

Principais tecnologias e ferramentas utilizadas:

- **Linguagem:** Python 3
- **Framework de Deep Learning:** PyTorch
- **Framework de Detecção:** Detectron2
- **Modelo Base:** Faster R-CNN (ResNet-50 + FPN)
- **Formato do Dataset:** COCO
- **Bibliotecas auxiliares:** OpenCV, NumPy
- **Ambiente de execução:** Google Colab (com GPU NVIDIA)

A organização do projeto segue uma estrutura modular, separando dados, treinamento, inferência, utilitários e resultados, facilitando manutenção, reprodutibilidade e extensões futuras.

---
## 3. Passo a Passo Para Execução
## Preparando o Ambiente e Instalando o Detectron2
Inicialmente, estando no ambiente de nuvem (Colab), altere o ambiente de execução para GPU. Depois, em uma célula, verifique a existência da GPU:

```python
!nvidia-smi
```

Se bem-sucedida, você verá algo como:
```python
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------|
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   58C    P0             29W /   70W |    5554MiB /  15360MiB |      0%      Default |
+-----------------------------------------------------------------------------------------+
```
Em seguida, adicione o arquivo zipado do seu dataset no formato COCO-like ao diretório /content do ambiente e execute:
```python
!unzip "PESSOA.v1-roboflow-instant-1--eval-.coco.zip"
```
Instale o Detectron2 no ambiente:
```python
!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
## Configuração do Dataset e Treinamento do Modelo
Nesta etapa, o código realiza o registro dos datasets no formato COCO, configurando os conjuntos de treino, validação e teste. Em seguida, prepara o modelo Faster R-CNN usando o Detectron2, definindo parâmetros essenciais como número de classes, taxa de aprendizado, tamanho do batch, número de iterações e pesos iniciais.

O código também cria a pasta de saída para armazenar os resultados e executa o treinamento do modelo, ajustando os pesos para que ele aprenda a detectar pessoas nas imagens do dataset
Esta fase pode ser implementada usando `/projeto-deteccao-pessoas/training/train.py`. Lembre-se de substituir a classe "person" pelas classes específicas do seu dataset.`
```python
#/train.py

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
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("person_train",)
cfg.DATASETS.TEST = ("person_valid",) # Validação durante o treino
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000 # Quantidade boa para 94 fotos
cfg.SOLVER.STEPS = []

# CORREÇÃO: Definindo 2 classes (0: objects, 1: person)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

# Adicionando a correção para o formato das máscaras

cfg.OUTPUT_DIR = "/content/output"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```
## Inferência e Visualização de Detecções do Modelo Treinado
Aqui, o código carrega o modelo treinado e define o limiar mínimo de confiança para considerar uma detecção válida. Em seguida, realiza inferência em imagens de teste, filtrando apenas a classe person para exibir as pessoas detectadas.

Os resultados são visualizados graficamente, com caixas delimitadoras sobrepostas em fundo preto, destacando os objetos detectados. Também é possível testar novas imagens externas, substituindo o caminho da imagem pelo da sua própria foto.
O código desta etapa está em `/projeto-deteccao-pessoas/inference/test_model.py.`
```python
#test_model.p

# Carregar o modelo que acabou de ser treinado
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Confiança mínima
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

dataset_dicts = DatasetCatalog.get("person_test")
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    # Filtramos para mostrar apenas a classe 1 (person)
    instances = outputs["instances"].to("cpu")
    mask = instances.pred_classes == 1
    person_only = instances[mask]

    v = Visualizer(im[:, :, ::-1],
                   metadata=person_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW # Fundo PB destaca a detecção
    )
    out = v.draw_instance_predictions(person_only)
    print(f"Resultado para: {d['file_name']}")
    cv2_imshow(out.get_image()[:, :, ::-1]) 
  
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

```
## Avaliação de Desempenho com Métricas COCO
Nesta etapa, o sistema realiza a avaliação quantitativa do modelo utilizando o COCOEvaluator para calcular métricas de desempenho sobre o conjunto de teste. Apenas as bounding boxes são avaliadas, evitando problemas com segmentação de máscaras.

O modelo processa todas as imagens de teste e gera métricas como Average Precision (AP) e recall, permitindo verificar a acurácia do detector de pessoas de forma objetiva. Os resultados são exibidos no console e podem ser salvos para análises posteriores.

O código de avaliação pode ser encontrado em `/projeto-deteccao-pessoas/results/metrics/evaluation.py`.
```python
#evaluation.py

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Avaliamos apenas BBOX (caixas) para evitar o erro de segmentação que você teve
evaluator = COCOEvaluator("person_test", output_dir="./output", tasks=("bbox",))
val_loader = build_detection_test_loader(cfg, "person_test")

print("--- MÉTRICAS DE DESEMPENHO ---")
results = inference_on_dataset(predictor.model, val_loader, evaluator)
print(results)
```

## Monitoramento em Tempo Real com Câmera
Essa etapa permite monitoramento de vídeo em tempo real usando a câmera do dispositivo. O código inicializa a captura, converte frames em imagens processáveis e exibe a interface de streaming no navegador com sobreposição de informações.

Para cada frame capturado, o modelo detecta pessoas (classe person) e exibe caixas delimitadoras sobre um canvas preto, junto com um contador do número de pessoas detectadas. O loop continua até o usuário interromper o monitoramento, permitindo avaliação dinâmica do modelo em cenários reais, útil para vigilância e sistemas de segurança.

O código do monitoramento por webcam está em `/projeto-deteccao-pessoas/inference/webcam-monitoring.py`.
```python
#webcam-monitoring.py

# --- 1. IMPORTS NECESSÁRIOS (Isso estava faltando) ---
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import numpy as np
import cv2
import io
import PIL.Image
import os

# Certifique-se de que o cfg e o predictor já foram definidos nas células anteriores!
# Se der erro de 'cfg not defined', rode a célula de configuração do modelo antes.

# --- 2. FUNÇÕES DE SUPORTE PARA O VÍDEO ---
def array_to_image(a):
    res = PIL.Image.fromarray(a)
    byte_io = io.BytesIO()
    res.save(byte_io, format='PNG')
    return b64encode(byte_io.getvalue()).decode('ascii')

def video_stream():
  js = Javascript('''
    var video; var div = null; var stream; var captureCanvas; var imgElement; var labelElement;
    var pendingResolve = null; var shutdown = false;

    function removeDom() {
       if (stream) stream.getTracks().forEach(t => t.stop());
       if (video) video.remove();
       if (div) div.remove();
       video = null; div = null; stream = null; imgElement = null; captureCanvas = null; labelElement = null;
    }

    function onAnimationFrame() {
      if (!shutdown) window.requestAnimationFrame(onAnimationFrame);
      if (pendingResolve) {
        var result = "";
        if (!shutdown) {
          captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
          result = captureCanvas.toDataURL('image/jpeg', 0.8)
        }
        var lp = pendingResolve; pendingResolve = null; lp(result);
      }
    }

    async function createDom() {
      div = document.createElement('div');
      div.style.border = '2px solid red'; div.style.padding = '10px'; div.style.width = '660px'; div.style.background = '#000';
      labelElement = document.createElement('div');
      labelElement.innerText = "SISTEMA DE SEGURANÇA ATIVO";
      labelElement.style.color = 'white'; labelElement.style.fontWeight = 'bold';
      div.appendChild(labelElement);
      video = document.createElement('video');
      video.style.display = 'block'; video.width = 640; video.height = 480;
      div.appendChild(video);
      stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}});
      video.srcObject = stream; await video.play();
      captureCanvas = document.createElement('canvas');
      captureCanvas.width = 640; captureCanvas.height = 480;
      imgElement = document.createElement('img');
      imgElement.style.position = 'absolute'; imgElement.style.top = '40px'; imgElement.style.left = '20px';
      imgElement.style.opacity = '0.8'; div.appendChild(imgElement);
      const stopBtn = document.createElement('button');
      stopBtn.textContent = "PARAR MONITORAMENTO";
      stopBtn.onclick = () => { shutdown = true; };
      div.appendChild(stopBtn);
      document.body.appendChild(div);
      window.requestAnimationFrame(onAnimationFrame);
    }

    async function stream_frame(label, imgData) {
      if (shutdown) { removeDom(); shutdown = false; return ""; }
      if (div === null) await createDom();
      if (labelElement) labelElement.innerText = label;
      if (imgData) imgElement.src = imgData;
      return new Promise((resolve) => { pendingResolve = resolve; });
    }
    ''')
  display(js)

# --- 3. LOOP PRINCIPAL ---
# Recarrega os pesos para garantir
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)

video_stream()
label_html = 'Iniciando Câmera...'
bbox_img_data = ''

try:
    while True:
        img_data = eval_js('stream_frame("{}", "{}")'.format(label_html, bbox_img_data))
        if not img_data: break

        binary = b64decode(img_data.split(',')[1])
        img = cv2.imdecode(np.frombuffer(binary, np.uint8), -1)
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        # Filtrar apenas classe 1 (person)
        person_only = instances[instances.pred_classes == 1]

        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        v = Visualizer(canvas, metadata=person_metadata, scale=1.0)
        out = v.draw_instance_predictions(person_only)

        bbox_img_data = 'data:image/png;base64,' + array_to_image(out.get_image())
        label_html = f"SEGURANÇA CAMPUS: {len(person_only)} PESSOA(S) DETECTADA(S)"
except Exception as e:
    print("Monitoramento finalizado.")
```


## 4. Exemplos de Resultados

Após o treinamento do modelo, foram realizados testes de inferência sobre imagens não vistas durante o treinamento.  
Os resultados consistem em:

- Detecção visual de pessoas por meio de **bounding boxes**
- Filtragem exclusiva da classe *person*
- Salvamento das imagens processadas na pasta `results/images`

`exemplo:`
  ![detecção visual de pessoas por bounding boxes](results/images/2.png)
- Avaliação quantitativa utilizando métricas do padrão COCO (precisão, recall e AP), armazenadas em `results/metrics`

`exemplo:`
![avaliação quantitativa utilizando metricas COCO](results/images/metrics-COCO.jpg)

Os testes demonstraram que o modelo é capaz de identificar corretamente indivíduos em diferentes cenários, mesmo com variações de iluminação, postura e número de pessoas presentes na cena.

---

## 5. Aplicação em Segurança da Informação

A detecção automática de pessoas possui aplicações diretas e relevantes na área de **Segurança da Informação**, especialmente quando integrada a sistemas de monitoramento físico e ciberfísico. Alguns exemplos incluem:

- **Monitoramento de ambientes sensíveis**, como laboratórios, datacenters e áreas restritas, auxiliando na detecção de acessos não autorizados.
- **Sistemas de vigilância inteligente**, reduzindo a dependência de observação humana contínua e diminuindo a probabilidade de falhas por fadiga ou erro humano.
- **Correlação com eventos de segurança**, onde a presença física pode ser associada a incidentes lógicos, como acessos indevidos a sistemas ou equipamentos.
- **Base para sistemas mais complexos**, como reconhecimento de padrões de comportamento, contagem de pessoas ou detecção de anomalias.

Do ponto de vista da Segurança da Informação, o projeto evidencia como técnicas de Inteligência Artificial podem atuar como **camada complementar de defesa**, ampliando a capacidade de monitora
