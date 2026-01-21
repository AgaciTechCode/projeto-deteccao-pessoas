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
#test_model.py

```
## Avaliação de Desempenho com Métricas COCO
Nesta etapa, o sistema realiza a avaliação quantitativa do modelo utilizando o COCOEvaluator para calcular métricas de desempenho sobre o conjunto de teste. Apenas as bounding boxes são avaliadas, evitando problemas com segmentação de máscaras.

O modelo processa todas as imagens de teste e gera métricas como Average Precision (AP) e recall, permitindo verificar a acurácia do detector de pessoas de forma objetiva. Os resultados são exibidos no console e podem ser salvos para análises posteriores.

O código de avaliação pode ser encontrado em `/projeto-deteccao-pessoas/results/metrics/evaluation.py`.
```python
#evaluation.py
```

## Monitoramento em Tempo Real com Câmera
Essa etapa permite monitoramento de vídeo em tempo real usando a câmera do dispositivo. O código inicializa a captura, converte frames em imagens processáveis e exibe a interface de streaming no navegador com sobreposição de informações.

Para cada frame capturado, o modelo detecta pessoas (classe person) e exibe caixas delimitadoras sobre um canvas preto, junto com um contador do número de pessoas detectadas. O loop continua até o usuário interromper o monitoramento, permitindo avaliação dinâmica do modelo em cenários reais, útil para vigilância e sistemas de segurança.

O código do monitoramento por webcam está em `/projeto-deteccao-pessoas/inference/webcam-monitoring.py`.
```python
#webcam-monitoring.py
```


## 4. Exemplos de Resultados

Após o treinamento do modelo, foram realizados testes de inferência sobre imagens não vistas durante o treinamento.  
Os resultados consistem em:

- Detecção visual de pessoas por meio de **bounding boxes**
- Filtragem exclusiva da classe *person*
- Salvamento das imagens processadas na pasta `results/images`
- Avaliação quantitativa utilizando métricas do padrão COCO (precisão, recall e AP), armazenadas em `results/metrics`

Os testes demonstraram que o modelo é capaz de identificar corretamente indivíduos em diferentes cenários, mesmo com variações de iluminação, postura e número de pessoas presentes na cena.

---

## 4. Aplicação em Segurança da Informação

A detecção automática de pessoas possui aplicações diretas e relevantes na área de **Segurança da Informação**, especialmente quando integrada a sistemas de monitoramento físico e ciberfísico. Alguns exemplos incluem:

- **Monitoramento de ambientes sensíveis**, como laboratórios, datacenters e áreas restritas, auxiliando na detecção de acessos não autorizados.
- **Sistemas de vigilância inteligente**, reduzindo a dependência de observação humana contínua e diminuindo a probabilidade de falhas por fadiga ou erro humano.
- **Correlação com eventos de segurança**, onde a presença física pode ser associada a incidentes lógicos, como acessos indevidos a sistemas ou equipamentos.
- **Base para sistemas mais complexos**, como reconhecimento de padrões de comportamento, contagem de pessoas ou detecção de anomalias.

Do ponto de vista da Segurança da Informação, o projeto evidencia como técnicas de Inteligência Artificial podem atuar como **camada complementar de defesa**, ampliando a capacidade de monitora
