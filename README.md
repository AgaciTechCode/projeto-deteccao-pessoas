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
`Esta fase pode ser implementada usando /projeto-deteccao-pessoas/training/train.py. Lembre-se de substituir a classe "person" pelas classes específicas do seu dataset.`
```python

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
