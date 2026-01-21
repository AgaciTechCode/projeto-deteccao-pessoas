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

```python
!nvidia-smi
```

---

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
