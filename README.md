# Neural Machine Translation with Attention

Este projeto implementa um modelo de tradução automática neural (NMT - Neural Machine Translation) com mecanismo de atenção (Attention) usando TensorFlow. O modelo é treinado para traduzir frases do português para o inglês, utilizando o dataset de TED Talks.

## Setup

Primeiro, faça a importação das bibliotecas necessárias, como TensorFlow, TensorFlow Datasets, NumPy e Matplotlib.

## Dataset

O dataset utilizado é o `ted_hrlr_translate/pt_to_en`, que contém pares de frases em português e inglês extraídos de TED Talks.

## Tokenização

A tokenização converte as frases em uma sequência de tokens, que são processadas pelo modelo. Utilizamos um tokenizador pré-treinado para o português e inglês.

## Modelo

O modelo segue a arquitetura Transformer, com múltiplas camadas de atenção e feed-forward. Ele é composto por:

- Encoder: responsável por processar a sequência de entrada (português).
- Decoder: gera a sequência de saída (inglês), utilizando a sequência processada pelo encoder e os tokens gerados anteriormente.
- Mecanismo de atenção: permite que o modelo foque em diferentes partes da sequência de entrada ao gerar cada token de saída.

## Hiperparâmetros

Alguns dos principais hiperparâmetros do modelo são:

- `num_layers`: número de camadas no encoder e decoder.
- `d_model`: dimensão dos embeddings.
- `num_heads`: número de cabeças no mecanismo de atenção.
- `dff`: número de unidades na camada feed-forward.
- `dropout_rate`: taxa de dropout para regularização.

## Treinamento

O modelo é treinado por 20 épocas, e checkpoints são salvos a cada 5 épocas. O otimizador utilizado é o Adam, com um agendador de taxa de aprendizado personalizado.

### Observação

Durante o treinamento, notei que na minha máquina o tempo de treinamento era muito alto, cerca de 10 minutos por etapa de um único batch. Isso tornou inviável o treinamento localmente. Para contornar isso, utilizei o Google Colab com uma GPU A100, o que acelerou significativamente o processo.

### Outras Percepções

- **Tokenização**: A etapa de tokenização é fundamental para garantir que o modelo entenda a estrutura das frases. A tokenização pré-treinada fornecida pelo TensorFlow simplifica bastante este processo.
- **Atenção**: O mecanismo de atenção é o que diferencia os Transformers dos modelos anteriores de tradução automática. Ele permite que o modelo foque nas partes mais relevantes da frase de entrada ao gerar a tradução.
- **Desempenho**: O uso de GPUs poderosas, como a A100 no Colab, é essencial para treinar modelos complexos como este em um tempo razoável. Treinar em CPUs ou GPUs menos potentes pode ser inviável devido ao tempo necessário.
- **Checkpointing**: O salvamento de checkpoints a cada 5 épocas permitiu que o treinamento fosse retomado facilmente em caso de interrupções, o que foi muito útil ao usar o Colab, onde a conexão pode ser instável.

## Inferência

Após o treinamento, o modelo pode ser utilizado para traduzir novas frases. A classe `Translator` foi implementada para facilitar o processo de inferência. Basta fornecer uma frase em português, e o modelo retorna a tradução em inglês.

## Exemplo de Tradução

Aqui está um exemplo de tradução gerada pelo modelo:

- **Entrada**: "Este é um problema que temos que resolver."
- **Predição**: "this is a problem we have to solve."
- **Verdadeiro**: "this is a problem we have to solve."

## Requisitos

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib
- TensorFlow Datasets

## Como Executar

1. Clone o repositório.
2. Instale os requisitos.
3. Execute o notebook `neural_machine_translation_with_attention.ipynb`.

```bash
git clone https://github.com/eliasbiondo/neural_machine_translation.git
cd neural_machine_translation
pip install -r requirements.txt
jupyter notebook neural_machine_translation_with_attention.ipynb
```

## Conclusão

Este projeto mostra como utilizar a arquitetura Transformer para construir um modelo de tradução automática neural. A implementação inclui tokenização, treinamento com checkpoints, e um pipeline de inferência para gerar traduções de novas frases.

## Referências

- Tutorial oficial do TensorFlow sobre Transformers: [https://www.tensorflow.org/text/tutorials/transformer?hl=en](https://www.tensorflow.org/text/tutorials/transformer?hl=en)
